"""Explicit CPU thread policy shared by portable Stage 3 and Stage 4."""

import argparse
import os
from functools import wraps


def positive_threads(value):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("thread count must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("thread count must be a positive integer")
    return number


def add_thread_arguments(parser):
    for flag, env, default in (
        ("cpu", "SSD_CPU_THREADS", 8),
        ("interop", "SSD_INTEROP_THREADS", 1),
        ("opencv", "SSD_OPENCV_THREADS", 1),
    ):
        parser.add_argument(
            f"--{flag}-threads", type=positive_threads,
            default=os.environ.get(env, str(default)),
            help=f"CPU thread count ({env}; default {default})",
        )


def install_thread_policy(pilot, args):
    """Configure native libraries without modifying the frozen pilot source.

    Call before the pilot's _configure_runtime; inter-op setup must precede work.
    Keep the wrappers for the lifetime of this process.
    """
    import cv2
    import torch
    from ultralytics.utils import torch_utils
    from threadpoolctl import threadpool_limits

    if getattr(pilot, "_explicit_thread_policy", None) is not None:
        raise RuntimeError("CPU thread policy has already been installed")
    policy = (args.cpu_threads, args.interop_threads, args.opencv_threads)
    for value in policy:
        positive_threads(value)
    pilot.CPU_THREADS = args.cpu_threads
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.cpu_threads)
    # Retain the controller: environment variables alone miss already loaded BLAS.
    pilot._native_thread_limiter = threadpool_limits(limits=args.cpu_threads, user_api="blas")
    # Ultralytics select_device otherwise changes intra-op threads on first predict.
    torch_utils.NUM_THREADS = args.cpu_threads

    def configure():
        torch.set_num_threads(args.cpu_threads)
        if torch.get_num_interop_threads() != args.interop_threads:
            torch.set_num_interop_threads(args.interop_threads)
        cv2.setNumThreads(args.opencv_threads)

    def check():
        actual = (torch.get_num_threads(), torch.get_num_interop_threads(), cv2.getNumThreads())
        if actual != policy:
            raise RuntimeError(f"CPU thread policy changed: expected {policy}, observed {actual}")

    original_environment = pilot._runtime_environment

    def environment():
        runtime = original_environment()
        runtime["blas_threads_limit"] = args.cpu_threads
        return runtime

    pilot._runtime_environment = environment
    original_inference = pilot.run_model_inference

    @wraps(original_inference)
    def inference(*call_args, **kwargs):
        check()
        result = original_inference(*call_args, **kwargs)
        check()
        return result

    pilot._configure_runtime = configure
    pilot.run_model_inference = inference
    pilot._explicit_thread_policy = policy
    print(
        f"CPU threads: intra-op={policy[0]}, inter-op={policy[1]}, OpenCV={policy[2]}",
        flush=True,
    )
