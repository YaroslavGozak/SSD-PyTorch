import argparse
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments.cpu_pilot.runtime_threads import add_thread_arguments, install_thread_policy


class ThreadPolicyTests(unittest.TestCase):
    def parse(self, values):
        parser = argparse.ArgumentParser()
        add_thread_arguments(parser)
        return parser.parse_args(values)

    def test_environment_and_cli_precedence(self):
        with patch.dict(os.environ, {"SSD_CPU_THREADS": "4", "SSD_INTEROP_THREADS": "2", "SSD_OPENCV_THREADS": "3"}):
            args = self.parse(["--cpu-threads", "6"])
        self.assertEqual((args.cpu_threads, args.interop_threads, args.opencv_threads), (6, 2, 3))

    def test_invalid_thread_counts(self):
        for value in ("0", "-1", "1.5"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.parse(["--cpu-threads", value])

    def test_policy_controls_libraries_and_detects_inference_mutation(self):
        state = {"intra": 8, "inter": 1, "cv": 1}
        torch = SimpleNamespace(
            set_num_threads=lambda n: state.update(intra=n),
            set_num_interop_threads=lambda n: state.update(inter=n),
            get_num_threads=lambda: state["intra"],
            get_num_interop_threads=lambda: state["inter"],
        )
        cv = SimpleNamespace(setNumThreads=lambda n: state.update(cv=n), getNumThreads=lambda: state["cv"])
        utils = SimpleNamespace(NUM_THREADS=3)
        limiter = Mock()
        modules = {
            "torch": torch, "cv2": cv,
            "ultralytics.utils": SimpleNamespace(torch_utils=utils),
            "threadpoolctl": SimpleNamespace(threadpool_limits=limiter),
        }
        pilot = SimpleNamespace(run_model_inference=lambda: "ok", _runtime_environment=lambda: {})
        args = SimpleNamespace(cpu_threads=4, interop_threads=2, opencv_threads=3)
        with patch.dict("sys.modules", modules), patch.dict(os.environ):
            install_thread_policy(pilot, args)
            pilot._configure_runtime()
            self.assertEqual(pilot.run_model_inference(), "ok")
            self.assertEqual(pilot._runtime_environment()["blas_threads_limit"], 4)
            self.assertEqual(utils.NUM_THREADS, 4)
            self.assertEqual(pilot.CPU_THREADS, 4)
            limiter.assert_called_once_with(limits=4, user_api="blas")
            state["intra"] = 3
            with self.assertRaisesRegex(RuntimeError, "CPU thread policy changed"):
                pilot.run_model_inference()


if __name__ == "__main__":
    unittest.main()
