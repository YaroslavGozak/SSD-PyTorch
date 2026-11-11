"""Portable local-path resolution for Stage 3 and Stage 4 runs."""

import os
import platform
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional, Union


ROOT = Path(__file__).resolve().parents[2]
PathLike = Union[str, Path]

CHECKPOINT_RELATIVE_PATHS = {
    "yolo26n": Path("trained_models/best-imagenet-vid-yolo/best.pt"),
    "roi_ssd_mobilenet": Path(
        "trained_models/imagenet-vid-roissd-mobilenet-20261027/stage3/best_map.pt"
    ),
}
CHECKPOINT_SHA256 = {
    "yolo26n": "5a35043701d4dadd30319e72246f2e6f349688642b75885bdb4a36b294f66027",
    "roi_ssd_mobilenet": "1a830c339f060905f1042e10e98906be0cab43fa94cb3867bfc683ad14adc804",
}
EXTERNAL_CHECKPOINT_RELATIVE_PATHS = {
    "yolo26n": Path("yolo26n/best.pt"),
    "roi_ssd_mobilenet": Path("roi_ssd_mobilenet/best_map.pt"),
}


def env_path(name: str, default: Optional[PathLike] = None) -> Optional[Path]:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return Path(default).expanduser().resolve() if default is not None else None


def display_path(path: PathLike, root: Path = ROOT) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def cpu_model_fallback() -> str:
    model = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")
    if model.strip():
        return model.strip()
    device_tree = Path("/proc/device-tree/model")
    if device_tree.is_file():
        model = device_tree.read_text(encoding="utf-8", errors="replace").strip("\x00 \r\n")
        if model:
            return model
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware", "processor")):
                _, _, value = line.partition(":")
                if value.strip():
                    return value.strip()
    raise RuntimeError("Unable to determine CPU model from platform or Linux system metadata")


def portable_dataset_config(config: dict, dataset_root: Path) -> dict:
    result = dict(config)
    result["path"] = str(dataset_root.resolve())

    def relative_split(value):
        if isinstance(value, list):
            return [relative_split(item) for item in value]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Dataset split path must be a non-empty string: {value!r}")
        text = value.strip()
        windows_path = PureWindowsPath(text)
        if windows_path.is_absolute():
            parts = list(windows_path.parts)
            image_index = next(
                (index for index, part in enumerate(parts) if part.lower() == "images"),
                None,
            )
            if image_index is None:
                raise ValueError(f"Cannot make absolute dataset split path relative: {text}")
            text = PurePosixPath(*parts[image_index:]).as_posix()
        else:
            path = Path(text)
            if path.is_absolute():
                try:
                    text = path.resolve().relative_to(dataset_root.resolve()).as_posix()
                except ValueError as exc:
                    raise ValueError(
                        f"Dataset split path is outside SSD_DATASET_ROOT: {text}"
                    ) from exc
            else:
                text = text.replace("\\", "/")
                while text.startswith("./"):
                    text = text[2:]
        split_path = PurePosixPath(text)
        if split_path.is_absolute() or ".." in split_path.parts:
            raise ValueError(f"Dataset split path is not dataset-relative: {value!r}")
        return split_path.as_posix()

    for split_name in ("train", "val", "test"):
        if split_name in result and result[split_name] is not None:
            result[split_name] = relative_split(result[split_name])
    return result


def stage3_dir(default: Optional[PathLike] = None) -> Path:
    fallback = Path(default) if default is not None else ROOT / "final_experiments" / "pilot"
    return env_path("SSD_STAGE3_DIR", fallback)


def dataset_yaml_path(configured_path: PathLike, override: Optional[PathLike] = None) -> Path:
    selected = override or os.environ.get("SSD_DATASET_YAML") or configured_path
    return Path(selected).expanduser().resolve()


def dataset_root_path(default: Optional[PathLike] = None) -> Optional[Path]:
    return env_path("SSD_DATASET_ROOT", default)


def checkpoint_path(model_name: str) -> Path:
    if model_name not in CHECKPOINT_RELATIVE_PATHS:
        raise ValueError(f"No frozen checkpoint is defined for model {model_name!r}")
    models_root = env_path("SSD_MODELS_ROOT")
    if models_root is not None:
        return (models_root / EXTERNAL_CHECKPOINT_RELATIVE_PATHS[model_name]).resolve()
    return (ROOT / CHECKPOINT_RELATIVE_PATHS[model_name]).resolve()
