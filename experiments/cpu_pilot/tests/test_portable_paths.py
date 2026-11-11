import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.cpu_pilot.portable_paths import portable_dataset_config
from experiments.pi_preflight import _validate_manifest_paths


class PortablePathTests(unittest.TestCase):
    def test_portable_dataset_config_rewrites_absolute_windows_splits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "path": r"D:\old\dataset",
                "train": r"D:\old\dataset\images\train",
                "val": "images/val",
                "names": {0: "airplane"},
            }

            result = portable_dataset_config(config, root)

            self.assertEqual(result["path"], str(root.resolve()))
            self.assertEqual(result["train"], "images/train")
            self.assertEqual(result["val"], "images/val")
            self.assertEqual(result["names"], config["names"])
            self.assertEqual(config["path"], r"D:\old\dataset")

    def test_portable_dataset_config_rejects_split_outside_dataset_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            outside = Path(directory) / "outside"
            with self.assertRaisesRegex(ValueError, "(?i)dataset split path"):
                portable_dataset_config({"val": str(outside)}, root)

    def test_preflight_validates_dataset_relative_manifest_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images" / "val" / "frame.jpg"
            label = root / "labels" / "val" / "frame.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            row = {
                "video_id": "video-1",
                "frame_index": "1",
                "image": "images/val/frame.jpg",
                "label": "labels/val/frame.txt",
                "gt_json": "[[0,0.5,0.5,0.2,0.2]]",
            }
            audit = {
                ("video-1", 1): {
                    "image": row["image"],
                    "label": row["label"],
                    "gt": [[0, 0.5, 0.5, 0.2, 0.2]],
                }
            }
            failures = []

            keys = _validate_manifest_paths(
                [row], root, audit, {("video-1", 1): 0}, failures
            )

            self.assertEqual(keys, [("video-1", 1)])
            self.assertEqual(failures, [])
