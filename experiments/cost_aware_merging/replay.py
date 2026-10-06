"""Portable pair and frame identities shared across models and machines."""

import hashlib
import json
from pathlib import PurePosixPath


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def frame_key(filename):
    # Ignore the platform-specific dataset root and path separators.
    return "/".join(PurePosixPath(str(filename).replace("\\", "/")).parts[-2:])


def resolve_frames(images_info, manifest):
    wanted = [item["frame_key"] for group in ("calibration", "evaluation") for item in manifest[group]]
    if not manifest["calibration"] or not manifest["evaluation"] or len(wanted) != len(set(wanted)):
        raise ValueError("Replay must contain distinct calibration and evaluation frames")
    wanted_set = set(wanted)
    candidates = {}
    for index, info in enumerate(images_info):
        key = frame_key(info["filename"])
        if key in wanted_set:
            if key in candidates:
                raise ValueError(f"Ambiguous replay frame identity: {key}")
            candidates[key] = index
    missing = wanted_set - candidates.keys()
    if missing:
        raise ValueError(f"Replay frames missing from configured dataset/split: {sorted(missing)}")
    return [candidates[key] for key in wanted]


def validate_pairs(payload, canvas_hw):
    if tuple(payload["canvas_hw"]) != tuple(canvas_hw):
        raise ValueError("Replay pair canvas differs from configured canvas")
    pairs = payload["pairs"]
    if not pairs or len({str(pair["pair_id"]) for pair in pairs}) != len(pairs):
        raise ValueError("Replay pairs must be nonempty and have unique pair IDs")
    height, width = canvas_hw
    evaluation = {item["frame_key"] for item in payload.get("frame_manifest", {}).get("evaluation", [])}
    if payload.get("schema_version", 1) >= 2 and not evaluation:
        raise ValueError("Portable replay requires an evaluation frame manifest")
    for pair in pairs:
        for key in ("r1", "r2"):
            coords = pair[key]
            if len(coords) != 4 or any(type(v) is not int for v in coords):
                raise ValueError(f"Pair {pair['pair_id']}: coordinates must be four integers")
            x1, y1, x2, y2 = coords
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Pair {pair['pair_id']}: invalid rectangle for canvas")
        if evaluation and pair.get("frame_key") not in evaluation:
            raise ValueError(f"Pair {pair['pair_id']}: frame is not in replay evaluation manifest")
    geometry_hash = canonical_hash(dict(canvas_hw=list(canvas_hw), pairs=[
        {key: pair[key] for key in ("pair_id", "r1", "r2")} for pair in pairs]))
    if payload.get("pairs_geometry_sha256", geometry_hash) != geometry_hash:
        raise ValueError("Replay geometry hash mismatch")
    return geometry_hash
