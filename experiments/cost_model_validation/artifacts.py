"""Portable, hash-verified references and strict JSON artifact serialization."""
import json
from pathlib import Path

from .common import file_sha256, write_json


def reference(path):
    path = Path(path)
    return {"path": path.name, "sha256": file_sha256(str(path))}


def read_reference(base, ref):
    path = Path(base).parent / ref["path"]
    if file_sha256(str(path)) != ref["sha256"]:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_calibration(path, bootstrap=False):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    for name in ("shape_statistics", "control_records"):
        ref = artifact.get(name+"_reference")
        if ref:
            artifact[name] = read_reference(path, ref)
    if bootstrap and artifact.get("bootstrap_reference"):
        artifact["bootstrap"] = read_reference(path,artifact["bootstrap_reference"])
    return artifact


def export_calibration(output, summary, bootstrap):
    output = Path(output)
    write_json(output / "bootstrap_models.json",bootstrap)
    summary["bootstrap"] = {k:v for k,v in bootstrap.items() if k != "replicates"}
    summary["bootstrap_reference"] = reference(output / "bootstrap_models.json")
    for name in ("shape_statistics", "control_records"):
        if name in summary:
            path = output / (name+".json")
            write_json(path,summary[name])
            summary[name+"_reference"] = reference(path)
    keys = ("schema_version", "artifact_type", "shape_policy", "calibration_envelope", "latency_models",
            "primary_fit_selector", "policy_declaration", "grid_hash", "lookup_coverage", "provenance", "config",
            "bootstrap", "bootstrap_reference", "shape_statistics_reference", "control_records_reference",
            "raw_observations_reference", "raw_observations_sha256", "quality_warnings", "schedule_hash")
    artifact = {k:summary[k] for k in keys if k in summary}
    artifact.update(summary["linear_fit"])
    write_json(output / "linear_fit.json",artifact)
    return artifact
