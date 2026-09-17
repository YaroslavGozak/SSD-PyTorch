import copy
import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.cost_model_validation.reproducibility import (
    schedule, canonical_hash, statistics, design, fit_models, bootstrap_models, order_estimate,
)
from experiments.cost_model_validation.collect_experiment_a import collect as collect_a
from experiments.cost_model_validation.collect_experiment_b import collect as collect_b
from experiments.cost_model_validation.analyze_experiment_a import analyze as analyze_a
from experiments.cost_model_validation.analyze_experiment_b import analyze as analyze_b
from experiments.cost_model_validation.aggregate_sessions import aggregate
from experiments.cost_model_validation.pair_specs import generate, load, stability
from experiments.cost_model_validation.common import write_json
from experiments.cost_model_validation.adapters import FakeAdapter
from experiments.cost_model_validation.timing import TimingResult


def deterministic_measure(adapter, image, hw, mode):
    prepared = adapter.prepare(image, hw)
    a = prepared.effective_area
    value = 1 + .00001*a + .000005*max(0,a-20000)
    return prepared, TimingResult(0, value, 0, value, None, None)


def config():
    return dict(seed=3, model=dict(backend="fake",stride=32),
                experiment_a=dict(max_requested_hw=[320,320],area_fractions=[.05,.1,.2,.3,.4,.5,.7,1],
                                  repetitions_per_effective_shape=4,global_warmup_iterations=0,prewarm_passes=1,
                                  primary_shape_statistic="trimmed_mean",primary_fit_level="shape_level",
                                  control_every_blocks=1,control_shapes=[[160,160]],breakpoint_min_support=2),
                experiment_b=dict(canvas_hw=[320,320],pair_count=4,repetitions=4,control_shapes=[],control_every_pairs=0,
                                  strata_quotas={"broad_random":1.},boundary_width_ms=0,linear_boundary_width_pixels=0,
                                  max_generation_attempts=30000))


class ReproducibilityTests(unittest.TestCase):
    def test_merge_probability_uses_only_valid_calibration_models(self):
        envelope = dict(min_effective_area=1,max_effective_area=10000,min_tensor_h=1,max_tensor_h=100,
                        min_tensor_w=1,max_tensor_w=100,min_aspect_ratio=.01,max_aspect_ratio=100)
        artifact = dict(calibration_envelope=envelope,latency_models={"linear":dict(coefficients=dict(b0=1,b1=1))},
                        bootstrap=dict(replicates=[{"linear":dict(valid=valid,coefficients=dict(b0=b0,b1=1))}
                                                   for b0,valid in [(10,True),(10,True),(1,True),(100,False)]]))
        report = stability(artifact,[4,4,13],{})["linear"]
        self.assertAlmostEqual(report["merge_probability"],2/3)
        self.assertEqual(report["valid_model_count"],3)
        self.assertFalse(report["decision_stable"])

    def test_unattainable_quota_and_boundary_sides_fail(self):
        envelope = dict(min_effective_area=1024,max_effective_area=102400,min_tensor_h=32,max_tensor_h=320,
                        min_tensor_w=32,max_tensor_w=320,min_aspect_ratio=.1,max_aspect_ratio=10)
        artifact = dict(calibration_envelope=envelope,latency_models={"linear":dict(coefficients=dict(b0=.01,b1=.000001))})
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write_json(root/"fit.json",artifact)
            cfg=config()
            cfg["experiment_b"].update(strata_quotas={"model_disagreement":1.},max_generation_attempts=100)
            with self.assertRaisesRegex(ValueError,"Unattainable"):
                generate(cfg,str(root/"fit.json"),str(root/"pairs.json"))
            cfg["experiment_b"].update(strata_quotas={"linear_boundary":1.},pair_count=1,linear_boundary_width_pixels=10000000)
            with self.assertRaisesRegex(ValueError,"Both sides"):
                generate(cfg,str(root/"fit.json"),str(root/"pairs.json"))
            self.assertFalse((root/"pairs.json").exists())

    def test_boundary_quota_success_and_actual_tensor_domain_validation(self):
        envelope = dict(min_effective_area=4096,max_effective_area=102400,min_tensor_h=64,max_tensor_h=320,
                        min_tensor_w=64,max_tensor_w=320,min_aspect_ratio=.1,max_aspect_ratio=10)
        artifact = dict(calibration_envelope=envelope,latency_models={"linear":dict(coefficients=dict(b0=.01,b1=.000001))})
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write_json(root/"fit.json",artifact)
            cfg=config()
            cfg["experiment_b"].update(strata_quotas={"broad_random":.5,"linear_boundary":.5},pair_count=8,
                                      linear_boundary_width_pixels=2048,max_generation_attempts=50000)
            payload=generate(cfg,str(root/"fit.json"),str(root/"pairs.json"))
            boundary=[p for p in payload["pairs"] if p["primary_stratum"]=="linear_boundary"]
            self.assertEqual(len(boundary),4)
            self.assertEqual(Counter(p["threshold_side"] for p in boundary),{"merge":2,"separate":2})
            self.assertTrue(all(p["domain_result"] for p in payload["pairs"]))
            load(str(root/"pairs.json"),cfg,str(root/"fit.json"),FakeAdapter(),None)
            payload["pairs"][0]["coordinates"]=[[0,0,1,1],[0,0,1,1],[0,0,1,1]]
            write_json(root/"pairs.json",dict(payload=payload,sha256=canonical_hash(payload)))
            with self.assertRaisesRegex(ValueError,"shape/domain"):
                load(str(root/"pairs.json"),cfg,str(root/"fit.json"),FakeAdapter(),None)

    def test_partial_resume_reuses_saved_schedule_without_duplicates(self):
        cfg=config()
        cfg["experiment_a"]["prewarm_passes"]=0
        calls=0

        def interrupted(*args):
            nonlocal calls
            calls+=1
            if calls==8:
                raise RuntimeError("interrupted")
            return deterministic_measure(*args)

        with tempfile.TemporaryDirectory() as folder:
            with patch("experiments.cost_model_validation.collect_experiment_a.measure",side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,"interrupted"):
                    collect_a(cfg,folder)
            root=Path(folder)
            before=(root/"experiment_a_schedule.json").read_bytes()
            with patch("experiments.cost_model_validation.collect_experiment_a.measure",side_effect=deterministic_measure), patch("experiments.cost_model_validation.collect_experiment_a.schedule",side_effect=AssertionError("regenerated")):
                collect_a(cfg,folder)
            self.assertEqual(before,(root/"experiment_a_schedule.json").read_bytes())
            with (root/"experiment_a_raw.csv").open(newline="") as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),len({r["global_position"] for r in rows}))
            self.assertEqual(set(Counter(r["shape_id"] for r in rows).values()),{4})

    def test_schedule_fixed_design_and_seeds(self):
        shapes = [(32,64),(64,96),(96,128),(128,160)]
        items = schedule(shapes,40,7)
        self.assertEqual(set(Counter(r["shape_id"] for r in items).values()),{40})
        orders = []
        for b in range(40):
            order = [r["shape_id"] for r in items if r["block_index"]==b]
            self.assertEqual(len(set(order)),4)
            orders.append(order)
        self.assertTrue(all(a!=b for a,b in zip(orders,orders[1:])))
        self.assertEqual(canonical_hash(items),canonical_hash(schedule(shapes,40,7)))
        self.assertNotEqual(canonical_hash(items),canonical_hash(schedule(shapes,40,8)))

    def test_robust_statistics(self):
        values = list(range(1,41))
        result = statistics(values,.1)
        self.assertEqual(result["trimmed_mean_s"],np.mean(values[4:-4]))
        self.assertEqual(result["mean_s"],20.5)
        self.assertEqual(result["median_s"],20.5)
        self.assertAlmostEqual(result["std_s"],np.std(values,ddof=1))
        self.assertAlmostEqual(result["coefficient_of_variation"],np.std(values,ddof=1)/20.5)

    def test_bootstrap_preserves_design_and_refits_breakpoints(self):
        groups = {(32*i,32):[.01+.000001*1024*i]*10 for i in range(1,9)}
        first = bootstrap_models(groups,8,4,"mean","shape_level",.1,2)
        self.assertEqual(first,bootstrap_models(groups,8,4,"mean","shape_level",.1,2))
        for r in first["replicates"]:
            self.assertAlmostEqual(r["linear"]["coefficients"]["b0"],.01)
            self.assertEqual(r["piecewise"]["breakpoint_area"]%1024,0)
        bad = fit_models(np.arange(1,9)*1024, np.arange(8,0,-1),2)
        self.assertFalse(bad["linear"]["valid"])
        self.assertFalse(bad["piecewise"]["valid"])
        self.assertFalse(fit_models([1,2,3],[1,2,3],2)["piecewise"]["valid"])

    def test_order_adjustment_and_stratified_bootstrap(self):
        rows = [dict(order="merged_first",difference_ms=11)]*2+[dict(order="separate_first",difference_ms=-9)]*8
        result = order_estimate(rows,3,100)
        self.assertEqual(result["estimate"],1)
        self.assertEqual(result["ci"],[1,1])
        self.assertFalse(result["balanced"])
        self.assertEqual(result,order_estimate(rows,3,100))
        with self.assertRaises(ValueError):
            order_estimate(rows[:2])

    def test_end_to_end_resume_pooling_and_frozen_measurement(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as folder, patch("experiments.cost_model_validation.collect_experiment_a.measure",side_effect=deterministic_measure), patch("experiments.cost_model_validation.collect_experiment_b.measure",side_effect=deterministic_measure):
            root = Path(folder)
            paths = []
            for index in range(3):
                directory = root / str(index)
                collect_a(cfg,str(directory))
                original = (directory/"experiment_a_raw.csv").read_bytes()
                schedule_hash = json.loads((directory/"metadata.json").read_text())["schedule_hash"]
                with patch("experiments.cost_model_validation.collect_experiment_a.schedule",side_effect=lambda *a: []):
                    collect_a(cfg,str(directory))
                self.assertEqual(original,(directory/"experiment_a_raw.csv").read_bytes())
                self.assertEqual(schedule_hash,json.loads((directory/"metadata.json").read_text())["schedule_hash"])
                report = analyze_a(str(directory/"experiment_a_raw.csv"),str(directory),10)
                self.assertTrue(report["control_records"])
                self.assertEqual(report["schema_version"],3)
                paths.append(str(directory/"linear_fit.json"))
            pooled_path = root/"pooled.json"
            pooled = aggregate(paths,str(pooled_path),10)
            self.assertTrue(pooled["compatibility_result"])
            self.assertEqual(len(pooled["session_ids"]),3)
            # A pooled model must ignore the already-fitted session coefficients.
            original_artifact=json.loads(Path(paths[0]).read_text())
            changed=copy.deepcopy(original_artifact)
            changed["latency_models"]["linear"]["coefficients"]["b0"]+=100
            write_json(Path(paths[0]),changed)
            refitted=aggregate(paths,str(root/"refitted.json"),2)
            self.assertEqual(refitted["latency_models"],pooled["latency_models"])
            write_json(Path(paths[0]),original_artifact)
            bad = json.loads(Path(paths[1]).read_text())
            bad["provenance"]["device"]="cuda"
            write_json(root/"bad.json",bad)
            with self.assertRaisesRegex(ValueError,"Incompatible"):
                aggregate([paths[0],str(root/"bad.json")],str(root/"unused.json"),2)
            pair_path = root/"pairs.json"
            payload = generate(cfg,str(pooled_path),str(pair_path))
            first_bytes = pair_path.read_bytes()
            generate(cfg,str(pooled_path),str(pair_path),True)
            self.assertEqual(first_bytes,pair_path.read_bytes())
            self.assertTrue(all(p["delta_area"]%1024==0 for p in payload["pairs"]))
            for run in ("b1","b2"):
                with patch("experiments.cost_model_validation.pair_specs.generate",side_effect=AssertionError("Must not generate")):
                    collect_b(cfg,str(pooled_path),str(root/run),pairs_path=str(pair_path))
                result = analyze_b(str(root/run/"experiment_b_raw.csv"),str(root/run))
                self.assertEqual(result["bootstrap_method"]["method"],"stratified_by_order")
                self.assertEqual(result["pairs"],4)
                self.assertTrue((root/run/"decision_metrics.json").exists())
            m1=json.loads((root/"b1"/"experiment_b_metadata.json").read_text())
            m2=json.loads((root/"b2"/"experiment_b_metadata.json").read_text())
            self.assertEqual(m1["pair_specs_hash"],m2["pair_specs_hash"])
            self.assertEqual([p["pair_id"] for p in m1["pair_specs"]["pairs"]],[p["pair_id"] for p in m2["pair_specs"]["pairs"]])
            document = json.loads(pair_path.read_text())
            document["payload"]["pairs"].append(document["payload"]["pairs"][0])
            document["sha256"] = canonical_hash(document["payload"])
            write_json(pair_path,document)
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                load(str(pair_path),cfg,str(pooled_path),FakeAdapter(),None)
            document["sha256"]="broken"
            write_json(pair_path,document)
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                load(str(pair_path),cfg,str(pooled_path),FakeAdapter(),None)

    def test_resume_rejects_changed_schedule_settings(self):
        with tempfile.TemporaryDirectory() as folder, patch("experiments.cost_model_validation.collect_experiment_a.measure",side_effect=deterministic_measure):
            cfg=config()
            collect_a(cfg,folder)
            cfg["experiment_a"]["schedule_seed"]=99
            with self.assertRaisesRegex(ValueError,"mismatch"):
                collect_a(cfg,folder)

    def test_analysis_rejects_missing_designed_shape(self):
        with tempfile.TemporaryDirectory() as folder, patch("experiments.cost_model_validation.collect_experiment_a.measure",side_effect=deterministic_measure):
            cfg=config()
            collect_a(cfg,folder)
            path=Path(folder)/"experiment_a_raw.csv"
            with path.open(newline="") as handle:
                reader=csv.DictReader(handle)
                fields=reader.fieldnames
                rows=list(reader)
            missing=rows[0]["shape_id"]
            with path.open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=fields)
                writer.writeheader()
                writer.writerows(r for r in rows if r["shape_id"] != missing)
            with self.assertRaisesRegex(ValueError,"designed_shape_set"):
                analyze_a(str(path),bootstrap_count=2)


if __name__ == "__main__":
    unittest.main()
