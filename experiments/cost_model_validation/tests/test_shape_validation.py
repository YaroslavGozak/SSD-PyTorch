import copy
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from experiments.cost_model_validation.geometry import stride_rounded_shape
from experiments.cost_model_validation.shape_model import (
    calibration_grid, ShapeLookupLatencyModel, MergePolicy, policy_declaration,
)
from experiments.cost_model_validation.models import (
    CalibrationEnvelope, PolynomialLatencyModel, decide_merge, piecewise_area_limit, validate_monotonic,
)
from experiments.cost_model_validation.validation_design import (
    practical_label, control_diagnostics, check_freshness, mark_used, validation_declaration,
)
from experiments.cost_model_validation.reproducibility import canonical_hash, gate
from experiments.cost_model_validation.artifacts import load_calibration, read_reference
from experiments.cost_model_validation.common import write_json, load_config
from experiments.cost_model_validation.collect_experiment_a import collect as collect_a
from experiments.cost_model_validation.analyze_experiment_a import analyze as analyze_a
from experiments.cost_model_validation.collect_experiment_b import collect as collect_b
from experiments.cost_model_validation.analyze_experiment_b import analyze as analyze_b, weighted_metrics
from experiments.cost_model_validation.pair_specs import classify, generate, load
from experiments.cost_model_validation.adapters import FakeAdapter
from experiments.cost_model_validation.timing import TimingResult


def fixture():
    envelope = dict(min_tensor_h=32,max_tensor_h=128,min_tensor_w=32,max_tensor_w=128,
                    min_effective_area=1024,max_effective_area=16384,min_aspect_ratio=.25,max_aspect_ratio=4.)
    models = {name:dict(coefficients=dict(b0=.001,b1=.0000001,**({"b2":0.} if name!="linear" else {})),
                        **({"breakpoint_area":4096} if name=="piecewise" else {})) for name in ("linear","quadratic","piecewise")}
    table = {f"{h}x{w}":dict(point_s=.001+.0000001*h*w+.000001*h,standard_error_s=.00001,ci_s=[.001,.01],repetitions=40)
             for h,w in calibration_grid(envelope,32)}
    models["shape_lookup"] = dict(table=table)
    return dict(schema_version=4,shape_policy=dict(stride=32,name="ceil_to_stride",numeric_dtype="float32"),
                latency_models=models,calibration_envelope=envelope,lookup_coverage=dict(complete=True),
                policy_declaration=policy_declaration({}),control_records=[])


def measured(adapter,image,hw,mode):
    prepared=adapter.prepare(image,hw)
    ms=1+.0001*prepared.effective_area+.001*prepared.tensor_h
    return prepared,TimingResult(0,ms,0,ms,None,None)


class ShapeValidationTests(unittest.TestCase):
    def test_lookup_boundary_scores_and_linear_conservative_disagreement(self):
        artifact=fixture()
        for entry in artifact["latency_models"]["shape_lookup"]["table"].values():
            entry.update(point_s=.001,standard_error_s=.001)
        models={name:PolynomialLatencyModel(name,data["coefficients"],CalibrationEnvelope(**artifact["calibration_envelope"]),data.get("breakpoint_area"))
                for name,data in artifact["latency_models"].items() if name != "shape_lookup"}
        lookup=ShapeLookupLatencyModel(artifact["latency_models"]["shape_lookup"],artifact["calibration_envelope"],32)
        options=dict(strata_quotas={"linear_tau_shape_lookup_conservative_disagreement":1.},boundary_width_ms=0)
        primary,tags,gains,_,scores=classify([(64,64)]*3,models,options,lookup,policy_declaration({}))
        self.assertEqual(primary,"linear_tau_shape_lookup_conservative_disagreement")
        self.assertIn(primary,tags)
        self.assertGreater(gains["linear"],0)
        self.assertLess(gains["shape_lookup_conservative"],0)
        self.assertEqual(scores["shape_lookup_conservative"],gains["shape_lookup_conservative"])

    def test_canonical_alignment_and_grid(self):
        self.assertEqual(stride_rounded_shape(161,319,32),(192,320))
        self.assertNotEqual(161*319,192*320)
        cfg=load_config("experiments/cost_model_validation/config.voc-yolo26n.yaml")
        grid=calibration_grid(cfg["experiment_a"]["calibration_envelope"],32)
        self.assertEqual(len(grid),200)
        self.assertEqual(canonical_hash(grid),canonical_hash(calibration_grid(cfg["experiment_a"]["calibration_envelope"],32)))
        self.assertTrue(all(h%32==w%32==0 for h,w in grid))

    def test_ordered_lookup_exact_and_missing(self):
        artifact=fixture()
        lookup=ShapeLookupLatencyModel(artifact["latency_models"]["shape_lookup"],artifact["calibration_envelope"])
        self.assertNotEqual(lookup.predict_seconds(32,64),lookup.predict_seconds(64,32))
        self.assertEqual(lookup.predict_seconds(32,64),artifact["latency_models"]["shape_lookup"]["table"]["32x64"]["point_s"])
        with self.assertRaisesRegex(ValueError,"Missing lookup"):
            lookup.predict_seconds(33,64)
        del artifact["latency_models"]["shape_lookup"]["table"]["32x64"]
        with self.assertRaisesRegex(ValueError,"Missing lookup"):
            MergePolicy(artifact).predict([(32,64),(64,32),(64,64)],"shape_lookup")

    def test_ties_margin_fallback_and_no_runtime_bootstrap(self):
        artifact=fixture()
        for entry in artifact["latency_models"]["shape_lookup"]["table"].values():
            entry.update(point_s=.01,standard_error_s=.001)
        artifact["latency_models"]["shape_lookup"]["table"]["128x128"]["point_s"]=.02
        policy=MergePolicy(artifact)
        self.assertFalse(policy.predict([(64,64),(64,64),(128,128)],"shape_lookup")["predicted_merge"])
        self.assertFalse(policy.predict([(64,64),(64,64),(128,128)],"shape_lookup_conservative")["predicted_merge"])
        declaration=policy_declaration({"policies":dict(decision_margin_s=.1,fallback="separate")})
        policy=MergePolicy(artifact,declaration)
        self.assertFalse(policy.predict([(64,64)]*3,"shape_lookup")["predicted_merge"])
        with patch("experiments.cost_model_validation.reproducibility.bootstrap_models",side_effect=AssertionError("runtime bootstrap")):
            result=policy.predict([(640,640)]*3,"shape_lookup",strict=False)
        self.assertFalse(result["predicted_merge"])
        self.assertEqual(policy.fallback_counts["separate"],1)
        self.assertEqual(policy.declaration["primary"],"linear_tau")

    def test_production_merger_uses_ordered_shapes(self):
        from tools.mergers.calibrated import CalibratedRoiMerger
        artifact=fixture()
        for entry in artifact["latency_models"]["shape_lookup"]["table"].values():
            entry["point_s"]=.001
        table=artifact["latency_models"]["shape_lookup"]["table"]
        table["32x128"]["point_s"]=.0015
        table["128x32"]["point_s"]=.003
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"fit.json"
            write_json(path,artifact)
            merger=CalibratedRoiMerger(str(path),dict(primary="shape_lookup"))
            self.assertEqual(len(merger([(0,0,64,32),(64,0,128,32)])),1)
            self.assertEqual(len(merger([(0,0,32,64),(0,64,32,128)])),2)
            self.assertEqual(CalibratedRoiMerger(str(path)).metadata()["policy_declaration"]["primary"],"linear_tau")

    def test_piecewise_limit_both_branches_and_monotonicity(self):
        env=CalibrationEnvelope(1,100,1,100,1,100,.01,100)
        model=PolynomialLatencyModel("piecewise",dict(b0=.1,b1=.2,b2=.1),env,20)
        for a1,a2 in [(2,3),(20,30)]:
            limit=piecewise_area_limit(a1,a2,model)
            for merged in range(1,100):
                self.assertEqual(decide_merge(model,(1,a1),(1,a2),(1,merged))["predicted_merge"],merged<limit)
        with self.assertRaisesRegex(ValueError,"strictly increasing"):
            validate_monotonic(PolynomialLatencyModel("quadratic",dict(b0=1,b1=1,b2=-.01),env))
        with self.assertRaisesRegex(ValueError,"strictly increasing"):
            piecewise_area_limit(2,3,PolynomialLatencyModel("piecewise",dict(b0=1,b1=1,b2=-2),env,20))

    def test_control_shift_is_observed_not_model_residual(self):
        artifact=fixture()
        artifact["control_records"]=[dict(tensor_h=64,tensor_w=64,measured_ms=10)]*3
        records=[dict(tensor_h=64,tensor_w=64,measured_ms=10,elapsed_s=i) for i in range(9)]
        report=control_diagnostics(artifact,records,{})
        self.assertFalse(report["hardware_state_shift"])
        self.assertFalse(report["within_session_drift_detected"])
        for r in records:r["measured_ms"]=12
        report=control_diagnostics(artifact,records,{})
        self.assertTrue(report["hardware_state_shift"])
        self.assertFalse(report["within_session_drift_detected"])
        warnings=[]
        for _ in range(2):gate(False,"shift",dict(quality_gates=dict(default="warn")),warnings)
        self.assertEqual(len(warnings),1)

    def test_practical_labels_distinct_from_significance(self):
        self.assertEqual(practical_label([-.1,.1],.2),"practically_tied")
        self.assertEqual(practical_label([.1,.3],.2),"uncertain")
        self.assertEqual(practical_label([.3,.4],.2),"confident_merge")
        self.assertEqual(practical_label([-.4,-.3],.2),"confident_separate")

    def test_freshness_guard_requires_explicit_override(self):
        cfg=dict(experiment_b=dict(confirmatory=True))
        payload=dict(schema_version=2,validation_declaration=validation_declaration(cfg))
        doc=dict(payload=payload,sha256=canonical_hash(payload))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"pairs.json"
            self.assertEqual(check_freshness(path,doc,cfg),[])
            mark_used(path,doc["sha256"],folder)
            with self.assertRaisesRegex(ValueError,"already used"):
                check_freshness(path,doc,cfg)
            self.assertIn("NON-CONFIRMATORY",check_freshness(path,doc,cfg,True)[0])
        doc["payload"]["used_for_policy_development"]=True
        with self.assertRaisesRegex(ValueError,"policy development"):
            check_freshness("missing.json",doc,cfg)

    def test_external_weights_not_observed_labels(self):
        from experiments.cost_model_validation.analyze_experiment_b import RULES
        rows=[]
        for name,difference in [("easy",1.),("hard",-10.)]:
            rows.append(dict(primary_stratum=name,predictions={r:dict(predicted_merge=True) for r,*_ in RULES},
                             measurements=dict(merged_mean_ms=20-difference,separate_mean_ms=20,mean_difference_ms=difference)))
        declaration=dict(source="external_reference_trace",strata=dict(easy=.9,hard=.1))
        result=weighted_metrics(rows,declaration)
        self.assertAlmostEqual(result["metrics"]["linear_tau"]["mean_regret_ms"],1.)
        self.assertAlmostEqual(result["metrics"]["linear_tau"]["expected_latency_gain_ms"],-.1)
        with self.assertRaises(ValueError):
            weighted_metrics(rows,{**declaration,"derived_from_labels":True})

    def test_fresh_full_grid_smoke_both_designs_and_compact_artifacts(self):
        cfg=dict(seed=89,model=dict(backend="fake",stride=32),policies=dict(primary="linear_tau",confidence_level=.95,decision_margin_s=0.),
                 experiment_a=dict(shape_design="full_grid",calibration_envelope=fixture()["calibration_envelope"],max_requested_hw=[128,128],
                                   repetitions_per_effective_shape=4,global_warmup_iterations=0,prewarm_passes=0,control_shapes=[[64,64]],control_every_blocks=1),
                 experiment_b=dict(canvas_hw=[128,128],pair_count=4,repetitions=4,challenge_repetitions=8,control_every_pairs=1,control_shapes=[[64,64]],
                                   strict_lookup=True,confirmatory=True,strata_quotas=dict(broad_random=1.),boundary_width_ms=0,linear_boundary_width_pixels=0),
                 quality_gates=dict(default="fail",control_drift="warn"))
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()),patch("experiments.cost_model_validation.collect_experiment_a.measure",side_effect=measured),patch("experiments.cost_model_validation.collect_experiment_b.measure",side_effect=measured):
            root=Path(folder)
            collect_a(cfg,str(root/"a"))
            report=analyze_a(str(root/"a"/"experiment_a_raw.csv"),str(root/"a"),12)
            fit=root/"a"/"linear_fit.json"
            artifact=load_calibration(fit)
            self.assertTrue(artifact["lookup_coverage"]["complete"])
            self.assertNotIn("replicates",artifact["bootstrap"])
            self.assertNotIn("replicates",json.loads((root/"a"/"experiment_a_summary.json").read_text())["bootstrap"])
            for design in ("representative","challenge"):
                cfg["experiment_b"].update(evaluation_design=design,generator_seed=90 if design=="representative" else 91)
                pairs=root/(design+"_pairs.json")
                payload=generate(cfg,str(fit),str(pairs))
                self.assertEqual(payload["evaluation_design"],design)
                if design=="representative":self.assertEqual(payload["quotas"],{"reference":4})
                collect_b(cfg,str(fit),str(root/design),pairs_path=str(pairs))
                result=analyze_b(str(root/design/"experiment_b_raw.csv"),str(root/design))
                self.assertEqual(result["evaluation_design"],design)
                self.assertEqual(result["aggregate_scope"],"reference_distribution" if design=="representative" else "unweighted_challenge_average")
                self.assertIn("metrics_by_stratum_and_boundary_side",result)
                self.assertFalse(result["fallback_counts"])
                metadata=json.loads((root/design/"experiment_b_metadata.json").read_text())
                self.assertNotIn("bootstrap",metadata)
                self.assertNotIn("near_pair_count",metadata)
                self.assertNotIn("raw_observations",result["pair_summaries"][0])
                self.assertEqual(len(result["quality_warnings"]),len(set(result["quality_warnings"])))
                with self.assertRaisesRegex(ValueError,"already used"):
                    collect_b(cfg,str(fit),str(root/(design+"_repeat")),pairs_path=str(pairs))
                frozen=root/design/"pair_specs.json"
                doc=json.loads(frozen.read_text())
                doc["payload"]["pairs"][0]["delta_area"]+=1
                write_json(frozen,doc)
                with self.assertRaisesRegex(ValueError,"hash mismatch"):
                    analyze_b(str(root/design/"experiment_b_raw.csv"))


if __name__=="__main__":
    unittest.main()
