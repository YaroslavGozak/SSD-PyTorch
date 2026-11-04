# CPU revalidation experiment manifest

PREPARED_NOT_BENCHMARKED; no measured results.

Code commit: `c5cdb772c9f576b4f0f3313cac4a4d8627ba9bc9`; working tree includes uncommitted changes, source hashes below.
Python: `H:\Projects\University\SSD-PyTorch\.venv\Scripts\python.exe`. Device CPU; torch intra/inter-op=8/1; OpenCV=1.
Environment: `{"CUDA_VISIBLE_DEVICES": "", "SSD_INFERENCE_DEVICE": "cpu", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE", "PYTHONHASHSEED": "0"}`.

Dataset: canonical sampled ImageNet-VID. Accuracy loops use the full frozen 5,064-frame val_manifest.jsonl;
canonical_validation.jsonl (256 selection frames) is retained as provenance, not substituted for the full split.
Pilot/final candidate are copied unchanged from final_experiments/cpu_pilot; video overlap=0.
30 foreground classes in the exact FINAL_EXPERIMENTS_MANIFEST.md order; 27 supported by GT.
SSD: 31 outputs, background 0, foreground i -> i+1. YOLO: no background; adapter adds 1.

Full-frame accuracy: all 5,064 validation frames, full scene resized to the configured 320x320 input; no ROI crop.
All quality evaluations: conf_floor=0.01 for all models; existing AP confidence ranking is unchanged.
Operational confidence: maximum micro F1@IoU=0.5 on full-frame pilot; ties: higher recall, then lower threshold; candidate grid (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95).
Pilot thresholds remain UNSELECTED until --stage confidence; subsequent stages require frozen pilot decisions.
Accuracy padding: 0,10,20,...,200. Fixed H/W: 32,64,96,128,160,192,224,256,288,320 (100 pairs).
Pilot padding: 32,64,96,112,128,144,160. Static, hold=0, interval=10.
Cost shapes: 32x32,64x64,96x96,128x128,192x192,256x256,320x320,64x128,128x64,160x320,320x160;
two warmups and eight repetitions/shape. Fit uses observed network tensor areas.

| Model | Checkpoint | SHA-256 | Config | Confidence | Config NMS IoU (effective differences: RUNBOOK) |
|---|---|---|---|---:|---:|
| yolo26n | trained_models/best-imagenet-vid-yolo/best.pt | 5a35043701d4dadd30319e72246f2e6f349688642b75885bdb4a36b294f66027 | final_experiments/cpu_revalidation_20261102/configs/yolo26n.yaml | 0.01 (quality); pilot-selected (operational) | 0.45 |
| roi_ssd_vgg16 | trained_models/imagenet-vid-roissd-20261028/stage3/best_map.pt | f8f49132e08a7c5bf3744ce32e46e1b0fed6595aabd9cf59cf68699569503f76 | final_experiments/cpu_revalidation_20261102/configs/roi_ssd_vgg16.yaml | 0.01 (quality); pilot-selected (operational) | 0.45 |
| roi_ssd_mobilenet | trained_models/imagenet-vid-roissd-mobilenet-20261027/stage3/best_map.pt | 1a830c339f060905f1042e10e98906be0cab43fa94cb3867bfc683ad14adc804 | final_experiments/cpu_revalidation_20261102/configs/roi_ssd_mobilenet.yaml | 0.01 (quality); pilot-selected (operational) | 0.45 |

Class order (zero-based foreground): 0: airplane, 1: antelope, 2: bear, 3: bicycle, 4: bird, 5: bus, 6: car, 7: cattle, 8: dog, 9: domestic_cat, 10: elephant, 11: fox, 12: giant_panda, 13: hamster, 14: horse, 15: lion, 16: lizard, 17: monkey, 18: motorcycle, 19: rabbit, 20: red_panda, 21: sheep, 22: snake, 23: squirrel, 24: tiger, 25: train, 26: turtle, 27: watercraft, 28: whale, 29: zebra

Separate ROI-SSD/VGG16 full-validation confidence diagnostic: (0.01, 0.05, 0.1, 0.25, 0.5); not operational tuning.

## Locked files (SHA-256)

- `final_experiments/cpu_revalidation_20261102/manifests/canonical_validation.jsonl`: `59c38606d753c5679297c7f7e932fc5e69c9f9dd0375d24ab5e5eed860869127`
- `final_experiments/cpu_revalidation_20261102/manifests/final_eval_manifest_candidate.txt`: `4f36f3455ced1a344ef69329479cd09013934d71aafa8dc800d7f7e5ea6bf6d4`
- `final_experiments/cpu_revalidation_20261102/manifests/pilot_manifest.txt`: `817f5b145b774dc91ee0fa8d271e02c11d71d70bb50ff13082a0ba9487aa63cb`
- `final_experiments/cpu_revalidation_20261102/manifests/val_manifest.jsonl`: `a375ebf0a2fb4c4aaac8cb810f5fef0d3579ad3f283484c536bb8904e89b1ac0`
- `tools/infer.py`: `060f56e53592727699ad68a934b7ff31841a403806a4e9e247d91a219626152f`
- `tools/helpers/pipeline.py`: `bab8b0c6e0a78c53befef5e95be5dccf3f51f72d1e2809c077373ee75baad1da`
- `model/model_adapters.py`: `06b7698996e14145aebe8a67629f8b2f9aaad5364124ab86794ca14adebb8b14`
- `model/roissd.py`: `97e720a36dbb7414b98d16e35bc1c8a921277a79a01eb6c30663993291a2020c`
- `model/roissd_mobilenet.py`: `c554fc8be4fe3949444485a0bfbfb36f234d3c984979f78afa1c6f6b7aaa8aeb`
- `experiments/cpu_pilot/run_pilot.py`: `022011bc446c90f67892190573af6f59da1c5ad4360b01b44941a09936b6b5f3`
- `experiments/cpu_revalidation.py`: `269ba9adcf9302eba76d1b837e8ec9e43a353f6ed7d6305f3f49b18f08201fe2`
- `experiments/cpu_pilot/confidence.py`: `3bd8be84c12982729604c5fa2ce2ebdec13fb295457b77a72e446e586637c0c7`
- `experiments/cpu_pilot/oracle_diagnosis.py`: `902c7ebe7389e39e739a7764be6fa4a842c965f4611fdf2ef09a3cc0389e969f`
- `dataset/transforms/fixed_size_test_transform.py`: `57a2e2004d0392869db6a1e45e16c6c9ee5025852c71eac2f75011fc9c935fb4`
- `dataset/transforms/fixed_padding_roi_crop_test_transform.py`: `09932f8ad91fb166b6048b458ef457149d2742742052b328882b4a20ef8abc82`
- `dataset/transforms/fixed_padding_roi_crop_yolo_test_transform.py`: `796ad2f6cc9fb40c411b1382b4e81eada2a8208edfdfcabb0da60c85c5fa142b`
- `transformers/fixed_roi_crop.py`: `51616bc9ee709f8aa25f0152af27ef0c15b89bfc3ae759cfca2b26244aae4c11`
- `tools/benchmarks/video_artifacts.py`: `75e402e9605178ecdc94de8b8032fe3151b81f93ff1eebf8735130695367768d`
- `final_experiments/FINAL_EXPERIMENTS_MANIFEST.md`: `73cb9c6f066924d66c54116e05d8eafaee4ba7eea144fdfba573e0a73088eeb3`
- `final_experiments/01_audit/dataset_audit.json`: `f3d2cdcd6f1e1a8342396817a39ed05ca4237f38645bca011d9b303c66ef95ec`
- `final_experiments/cpu_revalidation_20261102/configs/yolo26n.yaml`: `3451200111db73c00193d16b3d5a1a25951114d7a887998ad8f5736f4fda212a`
- `final_experiments/cpu_revalidation_20261102/configs/roi_ssd_vgg16.yaml`: `484db5459e3e50b5bdac2bef60c6b1874cd7e559368adc7aed27df70ecaa3692`
- `final_experiments/cpu_revalidation_20261102/configs/roi_ssd_mobilenet.yaml`: `c3f1d663288f26717c60c0286f9961f9f09b66d0466f92a2fd70c6fc3ff3d563`
- `final_experiments/cpu_revalidation_20261102/pilot/roi_ssd_mobilenet/final_eval_manifest_candidate.txt`: `4f36f3455ced1a344ef69329479cd09013934d71aafa8dc800d7f7e5ea6bf6d4`
- `final_experiments/cpu_revalidation_20261102/pilot/roi_ssd_mobilenet/pilot_manifest.txt`: `817f5b145b774dc91ee0fa8d271e02c11d71d70bb50ff13082a0ba9487aa63cb`
- `final_experiments/cpu_revalidation_20261102/pilot/roi_ssd_mobilenet/split_metadata.json`: `0b5ccbdf21eb69c3cd005fec38e775f080598858ec32cb1cb03f8ba03f75ba89`
- `final_experiments/cpu_revalidation_20261102/pilot/roi_ssd_vgg16/final_eval_manifest_candidate.txt`: `4f36f3455ced1a344ef69329479cd09013934d71aafa8dc800d7f7e5ea6bf6d4`
- `final_experiments/cpu_revalidation_20261102/pilot/roi_ssd_vgg16/pilot_manifest.txt`: `817f5b145b774dc91ee0fa8d271e02c11d71d70bb50ff13082a0ba9487aa63cb`
- `final_experiments/cpu_revalidation_20261102/pilot/roi_ssd_vgg16/split_metadata.json`: `fa2b081b9e17fd82297e2536a06b445ccf0a900f0b56e3f1a9993857bda6bde7`
- `final_experiments/cpu_revalidation_20261102/pilot/yolo26n/final_eval_manifest_candidate.txt`: `4f36f3455ced1a344ef69329479cd09013934d71aafa8dc800d7f7e5ea6bf6d4`
- `final_experiments/cpu_revalidation_20261102/pilot/yolo26n/pilot_manifest.txt`: `817f5b145b774dc91ee0fa8d271e02c11d71d70bb50ff13082a0ba9487aa63cb`
- `final_experiments/cpu_revalidation_20261102/pilot/yolo26n/split_metadata.json`: `9d9ee427f9d4bc70730a5dfb3728ebb444badf0ce89245909818a7a43047878b`
- `final_experiments/cpu_revalidation_20261102/pilot/yolo26n/configs/roi_ssd_mobilenet.yaml`: `dc8650f3d2d96609f475b175aa0dd851f3395a6c79433f9daafaadb13eadfdbf`
- `final_experiments/cpu_revalidation_20261102/pilot/yolo26n/configs/roi_ssd_vgg16.yaml`: `05242db7377c65bf3bd2ce2aae68255d24c13c1ab6108db311a045a34e49c942`
- `final_experiments/cpu_revalidation_20261102/pilot/yolo26n/configs/yolo26n.yaml`: `e31b8245c6dd0ec8c5088330a896fe883775e7f074a14bf208207775f721985a`
- `final_experiments/cpu_revalidation_20261102/pilot/roi_ssd_vgg16/configs/roi_ssd_mobilenet.yaml`: `dc8650f3d2d96609f475b175aa0dd851f3395a6c79433f9daafaadb13eadfdbf`
- `final_experiments/cpu_revalidation_20261102/pilot/roi_ssd_vgg16/configs/roi_ssd_vgg16.yaml`: `05242db7377c65bf3bd2ce2aae68255d24c13c1ab6108db311a045a34e49c942`
- `final_experiments/cpu_revalidation_20261102/pilot/roi_ssd_vgg16/configs/yolo26n.yaml`: `e31b8245c6dd0ec8c5088330a896fe883775e7f074a14bf208207775f721985a`
- `final_experiments/cpu_revalidation_20261102/pilot/roi_ssd_mobilenet/configs/roi_ssd_mobilenet.yaml`: `dc8650f3d2d96609f475b175aa0dd851f3395a6c79433f9daafaadb13eadfdbf`
- `final_experiments/cpu_revalidation_20261102/pilot/roi_ssd_mobilenet/configs/roi_ssd_vgg16.yaml`: `05242db7377c65bf3bd2ce2aae68255d24c13c1ab6108db311a045a34e49c942`
- `final_experiments/cpu_revalidation_20261102/pilot/roi_ssd_mobilenet/configs/yolo26n.yaml`: `e31b8245c6dd0ec8c5088330a896fe883775e7f074a14bf208207775f721985a`
- `D:\ImageNet-VID\ImageNet\data\ImageNet2015\imagenet_vid_yolo_sampled\imagenet_vid.yaml`: `29ea3cd7d5f6acbca59decce76ef2c1491d8c1c7661eeff45263c8a26d45ba22`
- `final_experiments/cpu_revalidation_20261102/configs/confidence_diagnostic/roi_ssd_vgg16/conf_0p01.yaml`: `d6f0c3184c58bce4ebe7ddd850d793204beb605f8a198b8a383fbf66180e828e`
- `final_experiments/cpu_revalidation_20261102/configs/confidence_diagnostic/roi_ssd_vgg16/conf_0p05.yaml`: `5a55912510e2c1f982cd76f2d4895c31717a1e5abb78b6457b5953eff5145702`
- `final_experiments/cpu_revalidation_20261102/configs/confidence_diagnostic/roi_ssd_vgg16/conf_0p1.yaml`: `dcbaf581ce18a4e08dc793d0511cbd35baa3c35950eef4364eab2f86409006d2`
- `final_experiments/cpu_revalidation_20261102/configs/confidence_diagnostic/roi_ssd_vgg16/conf_0p25.yaml`: `5ab316d7f773dcfc2bf94fd3b0edf95a59b8bd68ee9e3a293ebb6e216fd97698`
- `final_experiments/cpu_revalidation_20261102/configs/confidence_diagnostic/roi_ssd_vgg16/conf_0p5.yaml`: `dde30d1b1f2f231bb494c8136a508720ecb877f89361c805091f6701b33dc36c`
