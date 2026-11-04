$src = 'final_experiments/best_map_sweeps'
$ssd = 'outputs/batch_infer/imagenet-vid-roissd-20261028/stage3/best_map.pt'
$mobile = 'outputs/batch_infer/imagenet-vid-roissd-mobilenet-20261027/stage3/best_map.pt'

New-Item -ItemType Directory -Force -Path "$ssd/pad-loop", "$ssd/fixed-size-loop", "$mobile/fixed-size-loop" | Out-Null

Copy-Item "$src/roi_ssd/pad-loop/results.csv" "$ssd/pad-loop/"
Copy-Item "$src/roi_ssd/fixed-size-loop/fixed_size_results_v2.csv" "$ssd/fixed-size-loop/"
Copy-Item "$src/roi_ssd/fixed-size-loop/fixed_size_coverage_v2.csv" "$ssd/fixed-size-loop/"
Copy-Item "$src/roi_ssd/fixed-size-loop/config.yaml" "$ssd/config.yaml"

Copy-Item "$src/roi_ssd_mobilenet/fixed-size-loop/fixed_size_results_v2.csv" "$mobile/fixed-size-loop/"
Copy-Item "$src/roi_ssd_mobilenet/fixed-size-loop/fixed_size_coverage_v2.csv" "$mobile/fixed-size-loop/"
Copy-Item "$src/roi_ssd_mobilenet/fixed-size-loop/config.yaml" "$mobile/config.yaml"