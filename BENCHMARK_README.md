# Benchmark Framework Documentation

## Overview

The `tools/benchmarks/benchmark_framework.py` provides a comprehensive benchmarking system for evaluating SSD/RoiSSD models on various datasets. It measures:

- **mAP** (mean Average Precision) 
- **Recall** (detection recall rate)
- **FPS** (frames per second)
- **Latency** (inference time with percentiles: p50, p95, p99)
- **Per-class metrics** (individual class AP values)

## Configuration

The benchmark is configured via `config/benchmark.yaml`. Here's the structure:

```yaml
benchmark_params:
  dataset:
    name: "voc"           # Dataset: voc, voc-small-objects, vis-drone, ytbb
    im_size: 300          # Input size (int for square, or [h, w] for rectangular)
    test_im_sets: [...]   # Path to test dataset
    
  model:
    name: "roissd"        # Model type: ssd, roissd, ssd_mobilenet
    checkpoint_path: "path/to/checkpoint.pth"  # Trained model weights
    
  inference:
    batch_size: 1
    confidence_threshold: 0.01     # Minimum detection confidence
    iou_threshold: 0.5             # IoU threshold for mAP
    nms_iou: 0.45                  # Non-maximum suppression IoU
    
  output:
    results_dir: "benchmark_results/"        # Output directory
    results_filename: "benchmark_metrics.csv" # Results CSV file
    verbose: true                            # Detailed console output
    
device:
  type: "cuda"  # cuda, mps, or cpu
```

## Checkpoint Format

The framework expects PyTorch checkpoint files in one of these formats:

### Format 1: Full checkpoint dict (preferred)
```python
{
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'epoch': epoch_number
}
```

### Format 2: Model state dict only
```python
model.state_dict()
```

### Finding Your Checkpoint

Your trained models are in `trained_models/{task_name}/`. Look for `.pth` files:
- Files in root of task directory (e.g., `voc-roissd/epoch.pth`) are usually epoch metadata, NOT model weights
- Look for files like `model_{epoch}.pth` or `checkpoint.pth` which contain actual weights
- Or check subdirectories like `{task_name}/{transform_name}_results/` for variant-specific checkpoints

## Usage

### Basic Usage
```bash
python -m tools.benchmarks.benchmark_framework --benchmark-config config/benchmark.yaml
```

### Override Output Directory
```bash
python -m tools.benchmarks.benchmark_framework \
  --benchmark-config config/benchmark.yaml \
  --output-dir benchmark_results/my_experiment/
```

## Output

The framework generates:

1. **Console Output**: Formatted summary showing:
   - Dataset and model information
   - Detection quality metrics (mAP, Recall, per-class AP)
   - Performance metrics (FPS, latency statistics)

2. **CSV Output** (in `benchmark_results/`):
   - All metrics in tabular format for comparison across runs
   - Per-class AP values
   - Performance metrics

Example CSV columns:
```
dataset,model,im_size,device,num_images,mAP,recall,fps,latency_mean_ms,latency_p95_ms,...
```

## Next Steps

1. **Find your model checkpoint**: Identify the `.pth` file containing actual model weights
2. **Update config**: Set `checkpoint_path` to your model file
3. **Set dataset paths**: Update `test_im_sets` for your VOC/VisDrone installation
4. **Run benchmark**: Execute the command above
5. **Compare results**: Use CSV output to compare different models/configs

## Extending the Framework

The framework is designed for easy ablation studies:

- Create new config files for each variant (e.g., `config/benchmark_kf_disabled.yaml`)
- Modify inference parameters (confidence thresholds, NMS settings)
- Add new metrics by extending `compute_metrics()`
- Support custom models by extending the dataset/model loading logic

## Troubleshooting

### Checkpoint Loading Error
```
TypeError: Expected state_dict to be dict-like, got <class 'int'>
```
→ The checkpoint file contains metadata (epoch number), not model weights. Find the actual `.pth` file with weights.

### Dataset Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: 'path/to/VOC2007'
```
→ Update `test_im_sets` in config to point to your actual VOC dataset location.

### Out of Memory Error
→ Reduce `batch_size` in config, or use a smaller `im_size`.

### Very Low Accuracy/mAP
→ Check that:
- Checkpoint path is correct
- Model and dataset match (same im_size as training)
- Confidence threshold isn't too high
