# GPU Inference Benchmarks

Benchmarks for local vision model inference performance on CPU and GPU hardware.

## Scripts

| Script | Model | Framework |
|--------|-------|-----------|
| `benchmark_yolo_gpu.py` | YOLOv8 (Ultralytics) | Ultralytics |
| `benchmark_coco_gpu.py` | Faster R-CNN MobileNetV3 | Torchvision |

## Usage

Run from the **repo root**:
```bash
# YOLO benchmark
python benchmarks/benchmark_yolo_gpu.py

# YOLO with full options
python benchmarks/benchmark_yolo_gpu.py --model yolov8n.pt --runs 100 --warmup 5 --device cuda

# COCO benchmark
python benchmarks/benchmark_coco_gpu.py

# COCO with full options
python benchmarks/benchmark_coco_gpu.py --runs 100 --warmup 5 --device cuda
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--runs` | `100` | Number of timed inference runs |
| `--warmup` | `5` | Warmup runs excluded from stats |
| `--device` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `--model` | `yolov8n.pt` | Model name or path (YOLO only) |
| `--output-dir` | `benchmarks/results` | Output directory |

`auto` device priority: `cuda` → `mps` → `cpu`

## Output

Each run writes two files to `benchmarks/results/`:

- `yolo_benchmark_<timestamp>.json`
- `yolo_benchmark_<timestamp>.md`

## Running Tests
```bash
pytest tests/benchmarks/test_benchmarks_gpu.py -v
```

Tests run without GPU, camera, or model files.

## Notes

- Benchmarks are not run in CI — execute manually on target hardware.
- PyTorch >= 2.6 (cu128 build) required for RTX 5000 series (sm_120).
