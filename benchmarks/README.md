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

# YOLO dengan opsi lengkap
python benchmarks/benchmark_yolo_gpu.py --model yolov8n.pt --runs 100 --warmup 5 --device cuda

# COCO benchmark
python benchmarks/benchmark_coco_gpu.py

# COCO dengan opsi lengkap
python benchmarks/benchmark_coco_gpu.py --runs 100 --warmup 5 --device cuda
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--runs` | `100` | Jumlah timed inference runs |
| `--warmup` | `5` | Warmup runs (tidak dihitung) |
| `--device` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `--model` | `yolov8n.pt` | Nama atau path model YOLO |
| `--output-dir` | `benchmarks/results` | Folder output |

`auto` device priority: `cuda` → `mps` → `cpu`

## Output

Setiap run menghasilkan dua file di `benchmarks/results/`:

- `yolo_benchmark_<timestamp>.json`
- `yolo_benchmark_<timestamp>.md`

## Running Tests
```bash
pytest benchmarks/tests/test_benchmarks_gpu.py -v
```

Test berjalan tanpa GPU, kamera, atau model file.
