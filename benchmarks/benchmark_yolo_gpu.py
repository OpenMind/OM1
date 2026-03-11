"""Benchmark GPU inference latency for YOLOv8 (Ultralytics) vision model."""

import argparse
import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_device(preference: str) -> str:
    """Resolve compute device based on preference and availability.

    Parameters
    ----------
    preference : str
        One of 'auto', 'cuda', 'mps', or 'cpu'.

    Returns
    -------
    str
        Resolved device string.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preference


def get_gpu_name(device: str) -> str:
    """Return human-readable name for the given device.

    Parameters
    ----------
    device : str
        Device string, e.g. 'cuda', 'mps', or 'cpu'.

    Returns
    -------
    str
        Human-readable device name.
    """
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return "Apple Silicon MPS"
    return "CPU"


def make_sample_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a reproducible synthetic BGR image for benchmarking.

    Parameters
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.

    Returns
    -------
    np.ndarray
        Synthetic BGR image array.
    """
    rng = np.random.default_rng(seed=42)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (200, 300), (0, 255, 0), 3)
    cv2.circle(image, (400, 200), 80, (255, 0, 0), -1)
    return image


def load_sample_image_from_repo() -> np.ndarray | None:
    """Load the existing test image from system_hw_test/ if available.

    Returns
    -------
    np.ndarray or None
        Image array if found, None otherwise.
    """
    candidate = Path("system_hw_test/front_image.jpg")
    if candidate.exists():
        img = cv2.imread(str(candidate))
        if img is not None:
            logger.info(f"Using repo test image: {candidate}")
            return img
    return None


def run_yolo_benchmark(model_name, device, n_runs, n_warmup, image):
    """Run YOLO inference benchmark and return result dict.

    Parameters
    ----------
    model_name : str
        Path or name of the YOLO model file.
    device : str
        Compute device string.
    n_runs : int
        Number of timed inference runs.
    n_warmup : int
        Number of warmup runs excluded from stats.
    image : np.ndarray
        Input image for inference.

    Returns
    -------
    dict
        Benchmark result dictionary.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics is not installed. Run: pip install ultralytics")

    logger.info(f"Loading model: {model_name} on device: {device}")
    model = YOLO(model_name)
    model.to(device)

    logger.info(f"Running {n_warmup} warmup inference(s)...")
    for _ in range(n_warmup):
        model.predict(
            source=image, save=False, stream=False, verbose=False, device=device
        )

    logger.info(f"Running {n_runs} timed inference(s)...")
    latencies_ms = []
    results = []

    for i in range(n_runs):
        start = time.perf_counter()
        results = model.predict(
            source=image, save=False, stream=False, verbose=False, device=device
        )
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000)
        if (i + 1) % 25 == 0:
            logger.info(f"  Progress: {i + 1}/{n_runs}")

    n_detections = 0
    for r in results:
        if r.boxes is not None:
            n_detections += len(r.boxes)

    latencies_ms_sorted = sorted(latencies_ms)
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    min_ms = latencies_ms_sorted[0]
    max_ms = latencies_ms_sorted[-1]
    p50_ms = latencies_ms_sorted[int(len(latencies_ms_sorted) * 0.50)]
    p95_ms = latencies_ms_sorted[int(len(latencies_ms_sorted) * 0.95)]
    fps = 1000.0 / mean_ms

    return {
        "model": model_name,
        "device": device,
        "device_name": get_gpu_name(device),
        "n_runs": n_runs,
        "n_warmup": n_warmup,
        "image_shape": list(image.shape),
        "n_detections_last_run": n_detections,
        "latency_mean_ms": round(mean_ms, 2),
        "latency_min_ms": round(min_ms, 2),
        "latency_max_ms": round(max_ms, 2),
        "latency_p50_ms": round(p50_ms, 2),
        "latency_p95_ms": round(p95_ms, 2),
        "fps": round(fps, 2),
    }


def print_table(result):
    """Print benchmark results as a formatted table to stdout.

    Parameters
    ----------
    result : dict
        Benchmark result dictionary.
    """
    sep = "=" * 52
    print(f"\n{sep}")
    print("  YOLO GPU BENCHMARK RESULTS")
    print(sep)
    print(f"  Model         : {result['model']}")
    print(f"  Device        : {result['device']} ({result['device_name']})")
    print(f"  Runs          : {result['n_runs']}  (warmup: {result['n_warmup']})")
    print(f"  Image shape   : {result['image_shape']}")
    print(f"  Detections    : {result['n_detections_last_run']}")
    print(sep)
    print(f"  Mean latency  : {result['latency_mean_ms']:>8.2f} ms")
    print(f"  Min  latency  : {result['latency_min_ms']:>8.2f} ms")
    print(f"  Max  latency  : {result['latency_max_ms']:>8.2f} ms")
    print(f"  P50  latency  : {result['latency_p50_ms']:>8.2f} ms")
    print(f"  P95  latency  : {result['latency_p95_ms']:>8.2f} ms")
    print(f"  FPS           : {result['fps']:>8.2f}")
    print(f"{sep}\n")


def save_json(result, output_dir):
    """Save benchmark result as a JSON file.

    Parameters
    ----------
    result : dict
        Benchmark result dictionary.
    output_dir : Path
        Directory to write the file to.

    Returns
    -------
    Path
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"yolo_benchmark_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"JSON saved: {path}")
    return path


def save_markdown(result, output_dir):
    """Save benchmark result as a Markdown report.

    Parameters
    ----------
    result : dict
        Benchmark result dictionary.
    output_dir : Path
        Directory to write the file to.

    Returns
    -------
    Path
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"yolo_benchmark_{ts}.md"
    lines = [
        "# YOLO GPU Inference Benchmark\n",
        f"**Date:** {ts}  ",
        f"**Platform:** {platform.system()} {platform.machine()}  ",
        f"**Python:** {platform.python_version()}  ",
        f"**PyTorch:** {torch.__version__}  \n",
        "## Configuration\n",
        "| Key | Value |",
        "|-----|-------|",
        f"| Model | `{result['model']}` |",
        f"| Device | `{result['device']}` ({result['device_name']}) |",
        f"| Runs | {result['n_runs']} |",
        f"| Warmup | {result['n_warmup']} |",
        f"| Image shape | {result['image_shape']} |\n",
        "## Results\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean latency | {result['latency_mean_ms']} ms |",
        f"| Min latency  | {result['latency_min_ms']} ms |",
        f"| Max latency  | {result['latency_max_ms']} ms |",
        f"| P50 latency  | {result['latency_p50_ms']} ms |",
        f"| P95 latency  | {result['latency_p95_ms']} ms |",
        f"| **FPS**      | **{result['fps']}** |",
        f"| Detections (last run) | {result['n_detections_last_run']} |\n",
        "## Notes\n",
        "- Warmup runs are excluded from latency statistics.",
        "- Latency measured using `time.perf_counter()` (wall-clock).",
        "- Synthetic image used for reproducibility (seeded numpy RNG).",
        "- P95 latency indicates worst-case frame time under sustained load.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Markdown saved: {path}")
    return path


def main():
    """Run the YOLO GPU benchmark from the command line."""
    parser = argparse.ArgumentParser(
        description="Benchmark YOLOv8 inference latency on CPU/GPU."
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"]
    )
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    device = get_device(args.device)
    logger.info(f"Resolved device: {device} ({get_gpu_name(device)})")

    image = load_sample_image_from_repo()
    if image is None:
        logger.info("No repo image found — using synthetic image (640x480)")
        image = make_sample_image()

    result = run_yolo_benchmark(
        model_name=args.model,
        device=device,
        n_runs=args.runs,
        n_warmup=args.warmup,
        image=image,
    )

    result["system"] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),  # type: ignore[attr-defined],
    }

    output_dir = Path(args.output_dir)
    print_table(result)
    save_json(result, output_dir)
    save_markdown(result, output_dir)


if __name__ == "__main__":
    main()
