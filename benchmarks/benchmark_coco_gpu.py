"""Benchmark GPU inference latency for Faster R-CNN COCO (Torchvision) vision model."""

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
import torchvision
from torchvision.models import detection as detection_model

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


def get_device_name(device: str) -> str:
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


def preprocess_image(image: np.ndarray, device: str) -> torch.Tensor:
    """Convert BGR numpy image to normalised float tensor.

    Parameters
    ----------
    image : np.ndarray
        BGR image from OpenCV.
    device : str
        Target device string.

    Returns
    -------
    torch.Tensor
        Batch tensor of shape (1, C, H, W) on the target device.
    """
    img_chw = image.copy().transpose((2, 0, 1))
    batch = np.expand_dims(img_chw, axis=0)
    tensor = torch.tensor(batch / 255.0, dtype=torch.float, device=device)
    return tensor


def run_coco_benchmark(device, n_runs, n_warmup, image, detection_threshold=0.2):
    """Run Faster R-CNN MobileNetV3 inference benchmark.

    Parameters
    ----------
    device : str
        Compute device string.
    n_runs : int
        Number of timed inference runs.
    n_warmup : int
        Number of warmup runs excluded from stats.
    image : np.ndarray
        Input BGR image.
    detection_threshold : float
        Minimum score to count a detection.

    Returns
    -------
    dict
        Benchmark result dictionary.
    """
    logger.info(f"Loading Faster R-CNN MobileNetV3 on device: {device}")
    model = detection_model.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights="FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1",
        progress=True,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
    ).to(device)
    model.eval()

    tensor = preprocess_image(image, device)

    logger.info(f"Running {n_warmup} warmup inference(s)...")
    with torch.no_grad():
        for _ in range(n_warmup):
            model(tensor)

    logger.info(f"Running {n_runs} timed inference(s)...")
    latencies_ms = []
    output = [{}]

    with torch.no_grad():
        for i in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000)
            if (i + 1) % 25 == 0:
                logger.info(f"  Progress: {i + 1}/{n_runs}")

    n_detections = int((output[0]["scores"] >= detection_threshold).sum().item())

    latencies_ms_sorted = sorted(latencies_ms)
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    min_ms = latencies_ms_sorted[0]
    max_ms = latencies_ms_sorted[-1]
    p50_ms = latencies_ms_sorted[int(len(latencies_ms_sorted) * 0.50)]
    p95_ms = latencies_ms_sorted[int(len(latencies_ms_sorted) * 0.95)]
    fps = 1000.0 / mean_ms

    return {
        "model": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "device": device,
        "device_name": get_device_name(device),
        "n_runs": n_runs,
        "n_warmup": n_warmup,
        "image_shape": list(image.shape),
        "detection_threshold": detection_threshold,
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
    print("  COCO (Faster R-CNN) GPU BENCHMARK RESULTS")
    print(sep)
    print(f"  Model         : {result['model']}")
    print(f"  Device        : {result['device']} ({result['device_name']})")
    print(f"  Runs          : {result['n_runs']}  (warmup: {result['n_warmup']})")
    print(f"  Image shape   : {result['image_shape']}")
    print(f"  Det. threshold: {result['detection_threshold']}")
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
    path = output_dir / f"coco_benchmark_{ts}.json"
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
    path = output_dir / f"coco_benchmark_{ts}.md"
    lines = [
        "# COCO (Faster R-CNN) GPU Inference Benchmark\n",
        f"**Date:** {ts}  ",
        f"**Platform:** {platform.system()} {platform.machine()}  ",
        f"**Python:** {platform.python_version()}  ",
        f"**PyTorch:** {torch.__version__}  ",
        f"**Torchvision:** {torchvision.__version__}  \n",
        "## Configuration\n",
        "| Key | Value |",
        "|-----|-------|",
        f"| Model | `{result['model']}` |",
        f"| Device | `{result['device']}` ({result['device_name']}) |",
        f"| Runs | {result['n_runs']} |",
        f"| Warmup | {result['n_warmup']} |",
        f"| Detection threshold | {result['detection_threshold']} |",
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
        "- Warmup runs excluded from latency statistics.",
        "- CUDA torch.cuda.synchronize() called before/after each run for accurate timing.",
        "- The original vlm_coco_local.py plugin hardcodes device='cpu'.",
        "- This benchmark demonstrates the speedup available with GPU acceleration.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Markdown saved: {path}")
    return path


def main():
    """Run the COCO Faster R-CNN GPU benchmark from the command line."""
    parser = argparse.ArgumentParser(
        description="Benchmark Faster R-CNN COCO inference latency on CPU/GPU."
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"]
    )
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    device = get_device(args.device)
    logger.info(f"Resolved device: {device} ({get_device_name(device)})")

    image = load_sample_image_from_repo()
    if image is None:
        logger.info("No repo image found — using synthetic image (640x480)")
        image = make_sample_image()

    result = run_coco_benchmark(
        device=device,
        n_runs=args.runs,
        n_warmup=args.warmup,
        image=image,
        detection_threshold=args.threshold,
    )

    result["system"] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),  # type: ignore[attr-defined],
    }

    output_dir = Path(args.output_dir)
    print_table(result)
    save_json(result, output_dir)
    save_markdown(result, output_dir)


if __name__ == "__main__":
    main()
