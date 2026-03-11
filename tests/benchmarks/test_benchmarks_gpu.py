"""Tests for GPU benchmark scripts."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch


class TestYoloBenchmarkHelpers:

    def test_get_device_auto_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            from benchmarks.benchmark_yolo_gpu import get_device

            assert get_device("auto") == "cuda"

    def test_get_device_auto_mps(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            from benchmarks.benchmark_yolo_gpu import get_device

            assert get_device("auto") == "mps"

    def test_get_device_auto_cpu_fallback(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            from benchmarks.benchmark_yolo_gpu import get_device

            assert get_device("auto") == "cpu"

    def test_get_device_explicit(self):
        from benchmarks.benchmark_yolo_gpu import get_device

        assert get_device("cpu") == "cpu"
        assert get_device("cuda") == "cuda"

    def test_get_gpu_name_cpu(self):
        from benchmarks.benchmark_yolo_gpu import get_gpu_name

        assert get_gpu_name("cpu") == "CPU"

    def test_get_gpu_name_mps(self):
        from benchmarks.benchmark_yolo_gpu import get_gpu_name

        assert get_gpu_name("mps") == "Apple Silicon MPS"

    def test_get_gpu_name_cuda(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="RTX 5070 Ti"),
        ):
            from benchmarks.benchmark_yolo_gpu import get_gpu_name

            assert get_gpu_name("cuda") == "RTX 5070 Ti"

    def test_make_sample_image_shape(self):
        from benchmarks.benchmark_yolo_gpu import make_sample_image

        img = make_sample_image(width=320, height=240)
        assert img.shape == (240, 320, 3)
        assert img.dtype == np.uint8

    def test_make_sample_image_reproducible(self):
        from benchmarks.benchmark_yolo_gpu import make_sample_image

        assert np.array_equal(make_sample_image(), make_sample_image())

    def test_load_sample_image_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from benchmarks.benchmark_yolo_gpu import load_sample_image_from_repo

        assert load_sample_image_from_repo() is None

    def test_load_sample_image_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        hw_dir = tmp_path / "system_hw_test"
        hw_dir.mkdir()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(hw_dir / "front_image.jpg"), img)
        from benchmarks.benchmark_yolo_gpu import load_sample_image_from_repo

        result = load_sample_image_from_repo()
        assert result is not None
        assert result.shape == (100, 100, 3)


class TestYoloBenchmarkCore:

    def test_run_yolo_benchmark_cpu(self):
        mock_box = MagicMock()
        mock_box.xyxy = [torch.tensor([10.0, 20.0, 30.0, 40.0])]
        mock_box.cls = [torch.tensor(0)]
        mock_box.conf = [torch.tensor(0.9)]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "person"}

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_model.names = {0: "person"}

        with patch("ultralytics.YOLO", return_value=mock_model):
            from benchmarks.benchmark_yolo_gpu import (
                make_sample_image,
                run_yolo_benchmark,
            )

            image = make_sample_image()
            result = run_yolo_benchmark(
                "yolov8n.pt", "cpu", n_runs=3, n_warmup=1, image=image
            )

        assert "fps" in result
        assert "latency_mean_ms" in result
        assert result["fps"] > 0
        assert result["device"] == "cpu"
        assert result["n_runs"] == 3


class TestYoloBenchmarkOutput:

    def test_print_table_runs_without_error(self, capsys):
        from benchmarks.benchmark_yolo_gpu import print_table

        result = {
            "model": "yolov8n.pt",
            "device": "cpu",
            "device_name": "CPU",
            "n_runs": 10,
            "n_warmup": 2,
            "image_shape": [480, 640, 3],
            "n_detections_last_run": 3,
            "latency_mean_ms": 25.5,
            "latency_min_ms": 20.0,
            "latency_max_ms": 35.0,
            "latency_p50_ms": 25.0,
            "latency_p95_ms": 33.0,
            "fps": 39.2,
        }
        print_table(result)
        captured = capsys.readouterr()
        assert "yolov8n.pt" in captured.out
        assert "39.2" in captured.out

    def test_save_json(self, tmp_path):
        import json

        from benchmarks.benchmark_yolo_gpu import save_json

        result = {"model": "yolov8n.pt", "fps": 42.0, "latency_mean_ms": 23.8}
        out_path = save_json(result, tmp_path)
        assert out_path.exists()
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["fps"] == 42.0

    def test_save_markdown(self, tmp_path):
        from benchmarks.benchmark_yolo_gpu import save_markdown

        result = {
            "model": "yolov8n.pt",
            "device": "cuda",
            "device_name": "RTX 5070 Ti",
            "n_runs": 100,
            "n_warmup": 5,
            "image_shape": [480, 640, 3],
            "n_detections_last_run": 2,
            "latency_mean_ms": 22.0,
            "latency_min_ms": 18.0,
            "latency_max_ms": 30.0,
            "latency_p50_ms": 21.5,
            "latency_p95_ms": 28.0,
            "fps": 45.4,
        }
        out_path = save_markdown(result, tmp_path)
        assert out_path.exists()
        content = out_path.read_text()
        assert "YOLO" in content
        assert "45.4" in content

    def test_main_cpu(self, tmp_path):
        mock_box = MagicMock()
        mock_box.xyxy = [torch.tensor([10.0, 20.0, 30.0, 40.0])]
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        with (
            patch("ultralytics.YOLO", return_value=mock_model),
            patch(
                "sys.argv",
                [
                    "benchmark_yolo_gpu.py",
                    "--runs",
                    "3",
                    "--warmup",
                    "1",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(tmp_path),
                ],
            ),
        ):
            from benchmarks.benchmark_yolo_gpu import main

            main()

        json_files = list(tmp_path.glob("yolo_benchmark_*.json"))
        assert len(json_files) == 1


class TestCocoBenchmarkHelpers:

    def test_get_device_auto_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            from benchmarks.benchmark_coco_gpu import get_device

            assert get_device("auto") == "cuda"

    def test_get_device_auto_mps(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            from benchmarks.benchmark_coco_gpu import get_device

            assert get_device("auto") == "mps"

    def test_get_device_auto_cpu_fallback(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            from benchmarks.benchmark_coco_gpu import get_device

            assert get_device("auto") == "cpu"

    def test_get_device_name_cpu(self):
        from benchmarks.benchmark_coco_gpu import get_device_name

        assert get_device_name("cpu") == "CPU"

    def test_get_device_name_mps(self):
        from benchmarks.benchmark_coco_gpu import get_device_name

        assert get_device_name("mps") == "Apple Silicon MPS"

    def test_get_device_name_cuda(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="RTX 5070 Ti"),
        ):
            from benchmarks.benchmark_coco_gpu import get_device_name

            assert get_device_name("cuda") == "RTX 5070 Ti"

    def test_make_sample_image_shape(self):
        from benchmarks.benchmark_coco_gpu import make_sample_image

        img = make_sample_image(width=320, height=240)
        assert img.shape == (240, 320, 3)

    def test_make_sample_image_reproducible(self):
        from benchmarks.benchmark_coco_gpu import make_sample_image

        assert np.array_equal(make_sample_image(), make_sample_image())

    def test_preprocess_image_shape(self):
        from benchmarks.benchmark_coco_gpu import make_sample_image, preprocess_image

        img = make_sample_image(width=320, height=240)
        tensor = preprocess_image(img, device="cpu")
        assert tensor.shape == (1, 3, 240, 320)
        assert tensor.dtype == torch.float32

    def test_preprocess_image_normalised(self):
        from benchmarks.benchmark_coco_gpu import make_sample_image, preprocess_image

        tensor = preprocess_image(make_sample_image(), device="cpu")
        assert float(tensor.max()) <= 1.0
        assert float(tensor.min()) >= 0.0

    def test_load_sample_image_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from benchmarks.benchmark_coco_gpu import load_sample_image_from_repo

        assert load_sample_image_from_repo() is None

    def test_load_sample_image_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        hw_dir = tmp_path / "system_hw_test"
        hw_dir.mkdir()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(hw_dir / "front_image.jpg"), img)
        from benchmarks.benchmark_coco_gpu import load_sample_image_from_repo

        result = load_sample_image_from_repo()
        assert result is not None


class TestCocoBenchmarkCore:

    def test_run_coco_benchmark_cpu(self):
        mock_output = [
            {
                "scores": torch.tensor([0.9, 0.3, 0.1]),
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]] * 3),
                "labels": torch.tensor([1, 2, 3]),
            }
        ]
        mock_model = MagicMock()
        mock_model.return_value = mock_output

        with patch("benchmarks.benchmark_coco_gpu.detection_model") as mock_det:
            mock_det.fasterrcnn_mobilenet_v3_large_320_fpn.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = mock_model

            from benchmarks.benchmark_coco_gpu import (
                make_sample_image,
                run_coco_benchmark,
            )

            image = make_sample_image()
            result = run_coco_benchmark("cpu", n_runs=3, n_warmup=1, image=image)

        assert "fps" in result
        assert result["fps"] > 0
        assert result["device"] == "cpu"


class TestCocoBenchmarkOutput:

    def test_print_table_runs_without_error(self, capsys):
        from benchmarks.benchmark_coco_gpu import print_table

        result = {
            "model": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "device": "cpu",
            "device_name": "CPU",
            "n_runs": 10,
            "n_warmup": 2,
            "image_shape": [480, 640, 3],
            "detection_threshold": 0.2,
            "n_detections_last_run": 1,
            "latency_mean_ms": 120.0,
            "latency_min_ms": 100.0,
            "latency_max_ms": 150.0,
            "latency_p50_ms": 118.0,
            "latency_p95_ms": 145.0,
            "fps": 8.3,
        }
        print_table(result)
        captured = capsys.readouterr()
        assert "fasterrcnn" in captured.out
        assert "8.3" in captured.out

    def test_save_json(self, tmp_path):
        import json

        from benchmarks.benchmark_coco_gpu import save_json

        result = {"model": "fasterrcnn", "fps": 8.3, "latency_mean_ms": 120.0}
        out_path = save_json(result, tmp_path)
        assert out_path.exists()
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["fps"] == 8.3

    def test_save_markdown(self, tmp_path):
        from benchmarks.benchmark_coco_gpu import save_markdown

        result = {
            "model": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "device": "cuda",
            "device_name": "RTX 5070 Ti",
            "n_runs": 100,
            "n_warmup": 5,
            "image_shape": [480, 640, 3],
            "detection_threshold": 0.2,
            "n_detections_last_run": 1,
            "latency_mean_ms": 45.0,
            "latency_min_ms": 38.0,
            "latency_max_ms": 60.0,
            "latency_p50_ms": 44.0,
            "latency_p95_ms": 57.0,
            "fps": 22.2,
        }
        out_path = save_markdown(result, tmp_path)
        assert out_path.exists()
        content = out_path.read_text()
        assert "COCO" in content
        assert "22.2" in content
        assert "vlm_coco_local.py" in content

    def test_main_cpu(self, tmp_path):
        mock_output = [
            {
                "scores": torch.tensor([0.9]),
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "labels": torch.tensor([1]),
            }
        ]
        mock_model = MagicMock()
        mock_model.return_value = mock_output

        with (
            patch("benchmarks.benchmark_coco_gpu.detection_model") as mock_det,
            patch(
                "sys.argv",
                [
                    "benchmark_coco_gpu.py",
                    "--runs",
                    "3",
                    "--warmup",
                    "1",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(tmp_path),
                ],
            ),
        ):
            mock_det.fasterrcnn_mobilenet_v3_large_320_fpn.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = mock_model

            from benchmarks.benchmark_coco_gpu import main

            main()

        json_files = list(tmp_path.glob("coco_benchmark_*.json"))
        assert len(json_files) == 1


class TestYoloMissingBranches:

    def test_run_yolo_benchmark_import_error(self):
        """Test ImportError raised when ultralytics not installed."""
        with patch.dict("sys.modules", {"ultralytics": None}):
            from benchmarks.benchmark_yolo_gpu import (
                make_sample_image,
                run_yolo_benchmark,
            )

            image = make_sample_image()
            with pytest.raises(ImportError):
                run_yolo_benchmark(
                    "yolov8n.pt", "cpu", n_runs=1, n_warmup=0, image=image
                )

    def test_run_yolo_benchmark_progress_log(self):
        """Test progress logging triggers at every 25 runs."""
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        with patch("ultralytics.YOLO", return_value=mock_model):
            from benchmarks.benchmark_yolo_gpu import (
                make_sample_image,
                run_yolo_benchmark,
            )

            image = make_sample_image()
            result = run_yolo_benchmark(
                "yolov8n.pt", "cpu", n_runs=25, n_warmup=0, image=image
            )
        assert result["n_runs"] == 25

    def test_main_cpu_no_repo_image(self, tmp_path, monkeypatch):
        """Test main() uses synthetic image when repo image not found."""
        monkeypatch.chdir(tmp_path)
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]

        with (
            patch("ultralytics.YOLO", return_value=mock_model),
            patch(
                "sys.argv",
                [
                    "benchmark_yolo_gpu.py",
                    "--runs",
                    "3",
                    "--warmup",
                    "1",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(tmp_path),
                ],
            ),
        ):
            from benchmarks.benchmark_yolo_gpu import main

            main()

        assert len(list(tmp_path.glob("yolo_benchmark_*.json"))) == 1


class TestCocoMissingBranches:

    def test_run_coco_benchmark_cuda_sync(self):
        """Test cuda synchronize branches are hit when device is cuda."""
        mock_output = [
            {
                "scores": torch.tensor([0.9]),
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([1]),
            }
        ]
        mock_tensor = torch.zeros(1, 3, 480, 640)
        mock_model = MagicMock()
        mock_model.return_value = mock_output

        with (
            patch("benchmarks.benchmark_coco_gpu.detection_model") as mock_det,
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.synchronize"),
            patch(
                "benchmarks.benchmark_coco_gpu.get_device_name", return_value="Mock GPU"
            ),
            patch(
                "benchmarks.benchmark_coco_gpu.preprocess_image",
                return_value=mock_tensor,
            ),
        ):
            mock_det.fasterrcnn_mobilenet_v3_large_320_fpn.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = mock_model

            from benchmarks.benchmark_coco_gpu import (
                make_sample_image,
                run_coco_benchmark,
            )

            image = make_sample_image()
            result = run_coco_benchmark("cuda", n_runs=3, n_warmup=1, image=image)
        assert result["device"] == "cuda"

    def test_main_cpu_no_repo_image(self, tmp_path, monkeypatch):
        """Test main() uses synthetic image when repo image not found."""
        monkeypatch.chdir(tmp_path)
        mock_output = [
            {
                "scores": torch.tensor([0.9]),
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([1]),
            }
        ]
        mock_model = MagicMock()
        mock_model.return_value = mock_output

        with (
            patch("benchmarks.benchmark_coco_gpu.detection_model") as mock_det,
            patch(
                "sys.argv",
                [
                    "benchmark_coco_gpu.py",
                    "--runs",
                    "3",
                    "--warmup",
                    "1",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(tmp_path),
                ],
            ),
        ):
            mock_det.fasterrcnn_mobilenet_v3_large_320_fpn.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = mock_model

            from benchmarks.benchmark_coco_gpu import main

            main()

        assert len(list(tmp_path.glob("coco_benchmark_*.json"))) == 1
