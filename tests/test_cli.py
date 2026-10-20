from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from aidetector import cli
from aidetector.checkpoint_utils import SafeCheckpointLoadError
from aidetector.cli import app
from aidetector.evaluation import ImageSample
from aidetector.types import DetectionResult


class BrightnessDetector:
    threshold = 0.5

    def predict_images(self, images, paths=None):
        paths = paths or [None] * len(images)
        results = []
        for image, path in zip(images, paths, strict=True):
            probability_ai = image.convert("L").resize((1, 1)).getpixel((0, 0)) / 255
            results.append(
                DetectionResult(
                    path=path,
                    label="ai" if probability_ai >= self.threshold else "real",
                    probability_ai=probability_ai,
                    probability_real=1 - probability_ai,
                    confidence=max(probability_ai, 1 - probability_ai),
                    raw_score=probability_ai,
                    backend="brightness",
                )
            )
        return results

    def predict_image(self, image, path: Path | None = None):
        return self.predict_images([image], [path])[0]

    def predict_path(self, path):
        with Image.open(path) as image:
            return self.predict_image(image, Path(path))

    def model_info(self):
        return {"backend": "brightness", "threshold": self.threshold}


def test_detect_json_output(monkeypatch, tmp_path):
    monkeypatch.setattr("aidetector.cli.create_detector", lambda *args, **kwargs: BrightnessDetector())
    Image.new("RGB", (8, 8), "white").save(tmp_path / "fake.png")

    result = CliRunner().invoke(app, ["detect", str(tmp_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip())
    assert payload["label"] == "ai"
    assert payload["backend"] == "brightness"


def test_benchmark_folder_writes_report(monkeypatch, tmp_path):
    monkeypatch.setattr("aidetector.cli.create_detector", lambda *args, **kwargs: BrightnessDetector())
    (tmp_path / "real").mkdir()
    (tmp_path / "ai").mkdir()
    Image.new("RGB", (8, 8), "black").save(tmp_path / "real" / "real.png")
    Image.new("RGB", (8, 8), "white").save(tmp_path / "ai" / "fake.png")
    output = tmp_path / "report.json"

    result = CliRunner().invoke(app, ["benchmark-folder", str(tmp_path), "--output", str(output)])

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metrics"]["accuracy"] == 1.0
    assert payload["model"]["backend"] == "brightness"


def test_benchmark_calibrated_folder_writes_report(monkeypatch, tmp_path):
    monkeypatch.setattr("aidetector.cli.create_detector", lambda *args, **kwargs: BrightnessDetector())
    (tmp_path / "real").mkdir()
    (tmp_path / "ai").mkdir()
    Image.new("RGB", (8, 8), "black").save(tmp_path / "real" / "real-1.png")
    Image.new("RGB", (8, 8), "black").save(tmp_path / "real" / "real-2.png")
    Image.new("RGB", (8, 8), "white").save(tmp_path / "ai" / "fake-1.png")
    Image.new("RGB", (8, 8), "white").save(tmp_path / "ai" / "fake-2.png")
    output = tmp_path / "calibrated-report.json"

    result = CliRunner().invoke(
        app,
        [
            "benchmark-calibrated-folder",
            str(tmp_path),
            "--backend",
            "hf",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["test"]["metrics"]["accuracy"] == 1.0
    assert payload["model"]["backend"] == "brightness"


def test_benchmark_tiny_genimage_local_writes_group_report(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "aidetector.cli.collect_tiny_genimage_parquet_samples",
        lambda *args, **kwargs: [
            ImageSample(Image.new("RGB", (8, 8), "black"), 0, sample_id="r1", metadata={"generator": "Real", "source_shard": "s1"}),
            ImageSample(Image.new("RGB", (8, 8), "black"), 0, sample_id="r2", metadata={"generator": "Real", "source_shard": "s1"}),
            ImageSample(Image.new("RGB", (8, 8), "white"), 1, sample_id="f1", metadata={"generator": "SD14", "source_shard": "s1"}),
            ImageSample(Image.new("RGB", (8, 8), "white"), 1, sample_id="f2", metadata={"generator": "SD14", "source_shard": "s1"}),
        ],
    )
    monkeypatch.setattr("aidetector.cli.create_detector", lambda *args, **kwargs: BrightnessDetector())
    output = tmp_path / "tiny-report.json"
    dummy_parquet = tmp_path / "dummy.parquet"
    dummy_parquet.write_text("x", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "benchmark-tiny-genimage-local",
            str(dummy_parquet),
            "--backend",
            "hf",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["groups"]["generator"]["SD14"]["accuracy"] == 1.0
    assert payload["dataset"]["kind"] == "tiny-genimage-local"


@pytest.mark.parametrize(
    "backend",
    ["or-ensemble", "or-ensemble-high-recall"],
)
def test_fixed_tiny_genimage_writes_report(
    monkeypatch, tmp_path, backend
):
    samples = [
        ImageSample(
            Image.new("RGB", (8, 8), "black"),
            0,
            sample_id="r1",
            metadata={
                "generator": "Real",
                "source_shard": "shard-1.parquet",
                "label_name": "real",
            },
        ),
        ImageSample(
            Image.new("RGB", (8, 8), "black"),
            0,
            sample_id="r2",
            metadata={
                "generator": "Real",
                "source_shard": "shard-1.parquet",
                "label_name": "real",
            },
        ),
        ImageSample(
            Image.new("RGB", (8, 8), "white"),
            1,
            sample_id="f1",
            metadata={
                "generator": "SD14",
                "source_shard": "shard-1.parquet",
                "label_name": "ai",
            },
        ),
        ImageSample(
            Image.new("RGB", (8, 8), "white"),
            1,
            sample_id="f2",
            metadata={
                "generator": "SD14",
                "source_shard": "shard-1.parquet",
                "label_name": "ai",
            },
        ),
    ]

    captured = {}

    def fake_collect(paths, **kwargs):
        captured["paths"] = paths
        captured.update(kwargs)
        return samples

    def fake_create_detector(*args, **kwargs):
        assert kwargs["threshold"] == 0.5
        assert kwargs["backend"] == backend
        return BrightnessDetector()

    monkeypatch.setattr(
        cli, "collect_tiny_genimage_parquet_samples", fake_collect
    )
    monkeypatch.setattr(cli, "create_detector", fake_create_detector)

    parquet = tmp_path / "shard-1.parquet"
    parquet.write_text("placeholder", encoding="utf-8")
    output = tmp_path / "fixed-report.json"

    result = CliRunner().invoke(
        app,
        [
            "benchmark-tiny-genimage-fixed",
            str(parquet),
            "--backend",
            backend,
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["max_per_class_per_shard"] == 25

    payload = json.loads(output.read_text(encoding="utf-8"))
    evaluation = payload["evaluation"]

    assert evaluation["threshold_calibration"] is False
    assert evaluation["global_threshold"] == 0.5
    assert evaluation["backend"] == backend
    assert "not calibrated probabilities" in evaluation["score_interpretation"]
    assert payload["metrics"]["threshold"] == 0.5
    assert "SD14" in payload["by_generator"]
    assert "shard-1.parquet" in payload["by_source_shard"]
    assert "SD14" in payload["generator_vs_real"]


def test_fixed_tiny_genimage_refuses_existing_report(
    monkeypatch, tmp_path
):
    output = tmp_path / "existing-report.json"
    original = '{"keep": "untouched"}'
    output.write_text(original, encoding="utf-8")

    parquet = tmp_path / "shard.parquet"
    parquet.write_text("placeholder", encoding="utf-8")

    def unexpected_collect(*args, **kwargs):
        raise AssertionError("Existing output must be rejected before loading data")

    monkeypatch.setattr(
        cli, "collect_tiny_genimage_parquet_samples", unexpected_collect
    )

    result = CliRunner().invoke(
        app,
        [
            "benchmark-tiny-genimage-fixed",
            str(parquet),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "already exists" in result.output
    assert output.read_text(encoding="utf-8") == original


def test_fixed_tiny_genimage_rejects_unsupported_backend(
    monkeypatch, tmp_path
):
    parquet = tmp_path / "shard.parquet"
    parquet.write_text("placeholder", encoding="utf-8")

    def unexpected_collect(*args, **kwargs):
        raise AssertionError("Unsupported backend must be rejected before loading")

    monkeypatch.setattr(
        cli, "collect_tiny_genimage_parquet_samples", unexpected_collect
    )

    output = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "benchmark-tiny-genimage-fixed",
            str(parquet),
            "--backend",
            "hf",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "or-ensemble" in result.output
    assert not output.exists()

def test_calibrated_ultra_preserves_zero_weight_and_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "aidetector.cli.create_detector",
        lambda *args, **kwargs: BrightnessDetector(),
    )
    monkeypatch.setattr(
        "aidetector.cli.search_blend_weight_threshold",
        lambda *args, **kwargs: {
            "primary_weight": 0.0,
            "threshold": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        },
    )

    (tmp_path / "real").mkdir()
    (tmp_path / "ai").mkdir()
    Image.new("RGB", (8, 8), "black").save(tmp_path / "real" / "real-1.png")
    Image.new("RGB", (8, 8), "gray").save(tmp_path / "real" / "real-2.png")
    Image.new("RGB", (8, 8), "white").save(tmp_path / "ai" / "fake-1.png")
    Image.new("RGB", (8, 8), "red").save(tmp_path / "ai" / "fake-2.png")
    output = tmp_path / "zero-values-report.json"

    result = CliRunner().invoke(
        app,
        [
            "benchmark-calibrated-folder",
            str(tmp_path),
            "--backend",
            "ultra",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["model"]["selected_primary_weight"] == 0.0
    assert payload["model"]["selected_threshold"] == 0.0
    assert payload["test"]["metrics"]["threshold"] == 0.0


def test_main_handles_safe_checkpoint_error(monkeypatch, capsys):
    def fail_app():
        raise SafeCheckpointLoadError("blocked checkpoint for test")

    monkeypatch.setattr(cli, "app", fail_app)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert "blocked checkpoint for test" in capsys.readouterr().out



def test_predict_paths_discards_failed_batch_and_processes_next_file(tmp_path):
    class FailFirstBatchDetector(BrightnessDetector):
        def __init__(self):
            self.seen_batches = []

        def predict_images(self, images, paths=None):
            self.seen_batches.append(list(paths or []))
            if len(self.seen_batches) == 1:
                raise RuntimeError("synthetic prediction failure")
            return super().predict_images(images, paths)

    paths = [
        tmp_path / "first.png",
        tmp_path / "second.png",
        tmp_path / "third.png",
    ]
    for index, image_path in enumerate(paths):
        Image.new("RGB", (8, 8), ["black", "white", "gray"][index]).save(
            image_path
        )

    detector = FailFirstBatchDetector()
    results = cli._predict_paths(
        detector,
        paths,
        batch_size=2,
        quiet=True,
    )

    assert detector.seen_batches == [paths[:2], paths[2:]]
    assert [result.path for result in results] == [paths[2]]



def test_predict_paths_handles_failed_final_batch(tmp_path, capsys):
    class AlwaysFailDetector(BrightnessDetector):
        def predict_images(self, images, paths=None):
            raise RuntimeError("synthetic final-batch failure")

    paths = [tmp_path / f"image-{index}.png" for index in range(3)]
    for image_path in paths:
        Image.new("RGB", (8, 8), "gray").save(image_path)

    results = cli._predict_paths(
        AlwaysFailDetector(),
        paths,
        batch_size=5,
        quiet=True,
    )

    assert results == []
    output = capsys.readouterr()
    assert "Failed final batch:" in output.err
    assert "synthetic final-batch failure" in output.err



def test_detect_json_keeps_batch_errors_on_stderr(tmp_path, monkeypatch):
    class FailFirstBatchDetector(BrightnessDetector):
        def __init__(self):
            self.calls = 0

        def predict_images(self, images, paths=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic JSON batch failure")
            return super().predict_images(images, paths)

    monkeypatch.setattr(
        "aidetector.cli.create_detector",
        lambda *args, **kwargs: FailFirstBatchDetector(),
    )
    for index in range(3):
        Image.new("RGB", (8, 8), "white").save(
            tmp_path / f"image-{index}.png"
        )

    result = CliRunner().invoke(
        app,
        ["detect", str(tmp_path), "--json", "--batch-size", "2"],
    )

    assert result.exit_code == 1, result.output
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    payloads = [json.loads(line) for line in lines]
    assert len(payloads) == 1
    assert payloads[0]["path"].endswith("image-2.png")
    assert "Failed batch:" in result.stderr
    assert "synthetic JSON batch failure" in result.stderr
    assert "Incomplete run:" in result.stderr


def test_detect_rejects_nonfixed_or_ensemble_threshold(tmp_path, monkeypatch):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "gray").save(image_path)
    factory_called = False

    def unexpected_factory(*args, **kwargs):
        nonlocal factory_called
        factory_called = True
        raise AssertionError("Detector factory must not run for invalid CLI input")

    monkeypatch.setattr(cli, "create_detector", unexpected_factory)

    result = CliRunner().invoke(
        app,
        [
            "detect",
            str(image_path),
            "--backend",
            "or-ensemble",
            "--threshold",
            "0.6",
        ],
    )

    assert result.exit_code == 2
    assert "OR ensemble" in result.output
    assert factory_called is False



@pytest.mark.parametrize(
    ("command", "command_args"),
    [
        ("benchmark-folder", ["unused-dataset"]),
        ("benchmark-hf", ["unused/dataset"]),
        ("serve", []),
        ("api", []),
    ],
)
def test_cli_entrypoints_reject_nonfixed_or_threshold(
    command, command_args, monkeypatch
):
    import sys
    from types import ModuleType

    def unexpected_factory(*args, **kwargs):
        raise AssertionError("Detector factory must not run for invalid CLI input")

    monkeypatch.setattr(cli, "create_detector", unexpected_factory)

    if command == "api":
        fake_api = ModuleType("aidetector.api")
        fake_api.create_app = lambda **kwargs: object()
        fake_uvicorn = ModuleType("uvicorn")
        fake_uvicorn.run = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "aidetector.api", fake_api)
        monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    result = CliRunner().invoke(
        app,
        [
            command,
            *command_args,
            "--backend",
            "OR-ENSEMBLE",
            "--threshold",
            "0.6",
        ],
    )

    assert result.exit_code == 2, result.output
    assert "OR ensemble" in result.output
    assert "Traceback" not in result.output
