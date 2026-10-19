
from __future__ import annotations

import pytest
from PIL import Image

from aidetector.checkpoint_utils import load_torch_checkpoint
from aidetector.model import (
    HuggingFaceImageDetector,
    _build_fallback_image_processor,
    _resolve_local_hf_snapshot,
    iter_images,
)
from aidetector.sentry_adapter import convert_sentry_convnext_state_dict
from aidetector.types import DetectionResult, EvaluationMetrics


def test_iter_images_recurses_and_filters(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    Image.new("RGB", (4, 4), "red").save(tmp_path / "a.jpg")
    Image.new("RGB", (4, 4), "blue").save(nested / "b.png")
    (tmp_path / "note.txt").write_text("not an image", encoding="utf-8")

    paths = iter_images(tmp_path, recursive=True)

    assert [path.name for path in paths] == ["a.jpg", "b.png"]
    assert iter_images(tmp_path / "note.txt") == []


def test_detection_result_serializes_path_and_backend(tmp_path):
    result = DetectionResult(
        path=tmp_path / "image.png",
        label="ai",
        probability_ai=0.81234567,
        probability_real=0.18765433,
        confidence=0.81234567,
        raw_score=1.23456789,
        backend="test",
    )

    payload = result.as_dict()

    assert payload["path"].endswith("image.png")
    assert payload["probability_ai"] == 0.812346
    assert payload["backend"] == "test"


def test_evaluation_metrics_serializes_auc_none():
    metrics = EvaluationMetrics(
        dataset="toy",
        n_samples=1,
        n_real=1,
        n_ai=0,
        threshold=0.5,
        accuracy=1.0,
        balanced_accuracy=0.5,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        roc_auc=None,
        true_positive=0,
        true_negative=1,
        false_positive=0,
        false_negative=0,
        seconds=0.1,
        images_per_second=10.0,
    )

    assert metrics.as_dict()["roc_auc"] is None
    assert metrics.as_dict()["confusion"]["true_negative"] == 1


def test_convert_sentry_convnext_state_dict_maps_keys():
    converted = convert_sentry_convnext_state_dict(
        {
            "backbone.downsample_layers.0.0.weight": "a",
            "backbone.stages.0.0.depthwise_conv.weight": "b",
            "backbone.stages.0.0.pointwise_conv1.weight": "c",
            "head.fc.weight": "d",
        }
    )

    assert converted["stem.0.weight"] == "a"
    assert converted["stages.0.blocks.0.conv_dw.weight"] == "b"
    assert converted["stages.0.blocks.0.mlp.fc1.weight"] == "c"
    assert converted["head.fc.weight"] == "d"


def test_load_torch_checkpoint_requests_weights_only(tmp_path):
    class TorchStub:
        def __init__(self):
            self.calls = []

        def load(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"state_dict": {"example": "tensor"}}

    torch_stub = TorchStub()
    weight_path = tmp_path / "weights.pth"

    result = load_torch_checkpoint(torch_stub, weight_path)

    assert result == {"state_dict": {"example": "tensor"}}
    assert len(torch_stub.calls) == 1
    args, kwargs = torch_stub.calls[0]
    assert args == (str(weight_path),)
    assert kwargs == {"map_location": "cpu", "weights_only": True}


def test_load_torch_checkpoint_fails_closed_if_safe_loading_unsupported(tmp_path):
    class TorchStub:
        def __init__(self):
            self.calls = []

        def load(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            raise TypeError("unexpected keyword argument 'weights_only'")

    torch_stub = TorchStub()

    with pytest.raises(RuntimeError, match="Safe checkpoint loading"):
        load_torch_checkpoint(torch_stub, tmp_path / "weights.pth")

    assert len(torch_stub.calls) == 1
    assert torch_stub.calls[0][1]["weights_only"] is True


def test_load_torch_checkpoint_does_not_retry_after_rejection(tmp_path):
    import pickle

    class TorchStub:
        def __init__(self):
            self.calls = []

        def load(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            raise pickle.UnpicklingError("blocked unsupported global")

    torch_stub = TorchStub()

    with pytest.raises(RuntimeError, match="safe weights-only loader"):
        load_torch_checkpoint(torch_stub, tmp_path / "checkpoint.pth")

    assert len(torch_stub.calls) == 1
    assert torch_stub.calls[0][1]["weights_only"] is True


def test_detection_result_real_label_survives_serialization_rounding():
    result = DetectionResult(
        path=None,
        label="real",
        probability_ai=0.4999996,
        probability_real=0.5000004,
        confidence=0.5000004,
        raw_score=0.4999996,
        backend="or-ensemble",
        details={"score_semantics": "composite evidence score"},
    )

    payload = result.as_dict()

    assert payload["label"] == "real"
    assert payload["probability_ai"] < 0.5


def test_detection_result_as_dict_copies_nested_details():
    result = DetectionResult(
        path=None,
        label="ai",
        probability_ai=0.8,
        probability_real=0.2,
        confidence=0.8,
        raw_score=0.8,
        details={"components": {"univfd": {"threshold": 0.42259}}},
    )

    payload = result.as_dict()
    payload["details"]["components"]["univfd"]["threshold"] = 0.9

    assert result.details["components"]["univfd"]["threshold"] == 0.42259


def test_detection_result_is_hashable_with_details():
    result = DetectionResult(
        path=None,
        label="ai",
        probability_ai=0.8,
        probability_real=0.2,
        confidence=0.8,
        raw_score=0.8,
        details={"mode": "conservative"},
    )

    assert isinstance(hash(result), int)


def _create_hf_cache(tmp_path, model_id="example/model"):
    hub_dir = tmp_path / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = hub_dir / "snapshots"
    snapshots_dir.mkdir(parents=True)

    return hub_dir, snapshots_dir


def test_resolve_local_hf_snapshot_uses_valid_main_reference(
    tmp_path, monkeypatch
):
    hub_dir, snapshots_dir = _create_hf_cache(tmp_path)
    snapshot_dir = snapshots_dir / "revision-b"
    snapshot_dir.mkdir()
    (hub_dir / "refs").mkdir()
    (hub_dir / "refs" / "main").write_text(
        "revision-b", encoding="utf-8"
    )

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(
        "aidetector.model.Path.home", lambda: tmp_path / "no-home-cache"
    )

    result = _resolve_local_hf_snapshot("example/model")

    assert result == str(snapshot_dir)


def test_resolve_local_hf_snapshot_returns_none_without_main_reference(
    tmp_path, monkeypatch
):
    _, snapshots_dir = _create_hf_cache(tmp_path)
    (snapshots_dir / "revision-a").mkdir()
    (snapshots_dir / "revision-b").mkdir()

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(
        "aidetector.model.Path.home", lambda: tmp_path / "no-home-cache"
    )

    assert _resolve_local_hf_snapshot("example/model") is None


def test_resolve_local_hf_snapshot_returns_none_for_stale_main_reference(
    tmp_path, monkeypatch
):
    hub_dir, snapshots_dir = _create_hf_cache(tmp_path)
    (snapshots_dir / "revision-a").mkdir()
    (snapshots_dir / "revision-b").mkdir()
    (hub_dir / "refs").mkdir()
    (hub_dir / "refs" / "main").write_text(
        "missing-revision", encoding="utf-8"
    )

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(
        "aidetector.model.Path.home", lambda: tmp_path / "no-home-cache"
    )

    assert _resolve_local_hf_snapshot("example/model") is None

def test_hf_cached_snapshot_failure_does_not_retry_online(
    monkeypatch, tmp_path
):
    import sys
    from types import SimpleNamespace

    import aidetector.model as model_module

    calls = []

    def fake_from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        raise OSError("incomplete local snapshot")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(),
        device=lambda value: value,
    )

    fake_transformers = SimpleNamespace(
        AutoImageProcessor=object(),
        AutoModelForImageClassification=SimpleNamespace(
            from_pretrained=fake_from_pretrained
        ),
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        model_module,
        "_resolve_local_hf_snapshot",
        lambda model_id: str(tmp_path / "snapshot"),
    )

    with pytest.raises(
        RuntimeError, match="Online fallback was not attempted"
    ):
        HuggingFaceImageDetector(model_id="example/test-model")

    assert len(calls) == 1
    assert calls[0][0][0] == str(tmp_path / "snapshot")
    assert calls[0][1]["local_files_only"] is True

def test_build_fallback_image_processor_uses_model_config(
    monkeypatch,
):
    import sys
    from types import SimpleNamespace

    captured = {}

    class FakeViTImageProcessor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_transformers = SimpleNamespace(
        ViTImageProcessor=FakeViTImageProcessor,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    config = SimpleNamespace(
        image_size=384,
        image_mean=[0.48, 0.45, 0.40],
        image_std=[0.23, 0.22, 0.22],
    )

    processor = _build_fallback_image_processor(config)

    assert isinstance(processor, FakeViTImageProcessor)
    assert captured["size"] == {"height": 384, "width": 384}
    assert captured["image_mean"] == [0.48, 0.45, 0.40]
    assert captured["image_std"] == [0.23, 0.22, 0.22]
    assert captured["do_resize"] is True
    assert captured["do_rescale"] is True
    assert captured["rescale_factor"] == 1 / 255
    assert captured["do_normalize"] is True
