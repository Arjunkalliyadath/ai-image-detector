from __future__ import annotations

import pytest
from PIL import Image

from aidetector.checkpoint_utils import load_torch_checkpoint
from aidetector.model import iter_images

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
