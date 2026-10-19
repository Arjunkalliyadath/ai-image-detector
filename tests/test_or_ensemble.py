from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

import aidetector.model as model_module
from aidetector.types import DetectionResult


def _ensemble_class():
    cls = getattr(model_module, "OREnsembleDetector", None)
    assert cls is not None, "OREnsembleDetector has not been implemented yet"
    return cls


def _fake_component(scores, backend, captures):
    class FakeDetector:
        def __init__(self, *, threshold, **kwargs):
            self.threshold = threshold
            self._scores = scores
            captures[backend] = {"threshold": threshold, **kwargs}

        def predict_images(self, images, paths=None):
            if not images:
                return []
            if paths is None:
                paths = [None] * len(images)
            if len(paths) != len(images):
                raise ValueError("paths must have the same length as images")
            if len(images) != len(self._scores):
                raise ValueError("Unexpected fake score count")

            return [
                DetectionResult(
                    path=path,
                    label="ai" if score >= self.threshold else "real",
                    probability_ai=score,
                    probability_real=1.0 - score,
                    confidence=max(score, 1.0 - score),
                    raw_score=score,
                    backend=backend,
                )
                for score, path in zip(self._scores, paths, strict=True)
            ]

        def model_info(self):
            return {"backend": backend, "threshold": self.threshold}

    return FakeDetector


def test_detection_result_without_details_preserves_legacy_schema():
    result = DetectionResult(
        path=Path("example.png"),
        label="real",
        probability_ai=0.1,
        probability_real=0.9,
        confidence=0.9,
        raw_score=0.1,
        backend="univfd",
    )

    assert set(result.as_dict()) == {
        "path",
        "label",
        "probability_ai",
        "probability_real",
        "confidence",
        "raw_score",
        "backend",
    }


def test_detection_result_serializes_optional_details():
    details = {
        "mode": "conservative",
        "triggered_by": [],
    }
    result = DetectionResult(
        path=Path("example.png"),
        label="real",
        probability_ai=0.1,
        probability_real=0.9,
        confidence=0.9,
        raw_score=0.1,
        backend="or-ensemble",
        details=details,
    )

    assert result.as_dict()["details"] == details


@pytest.mark.parametrize(
    ("score", "threshold", "expected"),
    [
        (0.0, 0.5, 0.0),
        (0.25, 0.5, 0.25),
        (0.5, 0.5, 0.5),
        (1.0, 0.5, 1.0),
        (0.5, 0.8, 0.3125),
        (0.8, 0.8, 0.5),
        (1.0, 0.8, 1.0),
    ],
)
def test_threshold_centered_score(score, threshold, expected):
    helper = getattr(model_module, "_threshold_centered_score", None)
    assert callable(helper), "Threshold score normalization has not been implemented"
    assert helper(score, threshold) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("score", "threshold"),
    [
        (-0.01, 0.5),
        (1.01, 0.5),
        (float("nan"), 0.5),
        (0.5, 0.0),
        (0.5, 1.0),
        (0.5, float("inf")),
    ],
)
def test_threshold_centered_score_rejects_invalid_values(score, threshold):
    helper = getattr(model_module, "_threshold_centered_score", None)
    assert callable(helper), "Threshold score normalization has not been implemented"
    with pytest.raises(ValueError):
        helper(score, threshold)


@pytest.mark.parametrize(
    ("backend", "mode", "nonescape_threshold", "univfd_threshold"),
    [
        ("or-ensemble", "conservative", 0.912901, 0.419977),
        ("or-ensemble-high-recall", "higher-recall", 0.639516, 0.418636),
    ],
)
def test_or_ensemble_factory_modes_and_decisions(
    monkeypatch, backend, mode, nonescape_threshold, univfd_threshold
):
    captures = {}
    nonescape_scores = [0.3, 0.95, 0.3, 0.99, 0.75]
    univfd_scores = [0.3, 0.1, 0.6, 0.9, 0.1]

    monkeypatch.setattr(
        model_module,
        "NonescapeMiniDetector",
        _fake_component(nonescape_scores, "nonescape-mini", captures),
    )
    monkeypatch.setattr(
        model_module,
        "AIImageDetector",
        _fake_component(univfd_scores, "univfd", captures),
    )

    head_path = Path("example-univfd-head.pth")
    detector = model_module.create_detector(
        backend, device="cpu", weight_path=head_path
    )

    info = detector.model_info()
    assert info["mode"] == mode
    assert info["backend"] == backend
    assert info["threshold"] == 0.5
    assert info["component_thresholds"] == {
        "nonescape-mini": nonescape_threshold,
        "univfd": univfd_threshold,
    }
    assert captures["nonescape-mini"]["threshold"] == nonescape_threshold
    assert captures["univfd"]["threshold"] == univfd_threshold
    assert captures["univfd"]["weight_path"] == head_path
    assert "weight_path" not in captures["nonescape-mini"]

    paths = [Path(f"image-{i}.png") for i in range(5)]
    results = detector.predict_images(
        [Image.new("RGB", (4, 4)) for _ in range(5)],
        paths=paths,
    )

    assert [result.path for result in results] == paths
    assert [result.label for result in results] == [
        "real",
        "ai",
        "ai",
        "ai",
        "real" if mode == "conservative" else "ai",
    ]

    for result in results:
        assert 0.0 <= result.probability_ai <= 1.0
        assert result.label == (
            "ai" if result.probability_ai >= detector.threshold else "real"
        )
        assert result.probability_real == pytest.approx(
            1.0 - result.probability_ai
        )

    payloads = [result.as_dict() for result in results]
    assert payloads[0]["details"]["triggered_by"] == []
    assert payloads[1]["details"]["triggered_by"] == ["nonescape-mini"]
    assert payloads[2]["details"]["triggered_by"] == ["univfd"]
    assert payloads[3]["details"]["triggered_by"] == [
        "nonescape-mini", "univfd"
    ]
    assert payloads[4]["details"]["triggered_by"] == (
        [] if mode == "conservative" else ["nonescape-mini"]
    )

    for payload in payloads:
        details = payload["details"]
        assert details["mode"] == mode
        assert "not a calibrated probability" in details["score_semantics"]
        assert "not a calibrated confidence estimate" in details[
            "confidence_semantics"
        ]
        assert set(details["components"]) == {"nonescape-mini", "univfd"}


def test_or_ensemble_rejects_custom_global_threshold(monkeypatch):
    _ensemble_class()
    monkeypatch.setattr(
        model_module, "AIImageDetector",
        lambda **kwargs: pytest.fail("Model should not initialize"),
    )
    monkeypatch.setattr(
        model_module, "NonescapeMiniDetector",
        lambda **kwargs: pytest.fail("Model should not initialize"),
    )

    with pytest.raises(ValueError, match="fixed component thresholds"):
        model_module.create_detector("or-ensemble", threshold=0.6)


def test_or_ensemble_rejects_unknown_mode():
    cls = _ensemble_class()
    with pytest.raises(ValueError, match="mode"):
        cls(mode="unknown-mode")


def test_or_ensemble_empty_batch_and_path_validation(monkeypatch):
    captures = {}
    monkeypatch.setattr(
        model_module,
        "NonescapeMiniDetector",
        _fake_component([], "nonescape-mini", captures),
    )
    monkeypatch.setattr(
        model_module,
        "AIImageDetector",
        _fake_component([], "univfd", captures),
    )

    detector = model_module.create_detector("or-ensemble")
    assert detector.predict_images([]) == []
    with pytest.raises(ValueError, match="paths must have the same length"):
        detector.predict_images(
            [Image.new("RGB", (4, 4))],
            paths=[],
        )


@pytest.mark.parametrize(
    ("model_name", "pretrained"),
    [
        ("ViT-B-32", "openai"),
        ("ViT-L-14", "different-pretrained-weights"),
    ],
)
def test_or_ensemble_rejects_nondefault_backbone_configuration(
    model_name, pretrained
):
    with pytest.raises(ValueError, match="thresholds were calibrated"):
        model_module.OREnsembleDetector(
            model_name=model_name,
            pretrained=pretrained,
        )


def test_or_ensemble_warns_and_records_custom_head_provenance(monkeypatch):
    captures = {}
    monkeypatch.setattr(
        model_module,
        "NonescapeMiniDetector",
        _fake_component([0.1], "nonescape-mini", captures),
    )
    monkeypatch.setattr(
        model_module,
        "AIImageDetector",
        _fake_component([0.1], "univfd", captures),
    )

    head_path = Path("custom-univfd-head.pth")
    with pytest.warns(UserWarning, match="explicit UnivFD weight path"):
        detector = model_module.create_detector(
            "or-ensemble",
            device="cpu",
            weight_path=head_path,
        )

    info = detector.model_info()
    assert info["custom_univfd_head"] is True
    assert info["threshold_calibration"] == "unverified-custom-univfd-head"
    assert info["components"]["univfd"]["head_source"] == str(head_path)



def test_or_ensemble_predict_many_preserves_paths(tmp_path, monkeypatch):
    captures = {}
    monkeypatch.setattr(
        model_module,
        "NonescapeMiniDetector",
        _fake_component([0.1, 0.95], "nonescape-mini", captures),
    )
    monkeypatch.setattr(
        model_module,
        "AIImageDetector",
        _fake_component([0.1, 0.1], "univfd", captures),
    )

    paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for image_path in paths:
        Image.new("RGB", (8, 8), "gray").save(image_path)

    detector = model_module.create_detector("or-ensemble", device="cpu")
    results = detector.predict_many(paths, batch_size=2)

    assert [result.path for result in results] == paths
    assert [result.label for result in results] == ["real", "ai"]



def test_or_ensemble_model_info_records_checkpoint_fingerprints(
    tmp_path, monkeypatch
):
    captures = {}
    monkeypatch.setattr(
        model_module,
        "NonescapeMiniDetector",
        _fake_component([0.1], "nonescape-mini", captures),
    )
    monkeypatch.setattr(
        model_module,
        "AIImageDetector",
        _fake_component([0.1], "univfd", captures),
    )

    mini_checkpoint = tmp_path / "nonescape-mini.safetensors"
    head_checkpoint = tmp_path / "univfd-head.pth"
    mini_checkpoint.write_bytes(b"test mini checkpoint bytes")
    head_checkpoint.write_bytes(b"test UnivFD head checkpoint bytes")

    detector = model_module.create_detector("or-ensemble", device="cpu")
    detector.nonescape.weight_file = mini_checkpoint
    detector.univfd.head_weight_path = head_checkpoint

    fingerprint_calls = []
    original_fingerprint = model_module._checkpoint_fingerprint

    def track_fingerprint(checkpoint_path):
        fingerprint_calls.append(Path(checkpoint_path))
        return original_fingerprint(checkpoint_path)

    monkeypatch.setattr(
        model_module, "_checkpoint_fingerprint", track_fingerprint
    )

    info = detector.model_info()
    fingerprints = info["checkpoint_fingerprints"]

    expected_mini_hash = hashlib.sha256(mini_checkpoint.read_bytes()).hexdigest()
    expected_head_hash = hashlib.sha256(head_checkpoint.read_bytes()).hexdigest()

    assert fingerprints["nonescape-mini"] == {
        "source": str(mini_checkpoint),
        "sha256": expected_mini_hash,
    }
    assert fingerprints["univfd-head"] == {
        "source": str(head_checkpoint),
        "sha256": expected_head_hash,
    }
    assert info["threshold_calibration"] == "unverified-default-univfd-head"
    assert info["calibration_checkpoint_identity_verified"] is False
    assert (
        "not verified against calibration-time checkpoint hashes"
        in info["calibration_checkpoint_verification_note"]
    )
    assert len(fingerprint_calls) == 2

    repeated_info = detector.model_info()
    assert len(fingerprint_calls) == 2

    repeated_info["checkpoint_fingerprints"]["nonescape-mini"]["sha256"] = (
        "modified"
    )
    assert (
        detector.model_info()["checkpoint_fingerprints"]["nonescape-mini"][
            "sha256"
        ]
        == expected_mini_hash
    )
