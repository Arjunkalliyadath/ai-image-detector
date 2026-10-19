from __future__ import annotations

import pytest
import typer

import aidetector.cli as cli_module


@pytest.mark.parametrize(
    "backend",
    ["or-ensemble", "or-ensemble-high-recall"],
)
def test_calibrated_benchmark_rejects_fixed_threshold_or_modes(
    monkeypatch, backend
):
    def unexpected_detector_creation(*args, **kwargs):
        raise AssertionError("Detector must not be initialized for threshold calibration")

    monkeypatch.setattr(
        cli_module,
        "create_detector",
        unexpected_detector_creation,
    )

    with pytest.raises(typer.BadParameter, match="fixed component thresholds"):
        cli_module._run_calibrated_folder_benchmark(
            calibration_samples=[],
            test_samples=[],
            backend=backend,
            threshold=0.5,
            device=None,
            weight_path=None,
            model_name="test-model",
            pretrained="test-pretrained",
            hf_model="test-hf-model",
            hybrid_univfd_weight=0.5,
            hybrid_plus_primary_weight=0.5,
            ultra_primary_weight=0.5,
            hybrid_alpha_step=0.05,
            optimize_metric="f1",
            min_recall=None,
            batch_size=1,
            dataset_info={},
        )
