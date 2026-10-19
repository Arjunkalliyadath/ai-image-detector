from __future__ import annotations

from pathlib import Path

from rich.console import Console

import aidetector.cli as cli_module
from aidetector.types import DetectionResult


def _result(*, details=None):
    return DetectionResult(
        path=Path("example.png"),
        label="ai",
        probability_ai=0.62,
        probability_real=0.38,
        confidence=0.62,
        raw_score=0.62,
        backend="univfd" if details is None else "or-ensemble",
        details=details,
    )


def test_standard_detection_table_keeps_probability_columns(monkeypatch):
    recorded_console = Console(record=True, width=180)
    monkeypatch.setattr(cli_module, "console", recorded_console)

    cli_module._print_detection_table([_result()])
    output = recorded_console.export_text()

    assert "AI Prob" in output
    assert "Real Prob" in output
    assert "Confidence" in output
    assert "not a calibrated probability" not in output


def test_or_ensemble_table_explains_uncalibrated_score(monkeypatch):
    recorded_console = Console(record=True, width=180)
    monkeypatch.setattr(cli_module, "console", recorded_console)

    result = _result(
        details={
            "mode": "conservative",
            "score_semantics": (
                "Composite threshold-normalized evidence score; "
                "not a calibrated probability."
            ),
            "triggered_by": ["univfd"],
        }
    )

    cli_module._print_detection_table([result])
    output = recorded_console.export_text()

    assert "OR Score" in output
    assert "not a calibrated probability" in output
    assert "AI Prob" not in output
    assert "Real Prob" not in output
