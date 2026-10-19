from __future__ import annotations

import csv
import json
from pathlib import Path

from aidetector.cli import _write_csv
from aidetector.types import DetectionResult

LEGACY_FIELDS = [
    "path",
    "label",
    "probability_ai",
    "probability_real",
    "confidence",
    "raw_score",
    "backend",
]


def _result(*, details=None):
    return DetectionResult(
        path=Path("example.png"),
        label="real",
        probability_ai=0.1,
        probability_real=0.9,
        confidence=0.9,
        raw_score=0.1,
        backend="univfd" if details is None else "or-ensemble",
        details=details,
    )


def test_write_csv_preserves_legacy_columns_without_details(tmp_path):
    output = tmp_path / "results.csv"

    _write_csv([_result()], output)

    with output.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == LEGACY_FIELDS
    assert len(rows) == 1
    assert rows[0]["backend"] == "univfd"


def test_write_csv_serializes_ensemble_details_as_json(tmp_path):
    details = {
        "mode": "conservative",
        "triggered_by": ["univfd"],
    }
    output = tmp_path / "ensemble.csv"

    _write_csv([_result(details=details)], output)

    with output.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == LEGACY_FIELDS + ["details"]
    assert len(rows) == 1
    assert json.loads(rows[0]["details"]) == details



def test_write_csv_includes_details_for_empty_or_ensemble_output(tmp_path):
    output = tmp_path / "empty-ensemble.csv"

    _write_csv([], output, include_details=True)

    with output.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == LEGACY_FIELDS + ["details"]
    assert rows == []
