from __future__ import annotations

import pickle
from pathlib import Path


class SafeCheckpointLoadError(RuntimeError):
    """Raised when a checkpoint cannot be loaded safely."""


def load_torch_checkpoint(torch_module, weight_path: str | Path):
    """Load tensor weights without permitting unrestricted pickle loading."""
    try:
        return torch_module.load(
            str(weight_path),
            map_location="cpu",
            weights_only=True,
        )
    except pickle.UnpicklingError as exc:
        raise SafeCheckpointLoadError(
            "Checkpoint was rejected by PyTorch's safe weights-only loader. "
            "It may contain unsupported serialized objects; unrestricted pickle "
            "loading is disabled. Use a verified weights-only checkpoint or another backend."
        ) from exc
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        raise SafeCheckpointLoadError(
            "Safe checkpoint loading requires a PyTorch version that supports "
            "weights_only=True. Upgrade PyTorch; unrestricted loading is disabled."
        ) from exc
