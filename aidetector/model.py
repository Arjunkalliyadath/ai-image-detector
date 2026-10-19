from __future__ import annotations

import hashlib
import math
import os
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar, Protocol

from PIL import Image, ImageOps

from .checkpoint_utils import load_torch_checkpoint
from .config import (
    DEFAULT_BACKEND,
    DEFAULT_HF_MODEL_ID,
    DEFAULT_HYBRID_PLUS_PRIMARY_WEIGHT,
    DEFAULT_HYBRID_UNIVFD_WEIGHT,
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    DEFAULT_ULTRA_PRIMARY_WEIGHT,
    FC_WEIGHT_PATH_IN_REPO,
    IMAGE_EXTENSIONS,
    MODEL_REPO_ID,
)
from .nonescape_adapter import NonescapeFullDetector, NonescapeMiniDetector
from .sentry_adapter import SentryConvNeXtDetector
from .types import DetectionResult


class Detector(Protocol):
    threshold: float

    def predict_image(self, image: Image.Image, path: Path | None = None) -> DetectionResult:
        ...

    def predict_images(
        self,
        images: Sequence[Image.Image],
        paths: Sequence[Path | None] | None = None,
    ) -> list[DetectionResult]:
        ...

    def predict_path(self, path: str | Path) -> DetectionResult:
        ...

    def model_info(self) -> dict:
        ...


def _select_device(torch_module, requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _normalise_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def _threshold_centered_score(score: float, threshold: float) -> float:
    """Map a component score to [0, 1], centering its decision threshold at 0.5.

    This monotonic transformation preserves the component's threshold decision.
    The result is a composite evidence score when used in an ensemble, not a
    calibrated real-world probability.
    """
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be finite and between 0 and 1")
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be finite and strictly between 0 and 1")

    if score < threshold:
        return score / (2.0 * threshold)
    return 0.5 + (score - threshold) / (2.0 * (1.0 - threshold))


def _checkpoint_fingerprint(path: str | Path) -> dict[str, str]:
    """Return the resolved checkpoint path and its SHA-256 digest."""
    checkpoint_path = Path(path).resolve()
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"source": str(checkpoint_path), "sha256": digest.hexdigest()}


def _build_detection_result(
    *,
    path: Path | None,
    probability_ai: float,
    raw_score: float,
    threshold: float,
    backend: str,
) -> DetectionResult:
    probability_real = 1.0 - probability_ai
    label = "ai" if probability_ai >= threshold else "real"
    confidence = probability_ai if label == "ai" else probability_real
    return DetectionResult(
        path=path,
        label=label,
        probability_ai=probability_ai,
        probability_real=probability_real,
        confidence=confidence,
        raw_score=raw_score,
        backend=backend,
    )


class AIImageDetector:
    """UnivFD-style detector: CLIP image encoder plus a linear fake/real head."""

    backend_name = "univfd"

    def __init__(
        self,
        device: str | None = None,
        threshold: float = 0.5,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        weight_path: str | Path | None = None,
        repo_id: str = MODEL_REPO_ID,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install torch to use the UnivFD backend.") from exc
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install open_clip_torch to use the UnivFD backend.") from exc

        self._torch = torch
        self.device = torch.device(_select_device(torch, device))
        self.threshold = threshold
        self.model_name = model_name
        self.pretrained = pretrained
        self.repo_id = repo_id

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.clip_model.eval()

        output_dim = getattr(getattr(self.clip_model, "visual", None), "output_dim", 768)
        self.head = torch.nn.Linear(int(output_dim), 1).to(self.device)
        self._load_head(weight_path)
        self.head.eval()

    def _load_head(self, weight_path: str | Path | None) -> None:
        if weight_path is None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependency: install huggingface_hub or pass --weight-path."
                ) from exc
            weight_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=FC_WEIGHT_PATH_IN_REPO,
            )

        self.head_weight_path = Path(weight_path).resolve()
        state = load_torch_checkpoint(self._torch, weight_path)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        if isinstance(state, dict):
            cleaned = {}
            for key, value in state.items():
                key = key.replace("module.", "")
                key = key.replace("model.", "")
                key = key.replace("fc.", "")
                key = key.replace("head.", "")
                if key in {"weight", "bias"}:
                    cleaned[key] = value
            if cleaned:
                self.head.load_state_dict(cleaned, strict=True)
                return
            try:
                self.head.load_state_dict(state, strict=False)
                return
            except Exception as exc:
                raise RuntimeError(f"Could not load UnivFD head weights from {weight_path}: {exc}") from exc
        raise RuntimeError(f"Unsupported checkpoint format: {type(state)!r}")

    def model_info(self) -> dict:
        return {
            "backend": self.backend_name,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "head_repo": self.repo_id,
            "head_file": FC_WEIGHT_PATH_IN_REPO,
            "device": str(self.device),
            "threshold": self.threshold,
        }

    @property
    def _torch_inference_mode(self):
        return self._torch.inference_mode

    def predict_image(self, image: Image.Image, path: Path | None = None) -> DetectionResult:
        return self.predict_images([image], paths=[path])[0]

    def predict_images(
        self,
        images: Sequence[Image.Image],
        paths: Sequence[Path | None] | None = None,
    ) -> list[DetectionResult]:
        if not images:
            return []
        if paths is None:
            paths = [None] * len(images)
        if len(paths) != len(images):
            raise ValueError("paths must have the same length as images")

        torch = self._torch
        tensors = [self.preprocess(_normalise_image(image)) for image in images]
        batch = torch.stack(tensors, dim=0).to(self.device)
        with torch.inference_mode():
            features = self.clip_model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            logits = self.head(features).flatten().float().detach().cpu()
            probabilities = torch.sigmoid(logits)

        results: list[DetectionResult] = []
        for probability_ai_tensor, logit_tensor, path in zip(probabilities, logits, paths, strict=True):
            results.append(
                _build_detection_result(
                    path=path,
                    probability_ai=float(probability_ai_tensor.item()),
                    raw_score=float(logit_tensor.item()),
                    threshold=self.threshold,
                    backend=self.backend_name,
                )
            )
        return results

    def predict_path(self, path: str | Path) -> DetectionResult:
        path = Path(path)
        with Image.open(path) as image:
            return self.predict_image(image, path=path)

    def predict_many(
        self,
        paths: Iterable[str | Path],
        batch_size: int = 16,
    ) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        batch_images: list[Image.Image] = []
        batch_paths: list[Path] = []
        for path_like in paths:
            path = Path(path_like)
            with Image.open(path) as image:
                batch_images.append(_normalise_image(image))
            batch_paths.append(path)
            if len(batch_images) >= batch_size:
                results.extend(self.predict_images(batch_images, paths=batch_paths))
                batch_images = []
                batch_paths = []
        if batch_images:
            results.extend(self.predict_images(batch_images, paths=batch_paths))
        return results


class HuggingFaceImageDetector:
    """Generic Hugging Face image-classification backend."""

    backend_name = "hf"

    def __init__(
        self,
        model_id: str = DEFAULT_HF_MODEL_ID,
        device: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install torch to use the Hugging Face backend.") from exc
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install transformers to use the HF backend.") from exc

        self._torch = torch
        self.device = torch.device(_select_device(torch, device))
        self.threshold = threshold
        self.model_id = model_id
        cached_snapshot = _resolve_local_hf_snapshot(model_id)
        model_source = cached_snapshot or model_id
        local_only = cached_snapshot is not None

        try:
            self.model = AutoModelForImageClassification.from_pretrained(
                model_source,
                local_files_only=local_only,
            ).to(self.device)
        except Exception as first_exc:
            # A cached snapshot failed: do not silently retry online.
            if cached_snapshot is not None:
                raise RuntimeError(
                    "Could not load the cached Hugging Face model snapshot. "
                    "Online fallback was not attempted. Check the cached "
                    f"model files for {model_id!r}. Error: {first_exc}"
                ) from first_exc

            # No cached snapshot exists, so allow a normal online download.
            try:
                self.model = AutoModelForImageClassification.from_pretrained(
                    model_id,
                ).to(self.device)
            except Exception as second_exc:
                raise RuntimeError(
                    "Could not load the requested Hugging Face model as a standard "
                    "Transformers image-classification checkpoint. Some open-source "
                    "detectors, such as custom Sentry or Nonescape releases, need a "
                    "dedicated adapter instead of the generic --backend hf path. "
                    f"Model: {model_id}. Last error: {second_exc}"
                ) from second_exc

        self.model.eval()
        self.id2label = self.model.config.id2label

        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_source,
                local_files_only=local_only,
            )
        except Exception:  # noqa: BLE001
            # Some custom checkpoints lack a compatible processor config.
            self.processor = _build_fallback_image_processor(self.model.config)

    def model_info(self) -> dict:
        return {
            "backend": self.backend_name,
            "model_id": self.model_id,
            "device": str(self.device),
            "threshold": self.threshold,
            "id2label": {str(k): v for k, v in self.id2label.items()},
        }

    def predict_image(self, image: Image.Image, path: Path | None = None) -> DetectionResult:
        return self.predict_images([image], paths=[path])[0]

    def predict_images(
        self,
        images: Sequence[Image.Image],
        paths: Sequence[Path | None] | None = None,
    ) -> list[DetectionResult]:
        if not images:
            return []
        if paths is None:
            paths = [None] * len(images)
        if len(paths) != len(images):
            raise ValueError("paths must have the same length as images")

        torch = self._torch
        normalised = [_normalise_image(image) for image in images]
        inputs = self.processor(images=normalised, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits.detach().cpu()
            probabilities = torch.softmax(logits, dim=-1)

        fake_indices = self._fake_label_indices()
        if not fake_indices:
            raise RuntimeError(f"Could not infer fake/AI label from model labels: {self.id2label}")

        results: list[DetectionResult] = []
        for row, logit_row, path in zip(probabilities, logits, paths, strict=True):
            results.append(
                _build_detection_result(
                    path=path,
                    probability_ai=float(row[fake_indices].sum().item()),
                    raw_score=float(logit_row[fake_indices].mean().item()),
                    threshold=self.threshold,
                    backend=self.backend_name,
                )
            )
        return results

    def predict_path(self, path: str | Path) -> DetectionResult:
        path = Path(path)
        with Image.open(path) as image:
            return self.predict_image(image, path=path)

    def predict_many(
        self,
        paths: Iterable[str | Path],
        batch_size: int = 16,
    ) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        batch_images: list[Image.Image] = []
        batch_paths: list[Path] = []
        for path_like in paths:
            path = Path(path_like)
            with Image.open(path) as image:
                batch_images.append(_normalise_image(image))
            batch_paths.append(path)
            if len(batch_images) >= batch_size:
                results.extend(self.predict_images(batch_images, paths=batch_paths))
                batch_images = []
                batch_paths = []
        if batch_images:
            results.extend(self.predict_images(batch_images, paths=batch_paths))
        return results

    def _fake_label_indices(self) -> list[int]:
        fake_tokens = ("fake", "ai", "aigc", "synthetic", "generated")
        real_tokens = ("real", "human", "authentic", "natural")
        fake_indices: list[int] = []
        for index, label in self.id2label.items():
            label_l = str(label).lower()
            is_fake = any(token in label_l for token in fake_tokens)
            is_real = any(token in label_l for token in real_tokens)
            if is_fake and not is_real:
                fake_indices.append(int(index))
        return fake_indices


class HybridImageDetector:
    """Blend UnivFD and HF image classifier scores for a stronger practical baseline."""

    backend_name = "hybrid"

    def __init__(
        self,
        *,
        device: str | None = None,
        threshold: float = 0.5,
        univfd_weight: float = DEFAULT_HYBRID_UNIVFD_WEIGHT,
        weight_path: str | Path | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        hf_model: str = DEFAULT_HF_MODEL_ID,
    ) -> None:
        if not 0.0 <= univfd_weight <= 1.0:
            raise ValueError("univfd_weight must be between 0 and 1")
        self.threshold = threshold
        self.univfd_weight = univfd_weight
        self.univfd = AIImageDetector(
            device=device,
            threshold=threshold,
            model_name=model_name,
            pretrained=pretrained,
            weight_path=weight_path,
        )
        self.hf = HuggingFaceImageDetector(
            model_id=hf_model,
            device=device,
            threshold=threshold,
        )

    def model_info(self) -> dict:
        return {
            "backend": self.backend_name,
            "threshold": self.threshold,
            "univfd_weight": self.univfd_weight,
            "hf_weight": round(1.0 - self.univfd_weight, 6),
            "components": {
                "univfd": self.univfd.model_info(),
                "hf": self.hf.model_info(),
            },
        }

    def predict_image(self, image: Image.Image, path: Path | None = None) -> DetectionResult:
        return self.predict_images([image], paths=[path])[0]

    def predict_images(
        self,
        images: Sequence[Image.Image],
        paths: Sequence[Path | None] | None = None,
    ) -> list[DetectionResult]:
        if not images:
            return []
        if paths is None:
            paths = [None] * len(images)
        if len(paths) != len(images):
            raise ValueError("paths must have the same length as images")

        univfd_results = self.univfd.predict_images(images, paths=paths)
        hf_results = self.hf.predict_images(images, paths=paths)
        results: list[DetectionResult] = []
        for univfd_result, hf_result, path in zip(univfd_results, hf_results, paths, strict=True):
            probability_ai = (
                self.univfd_weight * univfd_result.probability_ai
                + (1.0 - self.univfd_weight) * hf_result.probability_ai
            )
            raw_score = (
                self.univfd_weight * univfd_result.raw_score
                + (1.0 - self.univfd_weight) * hf_result.raw_score
            )
            results.append(
                _build_detection_result(
                    path=path,
                    probability_ai=probability_ai,
                    raw_score=raw_score,
                    threshold=self.threshold,
                    backend=self.backend_name,
                )
            )
        return results


class HybridPlusDetector:
    """Blend the existing hybrid detector with Nonescape Mini for a stronger practical ensemble."""

    backend_name = "hybrid-plus"

    def __init__(
        self,
        *,
        device: str | None = None,
        threshold: float = 0.5,
        primary_weight: float = DEFAULT_HYBRID_PLUS_PRIMARY_WEIGHT,
        weight_path: str | Path | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        hf_model: str = DEFAULT_HF_MODEL_ID,
        hybrid_univfd_weight: float = DEFAULT_HYBRID_UNIVFD_WEIGHT,
    ) -> None:
        if not 0.0 <= primary_weight <= 1.0:
            raise ValueError("primary_weight must be between 0 and 1")
        self.threshold = threshold
        self.primary_weight = primary_weight
        self.primary = HybridImageDetector(
            device=device,
            threshold=threshold,
            univfd_weight=hybrid_univfd_weight,
            weight_path=weight_path,
            model_name=model_name,
            pretrained=pretrained,
            hf_model=hf_model,
        )
        self.secondary = NonescapeMiniDetector(
            threshold=threshold,
            device=device,
        )

    def model_info(self) -> dict:
        return {
            "backend": self.backend_name,
            "threshold": self.threshold,
            "primary_weight": self.primary_weight,
            "secondary_weight": round(1.0 - self.primary_weight, 6),
            "components": {
                "primary": self.primary.model_info(),
                "secondary": self.secondary.model_info(),
            },
        }

    def predict_image(self, image: Image.Image, path: Path | None = None) -> DetectionResult:
        return self.predict_images([image], paths=[path])[0]

    def predict_images(
        self,
        images: Sequence[Image.Image],
        paths: Sequence[Path | None] | None = None,
    ) -> list[DetectionResult]:
        if not images:
            return []
        if paths is None:
            paths = [None] * len(images)
        if len(paths) != len(images):
            raise ValueError("paths must have the same length as images")

        primary_results = self.primary.predict_images(images, paths=paths)
        secondary_results = self.secondary.predict_images(images, paths=paths)
        results: list[DetectionResult] = []
        for primary_result, secondary_result, path in zip(primary_results, secondary_results, paths, strict=True):
            probability_ai = (
                self.primary_weight * primary_result.probability_ai
                + (1.0 - self.primary_weight) * secondary_result.probability_ai
            )
            raw_score = (
                self.primary_weight * primary_result.raw_score
                + (1.0 - self.primary_weight) * secondary_result.raw_score
            )
            results.append(
                _build_detection_result(
                    path=path,
                    probability_ai=probability_ai,
                    raw_score=raw_score,
                    threshold=self.threshold,
                    backend=self.backend_name,
                )
            )
        return results

    def predict_path(self, path: str | Path) -> DetectionResult:
        path = Path(path)
        with Image.open(path) as image:
            return self.predict_image(image, path=path)


class UltraDetector:
    """Blend HybridPlus and Sentry ConvNeXt into the strongest current practical ensemble."""

    backend_name = "ultra"

    def __init__(
        self,
        *,
        device: str | None = None,
        threshold: float = 0.5,
        primary_weight: float = DEFAULT_ULTRA_PRIMARY_WEIGHT,
        weight_path: str | Path | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        hf_model: str = DEFAULT_HF_MODEL_ID,
        hybrid_univfd_weight: float = DEFAULT_HYBRID_UNIVFD_WEIGHT,
        hybrid_plus_primary_weight: float = DEFAULT_HYBRID_PLUS_PRIMARY_WEIGHT,
    ) -> None:
        if not 0.0 <= primary_weight <= 1.0:
            raise ValueError("primary_weight must be between 0 and 1")
        self.threshold = threshold
        self.primary_weight = primary_weight
        self.primary = HybridPlusDetector(
            device=device,
            threshold=threshold,
            primary_weight=hybrid_plus_primary_weight,
            weight_path=weight_path,
            model_name=model_name,
            pretrained=pretrained,
            hf_model=hf_model,
            hybrid_univfd_weight=hybrid_univfd_weight,
        )
        self.secondary = SentryConvNeXtDetector(
            threshold=threshold,
            device=device,
        )

    def model_info(self) -> dict:
        return {
            "backend": self.backend_name,
            "threshold": self.threshold,
            "primary_weight": self.primary_weight,
            "secondary_weight": round(1.0 - self.primary_weight, 6),
            "components": {
                "primary": self.primary.model_info(),
                "secondary": self.secondary.model_info(),
            },
        }

    def predict_image(self, image: Image.Image, path: Path | None = None) -> DetectionResult:
        return self.predict_images([image], paths=[path])[0]

    def predict_images(
        self,
        images: Sequence[Image.Image],
        paths: Sequence[Path | None] | None = None,
    ) -> list[DetectionResult]:
        if not images:
            return []
        if paths is None:
            paths = [None] * len(images)
        if len(paths) != len(images):
            raise ValueError("paths must have the same length as images")

        primary_results = self.primary.predict_images(images, paths=paths)
        secondary_results = self.secondary.predict_images(images, paths=paths)
        results: list[DetectionResult] = []
        for primary_result, secondary_result, path in zip(primary_results, secondary_results, paths, strict=True):
            probability_ai = (
                self.primary_weight * primary_result.probability_ai
                + (1.0 - self.primary_weight) * secondary_result.probability_ai
            )
            raw_score = (
                self.primary_weight * primary_result.raw_score
                + (1.0 - self.primary_weight) * secondary_result.raw_score
            )
            results.append(
                _build_detection_result(
                    path=path,
                    probability_ai=probability_ai,
                    raw_score=raw_score,
                    threshold=self.threshold,
                    backend=self.backend_name,
                )
            )
        return results

    def predict_path(self, path: str | Path) -> DetectionResult:
        path = Path(path)
        with Image.open(path) as image:
            return self.predict_image(image, path=path)

    def predict_many(
        self,
        paths: Iterable[str | Path],
        batch_size: int = 16,
    ) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        batch_images: list[Image.Image] = []
        batch_paths: list[Path] = []
        for path_like in paths:
            path = Path(path_like)
            with Image.open(path) as image:
                batch_images.append(_normalise_image(image))
            batch_paths.append(path)
            if len(batch_images) >= batch_size:
                results.extend(self.predict_images(batch_images, paths=batch_paths))
                batch_images = []
                batch_paths = []
        if batch_images:
            results.extend(self.predict_images(batch_images, paths=batch_paths))
        return results


class OREnsembleDetector:
    """OR ensemble using fixed, threshold-normalized component evidence scores."""

    backend_name = "or-ensemble"
    threshold = 0.5

    CONSERVATIVE_THRESHOLDS: ClassVar[dict[str, float]] = {
        "nonescape-mini": 0.912901,
        "univfd": 0.419977,
    }
    HIGH_RECALL_THRESHOLDS: ClassVar[dict[str, float]] = {
        "nonescape-mini": 0.639516,
        "univfd": 0.418636,
    }
    SCORE_SEMANTICS = (
        "Composite threshold-normalized evidence score; not a calibrated probability."
    )

    def __init__(
        self,
        *,
        mode: str = "conservative",
        device: str | None = None,
        threshold: float = 0.5,
        weight_path: str | Path | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        backend_name: str | None = None,
    ) -> None:
        if model_name != DEFAULT_MODEL_NAME or pretrained != DEFAULT_PRETRAINED:
            raise ValueError(
                "OR ensemble thresholds were calibrated for "
                f"model_name={DEFAULT_MODEL_NAME!r} and "
                f"pretrained={DEFAULT_PRETRAINED!r}; custom backbone "
                "configurations are not supported."
            )

        if mode not in {"conservative", "higher-recall"}:
            raise ValueError(
                "mode must be 'conservative' or 'higher-recall'"
            )
        if threshold != 0.5:
            raise ValueError(
                "OR ensemble uses fixed component thresholds and a global threshold of 0.5"
            )

        if weight_path is not None:
            warnings.warn(
                "Using an explicit UnivFD weight path with the OR ensemble. "
                "The thresholds were selected using the default UnivFD "
                "configuration; compatibility with this custom head is "
                "unverified, and calibration-time checkpoint identity has "
                "not been verified.",
                UserWarning,
                stacklevel=2,
            )

        self.mode = mode
        self.weight_path = weight_path
        self.backend_name = backend_name or (
            "or-ensemble"
            if mode == "conservative"
            else "or-ensemble-high-recall"
        )
        self.threshold = 0.5
        selected_thresholds = (
            self.CONSERVATIVE_THRESHOLDS
            if mode == "conservative"
            else self.HIGH_RECALL_THRESHOLDS
        )
        self.component_thresholds = dict(selected_thresholds)

        self.nonescape = NonescapeMiniDetector(
            threshold=self.component_thresholds["nonescape-mini"],
            device=device,
        )
        self.univfd = AIImageDetector(
            device=device,
            threshold=self.component_thresholds["univfd"],
            model_name=model_name,
            pretrained=pretrained,
            weight_path=weight_path,
        )

    def model_info(self) -> dict:
        univfd_info = self.univfd.model_info()
        univfd_info["head_source"] = (
            str(self.weight_path)
            if self.weight_path is not None
            else f"{MODEL_REPO_ID}/{FC_WEIGHT_PATH_IN_REPO}"
        )
        uses_custom_head = self.weight_path is not None

        cached_fingerprints = getattr(
            self, "_checkpoint_fingerprints_cache", None
        )
        if cached_fingerprints is None:
            cached_fingerprints = {}
            mini_path = getattr(self.nonescape, "weight_file", None)
            if mini_path is not None and Path(mini_path).is_file():
                cached_fingerprints["nonescape-mini"] = _checkpoint_fingerprint(
                    mini_path
                )

            head_path = getattr(self.univfd, "head_weight_path", None)
            if head_path is None:
                head_path = self.weight_path
            if head_path is not None and Path(head_path).is_file():
                cached_fingerprints["univfd-head"] = _checkpoint_fingerprint(
                    head_path
                )
            self._checkpoint_fingerprints_cache = cached_fingerprints

        checkpoint_fingerprints = {
            name: dict(fingerprint)
            for name, fingerprint in cached_fingerprints.items()
        }

        return {
            "backend": self.backend_name,
            "mode": self.mode,
            "threshold": self.threshold,
            "component_thresholds": dict(self.component_thresholds),
            "score_semantics": self.SCORE_SEMANTICS,
            "custom_univfd_head": uses_custom_head,
            "threshold_calibration": (
                "unverified-custom-univfd-head"
                if uses_custom_head
                else "unverified-default-univfd-head"
            ),
            "calibration_checkpoint_identity_verified": False,
            "calibration_checkpoint_verification_note": (
                "Runtime checkpoint fingerprints identify current files; they "
                "are not verified against calibration-time checkpoint hashes."
            ),
            "checkpoint_fingerprints": checkpoint_fingerprints,
            "components": {
                "nonescape-mini": self.nonescape.model_info(),
                "univfd": univfd_info,
            },
        }

    def predict_image(
        self,
        image: Image.Image,
        path: Path | None = None,
    ) -> DetectionResult:
        return self.predict_images([image], paths=[path])[0]

    def predict_path(self, path: str | Path) -> DetectionResult:
        path = Path(path)
        with Image.open(path) as image:
            normalized = _normalise_image(image)
        return self.predict_image(normalized, path=path)

    def predict_images(
        self,
        images: Sequence[Image.Image],
        paths: Sequence[Path | None] | None = None,
    ) -> list[DetectionResult]:
        if not images:
            return []
        if paths is None:
            paths = [None] * len(images)
        if len(paths) != len(images):
            raise ValueError("paths must have the same length as images")

        nonescape_results = self.nonescape.predict_images(images, paths=paths)
        univfd_results = self.univfd.predict_images(images, paths=paths)

        results: list[DetectionResult] = []
        for nonescape_result, univfd_result, path in zip(
            nonescape_results,
            univfd_results,
            paths,
            strict=True,
        ):
            nonescape_score = nonescape_result.probability_ai
            univfd_score = univfd_result.probability_ai

            nonescape_threshold = self.component_thresholds["nonescape-mini"]
            univfd_threshold = self.component_thresholds["univfd"]

            nonescape_normalized = _threshold_centered_score(
                nonescape_score, nonescape_threshold
            )
            univfd_normalized = _threshold_centered_score(
                univfd_score, univfd_threshold
            )
            composite_score = max(nonescape_normalized, univfd_normalized)

            triggered_by = []
            if nonescape_score >= nonescape_threshold:
                triggered_by.append("nonescape-mini")
            if univfd_score >= univfd_threshold:
                triggered_by.append("univfd")

            details = {
                "mode": self.mode,
                "score_semantics": self.SCORE_SEMANTICS,
                "confidence_semantics": (
                    "For OR ensemble results, confidence is the larger of "
                    "the composite evidence score and its complement; it is "
                    "not a calibrated confidence estimate."
                ),
                "triggered_by": triggered_by,
                "components": {
                    "nonescape-mini": {
                        "probability_ai": nonescape_score,
                        "probability_real": nonescape_result.probability_real,
                        "raw_score": nonescape_result.raw_score,
                        "threshold": nonescape_threshold,
                        "label": nonescape_result.label,
                        "triggered": "nonescape-mini" in triggered_by,
                    },
                    "univfd": {
                        "probability_ai": univfd_score,
                        "probability_real": univfd_result.probability_real,
                        "raw_score": univfd_result.raw_score,
                        "threshold": univfd_threshold,
                        "label": univfd_result.label,
                        "triggered": "univfd" in triggered_by,
                    },
                },
            }

            label = "ai" if composite_score >= self.threshold else "real"
            probability_real = 1.0 - composite_score
            confidence = (
                composite_score if label == "ai" else probability_real
            )
            results.append(
                DetectionResult(
                    path=path,
                    label=label,
                    probability_ai=composite_score,
                    probability_real=probability_real,
                    confidence=confidence,
                    raw_score=composite_score,
                    backend=self.backend_name,
                    details=details,
                )
            )

        return results


    def predict_many(
        self,
        paths: Iterable[str | Path],
        batch_size: int = 16,
    ) -> list[DetectionResult]:
        """Predict images from paths in batches."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        results: list[DetectionResult] = []
        batch_images: list[Image.Image] = []
        batch_paths: list[Path] = []

        for path_like in paths:
            image_path = Path(path_like)
            with Image.open(image_path) as image:
                batch_images.append(_normalise_image(image))
            batch_paths.append(image_path)

            if len(batch_images) >= batch_size:
                results.extend(
                    self.predict_images(batch_images, paths=batch_paths)
                )
                batch_images = []
                batch_paths = []

        if batch_images:
            results.extend(
                self.predict_images(batch_images, paths=batch_paths)
            )

        return results


def create_detector(
    backend: str = DEFAULT_BACKEND,
    *,
    device: str | None = None,
    threshold: float = 0.5,
    weight_path: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: str = DEFAULT_PRETRAINED,
    hf_model: str = DEFAULT_HF_MODEL_ID,
    hybrid_univfd_weight: float = DEFAULT_HYBRID_UNIVFD_WEIGHT,
    hybrid_plus_primary_weight: float = DEFAULT_HYBRID_PLUS_PRIMARY_WEIGHT,
    ultra_primary_weight: float = DEFAULT_ULTRA_PRIMARY_WEIGHT,
) -> Detector:
    backend = backend.lower()
    if backend in {"or-ensemble", "or-ensemble-high-recall"}:
        if threshold != 0.5:
            raise ValueError(
                "OR ensemble uses fixed component thresholds and a global threshold of 0.5"
            )
        mode = (
            "conservative"
            if backend == "or-ensemble"
            else "higher-recall"
        )
        return OREnsembleDetector(
            mode=mode,
            device=device,
            threshold=threshold,
            weight_path=weight_path,
            model_name=model_name,
            pretrained=pretrained,
            backend_name=backend,
        )
    if backend == "univfd":
        return AIImageDetector(
            device=device,
            threshold=threshold,
            model_name=model_name,
            pretrained=pretrained,
            weight_path=weight_path,
        )
    if backend in {"hf", "huggingface"}:
        return HuggingFaceImageDetector(model_id=hf_model, device=device, threshold=threshold)
    if backend in {"nonescape", "nonescape-full"}:
        return NonescapeFullDetector(
            threshold=threshold,
            device=device,
            weight_path=weight_path,
        )
    if backend in {"nonescape-mini", "nonescape_mini"}:
        return NonescapeMiniDetector(
            threshold=threshold,
            device=device,
            weight_path=weight_path,
        )
    if backend in {"sentry", "sentry-convnext", "sentry-convnext-small"}:
        return SentryConvNeXtDetector(
            threshold=threshold,
            device=device,
            weight_path=weight_path,
        )
    if backend in {"hybrid", "ensemble"}:
        return HybridImageDetector(
            device=device,
            threshold=threshold,
            univfd_weight=hybrid_univfd_weight,
            weight_path=weight_path,
            model_name=model_name,
            pretrained=pretrained,
            hf_model=hf_model,
        )
    if backend in {"hybrid-plus", "hybrid_plus", "stacked"}:
        return HybridPlusDetector(
            device=device,
            threshold=threshold,
            primary_weight=hybrid_plus_primary_weight,
            weight_path=weight_path,
            model_name=model_name,
            pretrained=pretrained,
            hf_model=hf_model,
            hybrid_univfd_weight=hybrid_univfd_weight,
        )
    if backend in {"ultra", "hybrid-ultra", "sentry-plus"}:
        return UltraDetector(
            device=device,
            threshold=threshold,
            primary_weight=ultra_primary_weight,
            weight_path=weight_path,
            model_name=model_name,
            pretrained=pretrained,
            hf_model=hf_model,
            hybrid_univfd_weight=hybrid_univfd_weight,
            hybrid_plus_primary_weight=hybrid_plus_primary_weight,
        )
    raise ValueError(
        "Unsupported backend: "
        f"{backend!r}. Choose 'univfd', 'hf', 'nonescape-mini', 'nonescape-full', 'sentry-convnext-small', 'hybrid', 'hybrid-plus', 'ultra', 'or-ensemble', or 'or-ensemble-high-recall'."
    )


def iter_images(path: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTENSIONS else []
    if not root.exists():
        return []
    globber = root.rglob if recursive else root.glob
    return sorted(p for p in globber("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _build_fallback_image_processor(config):
    try:
        from transformers import ViTImageProcessor
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install transformers to use the HF backend.") from exc

    image_size = getattr(config, "image_size", 224)
    if isinstance(image_size, int):
        size = {"height": image_size, "width": image_size}
    else:
        size = image_size
    image_mean = getattr(config, "image_mean", [0.5, 0.5, 0.5])
    image_std = getattr(config, "image_std", [0.5, 0.5, 0.5])
    return ViTImageProcessor(
        do_resize=True,
        size=size,
        resample=3,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=image_mean,
        image_std=image_std,
    )


def _resolve_local_hf_snapshot(model_id: str) -> str | None:
    candidates = []
    env_hf_home = os.environ.get("HF_HOME")
    if env_hf_home:
        candidates.append(Path(env_hf_home))
    candidates.append(Path.home() / ".cache" / "huggingface")

    for hf_home in candidates:
        hub_dir = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
        if not hub_dir.exists():
            continue

        ref_path = hub_dir / "refs" / "main"
        if not ref_path.is_file():
            continue

        revision = ref_path.read_text(encoding="utf-8").strip()
        if not revision:
            continue

        snapshot_dir = hub_dir / "snapshots" / revision
        if snapshot_dir.is_dir():
            return str(snapshot_dir)

    return None