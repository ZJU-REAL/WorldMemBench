"""Masked DINOv2 features and Human-aligned Object Appearance scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from worldmembench.errors import EvaluationError
from worldmembench.metrics.human.target_association import (
    cosine_matrix,
    normalized_embedding,
)

from .schemas import ObjectAppearanceResult, ObjectTrackAnalysis


class ObjectAppearanceFeatureExtractor(Protocol):
    def embed(self, frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None: ...


class DinoV2LargeObjectFeatureExtractor:
    """The same masked-crop DINOv2-Large CLS embedding as Human Appearance."""

    def __init__(
        self,
        *,
        checkpoint: str,
        device: str,
        input_size: int = 518,
    ) -> None:
        try:
            import torch
            import torchvision.transforms as transforms
            from transformers import AutoModel
        except ImportError as error:
            raise EvaluationError(
                "Object Appearance requires torch, torchvision, and transformers; "
                "install worldmembench[object]"
            ) from error
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise EvaluationError(f"DINOv2 checkpoint not found: {checkpoint_path}")
        self.model = (
            AutoModel.from_pretrained(
                checkpoint_path,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((input_size, input_size), antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.device = device
        self.torch = torch

    def embed(self, frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        if mask.shape != frame_rgb.shape[:2] or not mask.any():
            return None
        y_coordinates, x_coordinates = np.where(mask)
        x1, x2 = int(x_coordinates.min()), int(x_coordinates.max()) + 1
        y1, y2 = int(y_coordinates.min()), int(y_coordinates.max()) + 1
        crop = frame_rgb[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2]
        crop[~crop_mask] = 0
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tensor).last_hidden_state[:, 0][0]
        return normalized_embedding(output.detach().float().cpu().numpy())


def analyze_object_appearance(
    track: ObjectTrackAnalysis,
    feature_extractor: ObjectAppearanceFeatureExtractor,
) -> ObjectAppearanceResult:
    phase = track.phase
    frame_scores: dict[int, float] = {
        frame_index: 0.0 for frame_index in phase.return_indices
    }
    if not phase.is_scorable:
        empty_slots = tuple(0.0 for _ in phase.revisit_indices)
        empty_counts = tuple(0 for _ in phase.revisit_indices)
        return ObjectAppearanceResult(
            phase=phase,
            frame_scores=frame_scores,
            similarity=0.0,
            appearance_coverage=0.0,
            n_reference_features=0,
            n_return_features=0,
            revisit_scores=empty_slots,
            revisit_coverages=empty_slots,
            n_revisit_features=empty_counts,
        )
    references: list[np.ndarray] = []
    for frame_index in phase.visible_indices:
        observation = track.observations[frame_index]
        if (
            not observation.present
            or observation.mask is None
            or not observation.mask_valid
        ):
            continue
        embedding = feature_extractor.embed(
            track.decoded_frames[frame_index],
            observation.mask,
        )
        if embedding is not None:
            references.append(normalized_embedding(embedding))

    returned_by_frame: dict[int, np.ndarray] = {}
    for frame_index in phase.return_indices:
        observation = track.observations[frame_index]
        if (
            not observation.present
            or observation.mask is None
            or not observation.mask_valid
        ):
            continue
        candidate = feature_extractor.embed(
            track.decoded_frames[frame_index],
            observation.mask,
        )
        if candidate is None:
            continue
        normalized = normalized_embedding(candidate)
        returned_by_frame[frame_index] = normalized
        if not references:
            continue
        similarities = cosine_matrix(
            references,
            [normalized],
        )[:, 0]
        frame_scores[frame_index] = float(np.clip(similarities.max(), 0.0, 1.0))

    return_frame_count = len(phase.return_indices)
    n_return_features = len(returned_by_frame)
    similarity = (
        sum(frame_scores[index] for index in returned_by_frame) / n_return_features
        if n_return_features and references and phase.is_scorable
        else 0.0
    )
    coverage = n_return_features / return_frame_count if return_frame_count else 0.0
    revisit_scores: list[float] = []
    revisit_coverages: list[float] = []
    n_revisit_features: list[int] = []
    for revisit_indices in phase.revisit_indices:
        feature_indices = [
            index for index in revisit_indices if index in returned_by_frame
        ]
        n_features = len(feature_indices)
        n_revisit_features.append(n_features)
        revisit_coverages.append(
            n_features / len(revisit_indices) if revisit_indices else 0.0
        )
        revisit_scores.append(
            (
                sum(frame_scores[index] for index in feature_indices) / n_features
                if n_features and references and phase.is_scorable
                else 0.0
            )
        )
    return ObjectAppearanceResult(
        phase=phase,
        frame_scores=frame_scores,
        similarity=similarity,
        appearance_coverage=coverage,
        n_reference_features=len(references),
        n_return_features=n_return_features,
        revisit_scores=tuple(revisit_scores),
        revisit_coverages=tuple(revisit_coverages),
        n_revisit_features=tuple(n_revisit_features),
    )
