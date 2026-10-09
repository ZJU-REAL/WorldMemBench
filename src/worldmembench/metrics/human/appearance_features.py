"""Phase-local body tracking and masked DINOv2 appearance scoring."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from worldmembench.errors import EvaluationError

from .person_mask_tracker import MaskPropagator
from .schemas import (
    AppearanceAnalysis,
    AppearanceTargetResult,
    BoundingBox,
    IdentityAnalysis,
)
from .target_association import (
    cosine_matrix,
    normalized_embedding,
)


class AppearanceBackend(Protocol):
    def extract(
        self,
        identity: IdentityAnalysis,
    ) -> dict[int, dict[int, np.ndarray]]: ...


class AppearanceFeatureExtractor(Protocol):
    def embed_batch(
        self,
        samples: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> list[np.ndarray | None]: ...


class DinoV2LargeFeatureExtractor:
    """Batched masked-crop DINOv2-Large CLS embeddings."""

    def __init__(
        self,
        *,
        checkpoint: str,
        device: str,
        input_size: int = 518,
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("DINO batch_size must be positive")
        try:
            import torch
            import torchvision.transforms as transforms
            from transformers import AutoModel
        except ImportError as error:
            raise EvaluationError(
                "Human Appearance requires torch, torchvision, and transformers; "
                "install worldmembench[human]"
            ) from error
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise EvaluationError(f"DINOv2 checkpoint not found: {checkpoint_path}")
        self.model = AutoModel.from_pretrained(
            checkpoint_path,
            local_files_only=True,
        ).to(device).eval()
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
        self.effective_batch_size = batch_size

    @staticmethod
    def _masked_crop(
        frame_rgb: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray | None:
        if mask.shape != frame_rgb.shape[:2] or not mask.any():
            return None
        y_coordinates, x_coordinates = np.where(mask)
        x1, x2 = int(x_coordinates.min()), int(x_coordinates.max()) + 1
        y1, y2 = int(y_coordinates.min()), int(y_coordinates.max()) + 1
        crop = frame_rgb[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2]
        crop[~crop_mask] = 0
        return crop

    @staticmethod
    def _is_out_of_memory(error: RuntimeError) -> bool:
        return "out of memory" in str(error).lower()

    def embed_batch(
        self,
        samples: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> list[np.ndarray | None]:
        results: list[np.ndarray | None] = [None] * len(samples)
        position = 0
        while position < len(samples):
            chunk_size = min(
                self.effective_batch_size,
                len(samples) - position,
            )
            chunk = samples[position : position + chunk_size]
            tensors = []
            result_indices = []
            for offset, (frame_rgb, mask) in enumerate(chunk):
                crop = self._masked_crop(frame_rgb, mask)
                if crop is None:
                    continue
                tensors.append(self.transform(crop))
                result_indices.append(position + offset)
            if not tensors:
                position += chunk_size
                continue
            try:
                tensor_batch = self.torch.stack(tensors).to(self.device)
                with self.torch.inference_mode():
                    vectors = self.model(tensor_batch).last_hidden_state[:, 0]
                vectors = vectors.detach().float().cpu().numpy()
            except RuntimeError as error:
                if (
                    not self._is_out_of_memory(error)
                    or self.effective_batch_size == 1
                ):
                    raise
                self.effective_batch_size = max(
                    1,
                    self.effective_batch_size // 2,
                )
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
                continue
            for result_index, vector in zip(result_indices, vectors):
                results[result_index] = normalized_embedding(vector)
            position += chunk_size
        return results


class YoloSamDinoAppearanceBackend:
    def __init__(
        self,
        *,
        mask_propagator: MaskPropagator,
        feature_extractor: AppearanceFeatureExtractor,
    ) -> None:
        self.mask_propagator = mask_propagator
        self.feature_extractor = feature_extractor

    def _target_phase_features(
        self,
        identity: IdentityAnalysis,
        target_id: int,
        phase_indices: Sequence[int],
        person_assignments: dict[int, dict[int, BoundingBox]],
    ) -> dict[int, dict[int, np.ndarray]]:
        if not phase_indices:
            return {}
        frames = [identity.decoded_frames[index] for index in phase_indices]
        target_prompts: list[tuple[int, BoundingBox]] = []
        for local_index, frame_index in enumerate(phase_indices):
            person_box = person_assignments.get(frame_index, {}).get(target_id)
            if person_box is not None:
                target_prompts.append((local_index, person_box))
        prompts = {target_id: target_prompts} if target_prompts else {}
        # Appearance keeps only frames with a direct body-box prompt below.
        # Propagated masks cannot replace those direct masks and are discarded.
        local_masks = self.mask_propagator.track(
            frames,
            prompts,
            propagate=False,
        )
        features: dict[int, dict[int, np.ndarray]] = {}
        sample_keys: list[tuple[int, int]] = []
        samples: list[tuple[np.ndarray, np.ndarray]] = []
        for tracked_target_id, target_masks in sorted(local_masks.items()):
            for local_index, mask in sorted(target_masks.items()):
                if local_index < 0 or local_index >= len(phase_indices):
                    raise EvaluationError("SAM2 returned an out-of-range phase frame")
                frame_index = phase_indices[local_index]
                # A propagated mask is valid only while the body association
                # confirms that the tracked mask still overlaps a person.
                if tracked_target_id not in person_assignments.get(frame_index, {}):
                    continue
                sample_keys.append((tracked_target_id, frame_index))
                samples.append(
                    (
                        identity.decoded_frames[frame_index],
                        np.asarray(mask, dtype=bool),
                    )
                )
        embeddings = self.feature_extractor.embed_batch(samples)
        if len(embeddings) != len(sample_keys):
            raise EvaluationError(
                "DINO batch output count does not match its input count"
            )
        for (tracked_target_id, frame_index), embedding in zip(
            sample_keys,
            embeddings,
        ):
            if embedding is not None:
                features.setdefault(tracked_target_id, {})[frame_index] = (
                    normalized_embedding(embedding)
                )
        return features

    def extract(self, identity: IdentityAnalysis) -> dict[int, dict[int, np.ndarray]]:
        person_assignments = identity.body_assignments
        merged: dict[int, dict[int, np.ndarray]] = {}
        for phase in identity.appearance_target_phases:
            if not phase.is_scorable:
                continue
            visible = self._target_phase_features(
                identity,
                phase.target_id,
                phase.visible_indices,
                person_assignments,
            )
            returned = self._target_phase_features(
                identity,
                phase.target_id,
                phase.return_indices,
                person_assignments,
            )
            for source in (visible, returned):
                for target_id, frame_features in source.items():
                    merged.setdefault(target_id, {}).update(frame_features)
        return merged


def analyze_appearance(
    identity: IdentityAnalysis,
    backend: AppearanceBackend,
) -> AppearanceAnalysis:
    features = backend.extract(identity)
    results: list[AppearanceTargetResult] = []
    for phase in identity.appearance_target_phases:
        target_id = phase.target_id
        target_features = features.get(target_id, {})
        references = [
            target_features[index]
            for index in phase.visible_indices
            if index in target_features
        ]
        frame_scores: dict[int, float] = {}
        n_return_features = 0
        for frame_index in phase.return_indices:
            candidate = target_features.get(frame_index)
            if candidate is None:
                frame_scores[frame_index] = 0.0
                continue
            n_return_features += 1
            if not references or not phase.is_scorable:
                frame_scores[frame_index] = 0.0
                continue
            similarities = cosine_matrix(references, [candidate])[:, 0]
            frame_scores[frame_index] = float(
                np.clip(similarities.max(), 0.0, 1.0)
            )
        return_frame_count = len(phase.return_indices)
        score = (
            sum(frame_scores.values()) / n_return_features
            if n_return_features
            else 0.0
        )
        coverage = (
            n_return_features / return_frame_count if return_frame_count else 0.0
        )
        if not phase.is_scorable:
            score = 0.0
        revisit_scores: list[float] = []
        revisit_coverages: list[float] = []
        n_revisit_features: list[int] = []
        for revisit_indices in phase.revisit_indices:
            feature_indices = [
                index for index in revisit_indices if index in target_features
            ]
            n_features = len(feature_indices)
            n_revisit_features.append(n_features)
            revisit_coverages.append(
                n_features / len(revisit_indices) if revisit_indices else 0.0
            )
            if n_features and phase.is_scorable:
                revisit_scores.append(
                    sum(frame_scores[index] for index in feature_indices) / n_features
                )
            else:
                revisit_scores.append(0.0)
        results.append(
            AppearanceTargetResult(
                target_id=target_id,
                phase=phase,
                frame_scores=frame_scores,
                score=score,
                reappearance_coverage=coverage,
                n_reference_features=len(references),
                n_return_features=n_return_features,
                revisit_scores=tuple(revisit_scores),
                revisit_coverages=tuple(revisit_coverages),
                n_revisit_features=tuple(n_revisit_features),
            )
        )
    return AppearanceAnalysis(target_results=tuple(results))
