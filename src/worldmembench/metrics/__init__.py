"""WorldMemBench metric interfaces and built-in registry."""

from worldmembench.metrics.base import Metric
from worldmembench.metrics.registry import MetricRegistry
from worldmembench.metrics.dynamic import DynamicMemoryMetric
from worldmembench.metrics.camera import CameraRotationMetric, CameraTranslationMetric
from worldmembench.metrics.general_video_quality import AestheticQualityMetric, ImagingQualityMetric
from worldmembench.metrics.human import HumanAppearanceMetric, HumanIdentityMetric
from worldmembench.metrics.object import ObjectAppearanceMetric, ObjectGeometryMetric
from worldmembench.metrics.environment import (
    EnvironmentVisibleMemoryMetric, EnvironmentDepthFidelityMetric, EnvironmentSceneReconstructionMetric,
)

registry = MetricRegistry()
registry.register(DynamicMemoryMetric())
registry.register(CameraRotationMetric())
registry.register(CameraTranslationMetric())
registry.register(ImagingQualityMetric())
registry.register(AestheticQualityMetric())
registry.register(HumanIdentityMetric())
registry.register(HumanAppearanceMetric())
registry.register(ObjectAppearanceMetric())
registry.register(ObjectGeometryMetric())
registry.register(EnvironmentVisibleMemoryMetric())
registry.register(EnvironmentDepthFidelityMetric())
registry.register(EnvironmentSceneReconstructionMetric())

__all__ = ["Metric", "MetricRegistry", "registry"]
