"""Offline image inventory and map-node detection baseline."""

from .dataset import (
    DatasetInspectionError,
    InventoryEntry,
    InventoryResult,
    infer_scene_label,
    inspect_dataset,
    sha256_file,
)
from .detector import NodeDetector
from .image_io import VisionError
from .models import BoundingBox, DetectedNode, DetectionResult
from .output import annotate_image, write_detection_outputs

__all__ = [
    "BoundingBox",
    "DatasetInspectionError",
    "DetectedNode",
    "DetectionResult",
    "InventoryEntry",
    "InventoryResult",
    "NodeDetector",
    "VisionError",
    "annotate_image",
    "infer_scene_label",
    "inspect_dataset",
    "sha256_file",
    "write_detection_outputs",
]