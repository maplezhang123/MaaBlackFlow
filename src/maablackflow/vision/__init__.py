"""Offline image inventory and two-stage map-grid detection."""

from .dataset import (
    DatasetInspectionError,
    InventoryEntry,
    InventoryResult,
    infer_scene_label,
    inspect_dataset,
    sha256_file,
)
from .detector import NodeDetector
from .evaluation import (
    EvaluationError,
    EvaluationReport,
    ImageEvaluation,
    evaluate_directories,
)
from .grid import (
    CandidateEvidence,
    GridFit,
    MapViewport,
    RoadGridAnalysis,
    analyze_road_grid,
)
from .image_io import VisionError
from .models import BoundingBox, DetectedNode, DetectionResult, Point
from .output import annotate_image, write_detection_outputs, write_grid_debug_output
from .providers import EvidenceContext, EvidenceProvider

__all__ = [
    "BoundingBox",
    "CandidateEvidence",
    "DatasetInspectionError",
    "DetectedNode",
    "DetectionResult",
    "EvidenceContext",
    "EvidenceProvider",
    "EvaluationError",
    "EvaluationReport",
    "GridFit",
    "ImageEvaluation",
    "InventoryEntry",
    "InventoryResult",
    "MapViewport",
    "NodeDetector",
    "Point",
    "RoadGridAnalysis",
    "VisionError",
    "analyze_road_grid",
    "annotate_image",
    "evaluate_directories",
    "infer_scene_label",
    "inspect_dataset",
    "sha256_file",
    "write_detection_outputs",
    "write_grid_debug_output",
]