"""Serializable value objects produced by offline map-grid detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


NODE_CATEGORIES = {
    "event_node",
    "empty_waypoint",
    "current_position",
    "occluded_node",
    "uncertain",
}


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int

    def validate(self, image_width: int, image_height: int) -> None:
        if not 0 <= self.x < image_width or not 0 <= self.y < image_height:
            raise ValueError("point lies outside the image")


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def validate(self, image_width: int, image_height: int) -> None:
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("bounding box must have a positive size inside the image")
        if self.x + self.width > image_width or self.y + self.height > image_height:
            raise ValueError("bounding box extends outside the image")


@dataclass(frozen=True, slots=True)
class DetectedNode:
    id: str
    center: Point
    grid_center: Point
    bbox: BoundingBox
    confidence: float
    category: str = "uncertain"
    reliable: bool = False
    sources: tuple[str, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)
    evidence: tuple[dict[str, object], ...] = ()
    grid_row: int | None = None
    grid_col: int | None = None
    marker_center: Point | None = None
    marker_bbox: BoundingBox | None = None

    @property
    def center_x(self) -> int:
        return self.center.x

    @property
    def center_y(self) -> int:
        return self.center.y

    def validate(self, image_width: int, image_height: int) -> None:
        self.center.validate(image_width, image_height)
        self.grid_center.validate(image_width, image_height)
        if self.center != self.grid_center:
            raise ValueError("final node center must equal grid_center")
        self.bbox.validate(image_width, image_height)
        if self.marker_center is not None:
            self.marker_center.validate(image_width, image_height)
        if self.marker_bbox is not None:
            self.marker_bbox.validate(image_width, image_height)
        if (self.marker_center is None) != (self.marker_bbox is None):
            raise ValueError("marker_center and marker_bbox must be provided together")
        if self.category != "current_position" and self.marker_center is not None:
            raise ValueError("only current_position may have marker coordinates")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if self.category not in NODE_CATEGORIES:
            raise ValueError(f"unknown node category: {self.category}")
        if (self.grid_row is None) != (self.grid_col is None):
            raise ValueError("grid row and column must be provided together")
        if self.grid_row is not None and (self.grid_row < 0 or self.grid_col < 0):
            raise ValueError("grid row and column must be non-negative")
        for name, score in self.scores.items():
            if not name or not 0.0 <= score <= 1.0:
                raise ValueError("evidence scores must use names and values from zero to one")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    image_width: int
    image_height: int
    nodes: tuple[DetectedNode, ...]
    method: str = "opencv_global_periodic_grid_fusion_v3"
    analysis: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        current_positions = 0
        cells: set[tuple[int, int]] = set()
        for node in self.nodes:
            node.validate(self.image_width, self.image_height)
            current_positions += node.category == "current_position"
            if node.grid_row is not None:
                cell = (node.grid_row, node.grid_col)
                if cell in cells:
                    raise ValueError("a grid cell may contain at most one final node")
                cells.add(cell)
        if current_positions > 1:
            raise ValueError("a result may contain at most one current_position")

    def to_dict(self, input_name: str | None = None) -> dict[str, object]:
        self.validate()
        document: dict[str, object] = {
            "image": {"width": self.image_width, "height": self.image_height},
            "method": self.method,
            "node_count": len(self.nodes),
            "analysis": self.analysis,
            "nodes": [node.to_dict() for node in self.nodes],
        }
        if input_name is not None:
            document["input_name"] = input_name
        return document