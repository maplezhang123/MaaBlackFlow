"""Serializable value objects produced by the offline vision baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass


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
    center_x: int
    center_y: int
    bbox: BoundingBox
    confidence: float

    def validate(self, image_width: int, image_height: int) -> None:
        self.bbox.validate(image_width, image_height)
        if not 0 <= self.center_x < image_width or not 0 <= self.center_y < image_height:
            raise ValueError("node center lies outside the image")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    image_width: int
    image_height: int
    nodes: tuple[DetectedNode, ...]
    method: str = "opencv_hough_baseline_v1"

    def validate(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        for node in self.nodes:
            node.validate(self.image_width, self.image_height)

    def to_dict(self, input_name: str | None = None) -> dict[str, object]:
        self.validate()
        document: dict[str, object] = {
            "image": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "method": self.method,
            "node_count": len(self.nodes),
            "nodes": [node.to_dict() for node in self.nodes],
        }
        if input_name is not None:
            document["input_name"] = input_name
        return document