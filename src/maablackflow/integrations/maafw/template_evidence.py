"""Placeholder adapter for future MaaFramework TemplateMatch evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from maablackflow.vision.grid import CandidateEvidence


@dataclass(frozen=True, slots=True)
class MaaTemplateMatchHit:
    """Framework-neutral representation of an already-produced template hit."""

    template: str
    box: tuple[int, int, int, int]
    score: float


class MaaTemplateEvidenceProvider:
    """Convert supplied TemplateMatch hits; it does not perform matching."""

    name = "maafw_template_match"
    requires_grid = False

    def from_hits(
        self,
        hits: Iterable[MaaTemplateMatchHit | Mapping[str, object]],
    ) -> tuple[CandidateEvidence, ...]:
        evidence: list[CandidateEvidence] = []
        for raw in hits:
            hit = raw if isinstance(raw, MaaTemplateMatchHit) else _parse_hit(raw)
            x, y, width, height = hit.box
            if width <= 0 or height <= 0:
                raise ValueError("TemplateMatch box must have positive dimensions")
            if not 0.0 <= hit.score <= 1.0:
                raise ValueError("TemplateMatch score must be between zero and one")
            evidence.append(
                CandidateEvidence(
                    x=x + width // 2,
                    y=y + height // 2,
                    source=f"maafw_template:{hit.template}",
                    confidence=hit.score,
                    radius=max(width, height) // 2,
                    scores={"template": hit.score},
                    bbox=hit.box,
                )
            )
        return tuple(sorted(evidence, key=lambda item: (item.y, item.x, item.source)))

    def collect(self, context: object) -> tuple[CandidateEvidence, ...]:
        """No-op until an external runtime explicitly supplies real hits."""
        return ()


def _parse_hit(value: Mapping[str, object]) -> MaaTemplateMatchHit:
    template, box, score = value.get("template"), value.get("box"), value.get("score")
    if not isinstance(template, str) or not template:
        raise ValueError("TemplateMatch hit requires a template name")
    if (
        not isinstance(box, (list, tuple))
        or len(box) != 4
        or any(not isinstance(item, int) for item in box)
    ):
        raise ValueError("TemplateMatch hit box must contain four integers")
    if not isinstance(score, (int, float)):
        raise ValueError("TemplateMatch hit requires a numeric score")
    return MaaTemplateMatchHit(template, tuple(box), float(score))  # type: ignore[arg-type]
