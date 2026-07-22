"""Private local dataset inventory and contact-sheet generation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


SUPPORTED_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


class DatasetInspectionError(ValueError):
    """Raised when a private image directory cannot be inventoried."""


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    filename: str
    image_format: str
    width: int
    height: int
    channels: int
    file_size: int
    sha256: str
    duplicate_of: str | None
    scene_label: str
    suitable_for_node_detection: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InventoryResult:
    entries: tuple[InventoryEntry, ...]
    manifest_path: Path
    contact_sheet_path: Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetInspectionError(f"cannot hash image '{path}': {exc}") from exc
    return digest.hexdigest()


def infer_scene_label(filename: str) -> str:
    name = Path(filename).stem.casefold()
    if "内部" in name:
        return "节点内部界面"
    if "移动工具" in name or ("工具" in name and ("打开" in name or "拥有" in name)):
        return "移动工具展开状态"
    if any(keyword in name for keyword in ("移动过后", "移动后", "走了", "走过")):
        return "移动后状态"
    if "缩小" in name:
        return "地图缩小状态"
    if "放大" in name:
        return "地图放大状态"
    if "起点" in name or "终点" in name:
        return "起点或终点相关界面"
    if any(
        keyword in name
        for keyword in (
            "所在",
            "地图",
            "结点",
            "节点",
            "不期而遇",
            "应急助力",
            "先行一步",
            "秘境行商",
        )
    ):
        return "地图正常状态"
    return "暂时无法判断"


def is_detection_candidate(filename: str, scene_label: str) -> bool:
    name = Path(filename).stem.casefold()
    if scene_label == "节点内部界面" or "选项" in name:
        return False
    return scene_label != "暂时无法判断"


def inspect_dataset(
    source_dir: str | Path,
    output_dir: str | Path = Path("data/outputs_private/run01"),
) -> InventoryResult:
    source = Path(source_dir)
    output = Path(output_dir)
    if not source.exists():
        raise DatasetInspectionError(f"dataset directory does not exist: {source}")
    if not source.is_dir():
        raise DatasetInspectionError(f"dataset path is not a directory: {source}")

    image_paths = sorted(
        (
            item
            for item in source.rglob("*")
            if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda item: item.relative_to(source).as_posix().casefold(),
    )
    if not image_paths:
        raise DatasetInspectionError(f"dataset contains no supported images: {source}")

    entries: list[InventoryEntry] = []
    first_by_hash: dict[str, str] = {}
    for image_path in image_paths:
        relative_name = image_path.relative_to(source).as_posix()
        digest = sha256_file(image_path)
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                image_format = image.format or image_path.suffix.lstrip(".").upper()
                channels = len(image.getbands())
        except (OSError, UnidentifiedImageError) as exc:
            raise DatasetInspectionError(
                f"invalid or unsupported image '{relative_name}': {exc}"
            ) from exc
        label = infer_scene_label(relative_name)
        duplicate_of = first_by_hash.get(digest)
        first_by_hash.setdefault(digest, relative_name)
        entries.append(
            InventoryEntry(
                filename=relative_name,
                image_format=image_format,
                width=width,
                height=height,
                channels=channels,
                file_size=image_path.stat().st_size,
                sha256=digest,
                duplicate_of=duplicate_of,
                scene_label=label,
                suitable_for_node_detection=is_detection_candidate(relative_name, label),
            )
        )

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "dataset_manifest.json"
    contact_sheet_path = output / "contact_sheet.png"
    resolution_counts = Counter(f"{entry.width}x{entry.height}" for entry in entries)
    scene_counts = Counter(entry.scene_label for entry in entries)
    document = {
        "source_name": source.name,
        "image_count": len(entries),
        "duplicate_count": sum(entry.duplicate_of is not None for entry in entries),
        "resolutions": dict(sorted(resolution_counts.items())),
        "scene_labels": dict(sorted(scene_counts.items())),
        "images": [entry.to_dict() for entry in entries],
    }
    manifest_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_contact_sheet(source, entries, contact_sheet_path)
    return InventoryResult(tuple(entries), manifest_path, contact_sheet_path)


def _write_contact_sheet(
    source: Path, entries: list[InventoryEntry], destination: Path
) -> None:
    columns = min(4, max(1, len(entries)))
    thumb_width, thumb_height, caption_height = 320, 180, 28
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, rows * (thumb_height + caption_height)),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    for index, entry in enumerate(entries, 1):
        column = (index - 1) % columns
        row = (index - 1) // columns
        x = column * thumb_width
        y = row * (thumb_height + caption_height)
        with Image.open(source / Path(entry.filename)) as image:
            thumbnail = ImageOps.contain(image.convert("RGB"), (thumb_width, thumb_height))
        paste_x = x + (thumb_width - thumbnail.width) // 2
        paste_y = y + (thumb_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (paste_x, paste_y))
        draw.rectangle((x, y, x + thumb_width - 1, y + thumb_height - 1), outline=(90, 90, 90))
        duplicate_marker = " DUP" if entry.duplicate_of else ""
        draw.text(
            (x + 6, y + thumb_height + 6),
            f"{index:03d}  {entry.width}x{entry.height}{duplicate_marker}",
            fill=(235, 235, 235),
        )
    sheet.save(destination, format="PNG")