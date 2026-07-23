"""Optional MaaFramework integration with no eager runtime import."""

from .adapter import (
    MAA_CUSTOM_RECOGNITION_NAME,
    MaaAdapterError,
    MaaRecognitionPayload,
    MapRecognitionAdapter,
    image_from_maa,
)

__all__ = [
    "MAA_CUSTOM_RECOGNITION_NAME",
    "MaaAdapterError",
    "MaaRecognitionPayload",
    "MapRecognitionAdapter",
    "image_from_maa",
]
