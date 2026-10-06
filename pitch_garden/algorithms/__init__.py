"""The module for pitch tracking algorithms."""

from .base import BaseAlgorithm
from .sptk import (
    DioAlgorithm,
    HarvestAlgorithm,
    RaptAlgorithm,
    ReaperAlgorithm,
    SwipeAlgorithm,
)
from .straight import StraightAlgorithm
from .swift_f0 import SwiftF0Algorithm

ALGORITHM_MAP: dict[str, type[BaseAlgorithm]] = {
    "dio": DioAlgorithm,
    "harvest": HarvestAlgorithm,
    "rapt": RaptAlgorithm,
    "reaper": ReaperAlgorithm,
    "straight": StraightAlgorithm,
    "swift-f0": SwiftF0Algorithm,
    "swipe": SwipeAlgorithm,
}

__all__ = [
    "ALGORITHM_MAP",
]
