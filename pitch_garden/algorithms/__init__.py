"""The module for pitch tracking algorithms."""

from .base import BaseAlgorithm
from .crepe import CrepeAlgorithm
from .pesto import PestoAlgorithm
from .rmvpe import RmvpeAlgorithm
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
    "crepe": CrepeAlgorithm,
    "dio": DioAlgorithm,
    "harvest": HarvestAlgorithm,
    "pesto": PestoAlgorithm,
    "rapt": RaptAlgorithm,
    "reaper": ReaperAlgorithm,
    "rmvpe": RmvpeAlgorithm,
    "straight": StraightAlgorithm,
    "swift-f0": SwiftF0Algorithm,
    "swipe": SwipeAlgorithm,
}

__all__ = [
    "ALGORITHM_MAP",
]
