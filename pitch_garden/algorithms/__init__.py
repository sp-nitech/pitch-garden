"""The module for pitch tracking algorithms."""

from .base import BaseAlgorithm
from .crepe import CrepeAlgorithm
from .fcn_f0 import FcnF0Algorithm
from .pesto import PestoAlgorithm
from .pyin import PyinAlgorithm
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
from .yin import YinAlgorithm

ALGORITHM_MAP: dict[str, type[BaseAlgorithm]] = {
    "crepe": CrepeAlgorithm,
    "dio": DioAlgorithm,
    "fcn-f0": FcnF0Algorithm,
    "harvest": HarvestAlgorithm,
    "pesto": PestoAlgorithm,
    "pyin": PyinAlgorithm,
    "rapt": RaptAlgorithm,
    "reaper": ReaperAlgorithm,
    "rmvpe": RmvpeAlgorithm,
    "straight": StraightAlgorithm,
    "swift-f0": SwiftF0Algorithm,
    "swipe": SwipeAlgorithm,
    "yin": YinAlgorithm,
}

__all__ = [
    "ALGORITHM_MAP",
]
