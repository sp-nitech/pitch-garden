"""The module for pitch tracking algorithms."""

from .base import BaseAlgorithm
from .crepe import CrepeAlgorithm
from .fcn_f0 import FcnF0Algorithm
from .fcpe import FcpeAlgorithm
from .harmof0 import HarmoF0Algorithm
from .pesto import PestoAlgorithm
from .pienet import PiENetAlgorithm
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
from .yaapt import YaaptAlgorithm
from .yin import YinAlgorithm

ALGORITHM_MAP: dict[str, type[BaseAlgorithm]] = {
    "crepe": CrepeAlgorithm,
    "dio": DioAlgorithm,
    "fcn-f0": FcnF0Algorithm,
    "fcpe": FcpeAlgorithm,
    "harmof0": HarmoF0Algorithm,
    "harvest": HarvestAlgorithm,
    "pesto": PestoAlgorithm,
    "pienet": PiENetAlgorithm,
    "pyin": PyinAlgorithm,
    "rapt": RaptAlgorithm,
    "reaper": ReaperAlgorithm,
    "rmvpe": RmvpeAlgorithm,
    "straight": StraightAlgorithm,
    "swift-f0": SwiftF0Algorithm,
    "swipe": SwipeAlgorithm,
    "yaapt": YaaptAlgorithm,
    "yin": YinAlgorithm,
}

__all__ = [
    "ALGORITHM_MAP",
]
