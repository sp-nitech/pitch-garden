"""A unified interface to various pitch trackers with ensemble voting."""

from .algorithms.base import TwoPassPolicy
from .interfaces.pipeline import Pipeline, PipelineResult, TwoPass
from .interfaces.tracker import Tracker
from .interfaces.types import Audio, Pitch
from .utils.io import RawFormat, load_audio
from .version import __version__

__all__ = [
    "Audio",
    "Pipeline",
    "PipelineResult",
    "Pitch",
    "RawFormat",
    "Tracker",
    "TwoPass",
    "TwoPassPolicy",
    "__version__",
    "load_audio",
]
