"""A module defining the base class of voters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from ..interfaces.types import Pitch


class BaseVoter(ABC):
    """An abstract base class for voters integrating the outputs of trackers.

    A voter receives the outputs of trackers keyed by the tracker names, so that
    tracker-specific processing such as alignment and weighting can be applied.
    Voters requiring training can override `fit`, and their learned parameters can
    be saved and restored via `state_dict` and `load_state_dict`.

    """

    def fit(
        self,
        estimates: Sequence[Mapping[str, Pitch]],
        references: Sequence[Pitch],
    ) -> BaseVoter:
        """Fit the voter to training data.

        Parameters
        ----------
        estimates : Sequence[Mapping[str, Pitch]]
            The outputs of the trackers for each utterance.

        references : Sequence[Pitch]
            The reference F0 contours for each utterance.

        Returns
        -------
        out : BaseVoter
            The fitted voter itself.

        """
        return self

    def check_trackers(self, names: Sequence[str]) -> None:
        """Check that the voter can be used with the trackers.

        Parameters
        ----------
        names : Sequence[str]
            The names of the trackers.

        Raises
        ------
        ValueError
            If the voter cannot be used with the trackers.

        """

    def state_dict(self) -> dict[str, Any]:
        """Get the learned parameters.

        Returns
        -------
        out : dict[str, Any]
            The learned parameters, which must be JSON serializable.

        """
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the learned parameters.

        Parameters
        ----------
        state : Mapping[str, Any]
            The learned parameters.

        """

    def __call__(self, pitches: Mapping[str, Pitch]) -> Pitch:
        """Integrate the outputs of the trackers.

        Parameters
        ----------
        pitches : Mapping[str, Pitch]
            The outputs of the trackers keyed by the tracker names. All contours must
            be on the same frame grid.

        Returns
        -------
        out : Pitch
            The integrated F0 contour.

        Raises
        ------
        ValueError
            If the inputs are inconsistent.

        """
        if len(pitches) == 0:
            raise ValueError("At least one pitch contour is required.")
        first = next(iter(pitches.values()))
        for name, pitch in pitches.items():
            if pitch.length != first.length:
                raise ValueError(
                    f"The number of frames of '{name}' ({pitch.length}) differs "
                    f"from the others ({first.length})."
                )
            if pitch.frame_period != first.frame_period:
                raise ValueError(f"The frame period of '{name}' differs.")
        return self.vote(pitches)

    @abstractmethod
    def vote(self, pitches: Mapping[str, Pitch]) -> Pitch:
        """Integrate the validated outputs of the trackers.

        Parameters
        ----------
        pitches : Mapping[str, Pitch]
            The outputs of the trackers keyed by the tracker names.

        Returns
        -------
        out : Pitch
            The integrated F0 contour.

        """
