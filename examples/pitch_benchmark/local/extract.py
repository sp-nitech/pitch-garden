"""Extract F0 from the eval set of pitch-benchmark with a tracker under a setting.

The settings are as follows, where the first pass uses the fixed search range.

- single: the single pass with the fixed search range.
- two-pass-1st: the two-pass tracking taking the voicing from the first pass.
- two-pass-2nd: the two-pass tracking taking the voicing from the second pass.
"""

import argparse
import dataclasses
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from typing import cast

import pitch_garden as pg
from pitch_garden.algorithms import ALGORITHM_MAP
from pitch_garden.algorithms.base import VoicingSource

# The frame period dividing that of the labels (16 ms) into four.
FRAME_PERIOD = 4.0

# The fixed search range in Hz, which is also used in the first pass.
F0_RANGE = (40.0, 800.0)

SETTINGS = ("single", "two-pass-1st", "two-pass-2nd")

# The panel of the clean audio in each split, where the audio of the valid split is
# only scaled in level.
PANELS = {"test": "identity", "valid": "level"}

# The corpora of the voice, where those of musical instruments are excluded.
CORPORA = (
    "APLAWD",
    "AVID",
    "FDA",
    "KEELE",
    "OSFGlottis",
    "SVD",
    "SpeechSynth",
    "Vocadito",
)


def create_two_pass(setting: str) -> pg.TwoPass | None:
    """Create the settings of the two-pass tracking.

    Parameters
    ----------
    setting : str
        The name of the setting.

    Returns
    -------
    out : TwoPass | None
        The settings of the two-pass tracking, or None for the single pass.

    Raises
    ------
    ValueError
        If the setting is not supported.

    """
    if setting not in SETTINGS:
        raise ValueError(f"Unsupported setting '{setting}'. Choose from {SETTINGS}.")
    if setting == "single":
        return None
    return pg.TwoPass(fmin=F0_RANGE[0], fmax=F0_RANGE[1])


def create_tracker(
    algorithm: str,
    setting: str,
    device: str | None = None,
    num_threads: int | None = None,
) -> pg.Tracker:
    """Create a tracker with the fixed search range.

    The tracker is rerun in the two-pass tracking regardless of the recommended
    settings of the algorithm.

    Parameters
    ----------
    algorithm : str
        The name of the algorithm with the default parameters.

    setting : str
        The name of the setting.

    device : str | None, optional
        The device given to the algorithms having the parameter, e.g., cuda.

    num_threads : int | None, optional
        The number of threads given to the algorithms having the parameter.

    Returns
    -------
    out : Tracker
        The tracker, whose range is clipped to the limits of the algorithm.

    """
    policy = None
    if setting != "single":
        voicing = cast(VoicingSource, setting.removeprefix("two-pass-"))
        policy = pg.TwoPassPolicy(enabled=True, voicing_from=voicing)
    fields = {f.name for f in dataclasses.fields(ALGORITHM_MAP[algorithm].config_class)}
    options = {"device": device, "num_threads": num_threads}
    params = {k: v for k, v in options.items() if k in fields and v is not None}
    return pg.Tracker(
        algorithm,
        fmin=F0_RANGE[0],
        fmax=F0_RANGE[1],
        two_pass_policy=policy,
        **params,
    )


# The tracker of a worker process, which is created once by `_init_worker`.
_worker_tracker: pg.Tracker | None = None


def _init_worker(algorithm: str, setting: str, device: str | None) -> None:
    """Create the tracker of a worker process with a single thread."""
    global _worker_tracker
    _worker_tracker = create_tracker(algorithm, setting, device, num_threads=1)


def _extract_in_worker(job: tuple[str, str, str]) -> None:
    """Extract F0 from a file with the tracker of the worker process."""
    extract_file(cast(pg.Tracker, _worker_tracker), *job)


def extract_file(
    tracker: pg.Tracker, setting: str, audio_path: str, save_path: str
) -> None:
    """Extract F0 from a file and save it.

    Parameters
    ----------
    tracker : Tracker
        The tracker, which keeps the loaded algorithm over the files.

    setting : str
        The name of the setting.

    audio_path : str
        The path to the audio file.

    save_path : str
        The path to the output file.

    """
    audio = pg.load_audio(audio_path)
    two_pass = create_two_pass(setting)
    if two_pass is None:
        pitch = tracker(audio, frame_period=FRAME_PERIOD)
    else:
        with pg.Pipeline([tracker], two_pass=two_pass, num_workers=1) as pipeline:
            result = pipeline(audio, frame_period=FRAME_PERIOD)
        # The output of a single tracker is always available.
        pitch = cast(pg.Pitch, result.pitch)
    pitch.tofile(save_path, output_format="linear", file_type="npz")


def extract(
    database_dir: str,
    output_dir: str,
    algorithm: str,
    setting: str,
    split: str,
    num_workers: int = 1,
    device: str | None = None,
) -> None:
    """Extract F0 from all clips of the voice and save them for the scoring script.

    Parameters
    ----------
    database_dir : str
        The directory of the eval set.

    output_dir : str
        The directory where '<setting>/trackers/<algorithm>/<corpus>/*.npz' are
        saved.

    algorithm : str
        The name of the algorithm with the default parameters.

    setting : str
        The setting of the two-pass tracking.

    split : str
        The split, i.e., 'test' or 'valid'.

    num_workers : int, optional
        The number of worker processes for the trackers running on CPU.

    device : str | None, optional
        The device given to the algorithms having the parameter, e.g., cuda.

    """
    create_two_pass(setting)  # Validate the setting before the extraction.
    jobs = []
    for corpus in CORPORA:
        audio_dir = os.path.join(database_dir, corpus, split, "audio", PANELS[split])
        save_dir = os.path.join(output_dir, setting, "trackers", algorithm, corpus)
        os.makedirs(save_dir, exist_ok=True)
        for path in sorted(glob.glob(os.path.join(audio_dir, "*.flac"))):
            name = os.path.splitext(os.path.basename(path))[0]
            save_path = os.path.join(save_dir, f"{name}.npz")
            jobs.append((setting, path, save_path))

    # A tracker running on a GPU is used in the main process for all files, and the
    # others are run in worker processes, each of which keeps its own tracker.
    tracker = create_tracker(algorithm, setting, device)
    if num_workers == 1 or not tracker.parallel_safe:
        for job in jobs:
            extract_file(tracker, *job)
        return
    initargs = (algorithm, setting, device)
    with ProcessPoolExecutor(
        max_workers=num_workers, initializer=_init_worker, initargs=initargs
    ) as executor:
        list(executor.map(_extract_in_worker, jobs))


def main() -> None:
    """Extract F0 with the given tracker under the given setting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database_dir", help="The directory of the eval set.")
    parser.add_argument("output_dir", help="The directory of the outputs.")
    parser.add_argument("--algorithm", required=True, help="The name of the tracker.")
    parser.add_argument("--setting", required=True, choices=SETTINGS)
    parser.add_argument("--split", required=True, choices=sorted(PANELS))
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", help="The device of neural trackers, e.g., cuda.")
    args = parser.parse_args()
    extract(
        args.database_dir,
        args.output_dir,
        args.algorithm,
        args.setting,
        args.split,
        args.num_workers,
        args.device,
    )


if __name__ == "__main__":
    main()
