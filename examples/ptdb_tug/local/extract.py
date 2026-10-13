"""Extract F0 from PTDB-TUG with a tracker under its default two-pass policy.

The first pass uses the fixed search range, and the tracker is rerun with the
estimated range if its default policy, which is chosen on the speech of
pitch-benchmark, enables it.

"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from typing import cast

import pitch_garden as pg

import utils.trackers

# The frame period dividing those of the reference, i.e., 16 ms + 10 ms * i, so that
# the reference frames are scored without interpolation.
FRAME_PERIOD = 2.0

# The fixed search range in Hz, which is also used in the first pass.
F0_RANGE = (40.0, 800.0)


def create_tracker(
    algorithm: str, device: str | None = None, num_threads: int | None = None
) -> pg.Tracker:
    """Create a tracker with the fixed search range and its default two-pass policy.

    Parameters
    ----------
    algorithm : str
        The name of the algorithm with the default parameters.

    device : str | None, optional
        The device given to the algorithms supporting GPUs, e.g., cuda.

    num_threads : int | None, optional
        The number of threads given to the algorithms having the parameter.

    Returns
    -------
    out : Tracker
        The tracker, whose range is clipped to the limits of the algorithm.

    """
    return utils.trackers.create_tracker(
        algorithm,
        device=device,
        num_threads=num_threads,
        fmin=F0_RANGE[0],
        fmax=F0_RANGE[1],
    )


def list_jobs(database_dir: str, save_dir: str) -> list[tuple[str, str]]:
    """List the microphone signals and the paths to save their F0.

    Parameters
    ----------
    database_dir : str
        The directory of 'SPEECH DATA' of PTDB-TUG.

    save_dir : str
        The directory where '<speaker>_<sentence>.npz' are saved.

    Returns
    -------
    out : list[tuple[str, str]]
        The pairs of the audio path and the save path sorted by the audio path.

    """
    pattern = os.path.join(database_dir, "*", "MIC", "*", "mic_*.wav")
    jobs = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.splitext(os.path.basename(path))[0].removeprefix("mic_")
        jobs.append((path, os.path.join(save_dir, f"{name}.npz")))
    return jobs


def extract_file(tracker: pg.Tracker, audio_path: str, save_path: str) -> None:
    """Extract F0 from a file and save it.

    Parameters
    ----------
    tracker : Tracker
        The tracker, which keeps the loaded algorithm over the files.

    audio_path : str
        The path to the audio file.

    save_path : str
        The path to the output file.

    """
    two_pass = pg.TwoPass(fmin=F0_RANGE[0], fmax=F0_RANGE[1])
    with pg.Pipeline([tracker], two_pass=two_pass, num_workers=1) as pipeline:
        result = pipeline(pg.load_audio(audio_path), frame_period=FRAME_PERIOD)
    # The output of a single tracker is always available.
    pitch = cast(pg.Pitch, result.pitch)
    pitch.tofile(save_path, output_format="linear", file_type="npz")


# The tracker of a worker process, which is created once by `_init_worker`.
_worker_tracker: pg.Tracker | None = None


def _init_worker(algorithm: str, device: str | None) -> None:
    """Create the tracker of a worker process with a single thread."""
    global _worker_tracker
    _worker_tracker = create_tracker(algorithm, device, num_threads=1)


def _extract_in_worker(job: tuple[str, str]) -> None:
    """Extract F0 from a file with the tracker of the worker process."""
    extract_file(cast(pg.Tracker, _worker_tracker), *job)


def extract(
    database_dir: str,
    output_dir: str,
    algorithm: str,
    num_workers: int = 1,
    device: str | None = None,
) -> None:
    """Extract F0 from all files and save them in the layout of the scoring script.

    Parameters
    ----------
    database_dir : str
        The directory of 'SPEECH DATA' of PTDB-TUG.

    output_dir : str
        The directory where 'trackers/<algorithm>/*.npz' are saved.

    algorithm : str
        The name of the algorithm with the default parameters.

    num_workers : int, optional
        The number of worker processes for the trackers running on CPU.

    device : str | None, optional
        The device given to the algorithms supporting GPUs, e.g., cuda.

    """
    save_dir = os.path.join(output_dir, "trackers", algorithm)
    os.makedirs(save_dir, exist_ok=True)
    jobs = list_jobs(database_dir, save_dir)

    # A tracker running on a GPU is used in the main process for all files, and the
    # others are run in worker processes, each of which keeps its own tracker.
    tracker = create_tracker(algorithm, device)
    if num_workers == 1 or not tracker.parallel_safe:
        for job in jobs:
            extract_file(tracker, *job)
        return
    with ProcessPoolExecutor(
        max_workers=num_workers, initializer=_init_worker, initargs=(algorithm, device)
    ) as executor:
        list(executor.map(_extract_in_worker, jobs, chunksize=16))


def main() -> None:
    """Extract F0 with the given tracker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database_dir", help="The directory of 'SPEECH DATA'.")
    parser.add_argument("output_dir", help="The directory of the outputs.")
    parser.add_argument("--algorithm", required=True, help="The name of the tracker.")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", help="The device of neural trackers, e.g., cuda.")
    args = parser.parse_args()
    extract(
        args.database_dir,
        args.output_dir,
        args.algorithm,
        args.num_workers,
        args.device,
    )


if __name__ == "__main__":
    main()
