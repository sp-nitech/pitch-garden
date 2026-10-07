"""Extract F0 from Vocadito with a tracker in the fixed search range."""

import argparse
import glob
import os

import pitch_garden as pg

from utils.trackers import create_tracker

# The frame period of the reference F0, i.e., 256 samples at 44.1 kHz.
FRAME_PERIOD = 1000 * 256 / 44100

# The fixed search range in Hz for the singing voice.
F0_RANGE = (40.0, 800.0)


def extract(
    database_dir: str, output_dir: str, algorithm: str, device: str | None = None
) -> None:
    """Extract F0 from all files and save them in the layout of the scoring script.

    Parameters
    ----------
    database_dir : str
        The directory of Vocadito containing 'Audio' and 'Annotations'.

    output_dir : str
        The directory where 'trackers/<algorithm>/*.npz' are saved.

    algorithm : str
        The name of the algorithm with the default parameters.

    device : str | None, optional
        The device given to the algorithms supporting GPUs, e.g., cuda.

    """
    tracker = create_tracker(algorithm, device=device)
    save_dir = os.path.join(output_dir, "trackers", algorithm)
    os.makedirs(save_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(database_dir, "Audio", "*.wav"))):
        name = os.path.splitext(os.path.basename(path))[0]
        audio = pg.load_audio(path)
        pitch = tracker(audio, frame_period=FRAME_PERIOD, f0_range=F0_RANGE)
        pitch.tofile(
            os.path.join(save_dir, f"{name}.npz"),
            output_format="linear",
            file_type="npz",
        )


def main() -> None:
    """Extract F0 with the given tracker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database_dir", help="The directory of Vocadito.")
    parser.add_argument("output_dir", help="The directory of the outputs.")
    parser.add_argument("--algorithm", required=True, help="The name of the tracker.")
    parser.add_argument("--device", help="The device of neural trackers, e.g., cuda.")
    args = parser.parse_args()
    extract(args.database_dir, args.output_dir, args.algorithm, args.device)


if __name__ == "__main__":
    main()
