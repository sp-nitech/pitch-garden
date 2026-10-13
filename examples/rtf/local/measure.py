"""Measure the real-time factor (RTF) of a tracker on a synthetic signal."""

import argparse
import json
import os
import platform
import time

import numpy as np

from utils.trackers import create_tracker

SAMPLE_RATE = 16000  # The sample rate of the signal in Hz.
DURATION = 30.0  # The duration of the signal in seconds.
FRAME_PERIOD = 5.0  # The frame period of the trackers in milliseconds.

# The signal alternates 2 seconds of voice and 0.5 seconds of silence.
_VOICED = 2.0
_SILENT = 0.5


def generate_signal() -> np.ndarray:
    """Generate a voice-like signal with vibrato alternating with silence.

    The F0 of the voiced segments rises from 100 to 250 Hz over the segments with
    a vibrato of 5 Hz, and weak noise is added to the whole signal.

    Returns
    -------
    out : np.ndarray
        The signal of 30 seconds at 16 kHz, which is the same at every call.

    """
    segment = int((_VOICED + _SILENT) * SAMPLE_RATE)
    num_segments = int(DURATION * SAMPLE_RATE) // segment
    voiced = int(_VOICED * SAMPLE_RATE)
    t = np.arange(voiced) / SAMPLE_RATE
    pieces = []
    for base in np.linspace(100.0, 250.0, num_segments):
        f0 = base * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))
        phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
        harmonics = np.arange(1, int(SAMPLE_RATE / 2 / f0.max()) + 1)
        voice = (np.sin(np.outer(phase, harmonics)) / harmonics).sum(axis=1)
        pieces += [0.3 * voice, np.zeros(segment - voiced)]
    x = np.concatenate(pieces)
    return x + 1e-3 * np.random.default_rng(0).standard_normal(len(x))


def describe_machine(device: str) -> str:
    """Describe the processor of the device.

    Parameters
    ----------
    device : str
        The device, i.e., cpu or cuda.

    Returns
    -------
    out : str
        The name of the GPU, or that of the CPU read from /proc/cpuinfo if any.

    """
    if device == "cuda":
        import torch

        return torch.cuda.get_device_name()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def measure(algorithm: str, device: str, save_path: str, repeats: int = 5) -> None:
    """Measure the RTF of a tracker and save it with the times and the machine.

    The tracker is loaded and run once as warm-up before the measurement, and the
    RTF is the median of the processing times divided by the duration.

    Parameters
    ----------
    algorithm : str
        The name of the algorithm with the default parameters.

    device : str
        The device, i.e., cpu with a single thread or cuda.

    save_path : str
        The path to the JSON file of the result.

    repeats : int, optional
        The number of the measurements.

    """
    tracker = create_tracker(
        algorithm,
        device=device,
        num_threads=1 if device == "cpu" else None,
    )
    x = generate_signal()
    tracker.load()
    tracker(x, SAMPLE_RATE, frame_period=FRAME_PERIOD)  # Warm-up.
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        tracker(x, SAMPLE_RATE, frame_period=FRAME_PERIOD)
        times.append(time.perf_counter() - start)
    result = {
        "algorithm": algorithm,
        "device": device,
        "times": times,
        "rtf": float(np.median(times)) / DURATION,
        "machine": describe_machine(device),
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)


def main() -> None:
    """Measure the RTF of the given tracker on the given device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="The directory of '<device>/*.json'.")
    parser.add_argument("--algorithm", required=True, help="The name of the tracker.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    save_path = os.path.join(args.output_dir, args.device, f"{args.algorithm}.json")
    measure(args.algorithm, args.device, save_path, args.repeats)


if __name__ == "__main__":
    main()
