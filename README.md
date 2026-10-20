# pitch-garden

*pitch-garden* provides a unified interface to various pitch (F0) trackers and integrates their outputs by voting.
The outputs of all trackers are aligned on a common frame grid, so that they can be directly compared and voted.

[![Manual](https://img.shields.io/badge/docs-stable-blue.svg)](https://takenori-y.github.io/pitch-garden/stable/)
[![Downloads](https://static.pepy.tech/badge/pitch-garden)](https://pepy.tech/project/pitch-garden)
[![ClickPy](https://img.shields.io/badge/downloads-clickpy-yellow.svg)](https://clickpy.clickhouse.com/dashboard/pitch-garden)
[![Python Version](https://img.shields.io/pypi/pyversions/pitch-garden.svg)](https://pypi.python.org/pypi/pitch-garden)
[![PyPI](https://img.shields.io/pypi/v/pitch-garden.svg)](https://pypi.python.org/pypi/pitch-garden)
[![License](https://img.shields.io/github/license/takenori-y/pitch-garden.svg)](https://github.com/takenori-y/pitch-garden/blob/master/LICENSE)
[![GitHub Actions](https://github.com/takenori-y/pitch-garden/workflows/package/badge.svg)](https://github.com/takenori-y/pitch-garden/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Features

- Voting of multiple trackers
- Two-pass tracking with an F0 search range estimated from the first pass
- Parallel execution of the trackers

## Requirements

- Python 3.12+

## Documentation

- See [this page](https://takenori-y.github.io/pitch-garden/stable/) for the reference manual.

## Installation

```sh
pip install "pitch-garden[all]"  # All trackers

# To avoid installing PyTorch, use the following instead.
# pip install "pitch-garden[lite]"  # Trackers not requiring PyTorch
```

## Supported Algorithms

| Name | Neural | F0 Range [Hz] | Voicing Threshold | Paper | Source |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `crepe` | ✓ | 32 – 2006 | 0.1 – 0.4 | [arXiv](https://arxiv.org/abs/1802.06182) | [GitHub](https://github.com/maxrmorrison/torchcrepe) |
| `dio` | | 10 – Nyquist | 0.2 – 0.02 | [AES E-Library](https://aes.org/publications/elibrary/elibrary-page/?id=15165) | [GitHub](https://github.com/mmorise/World) |
| `fcn-f0` | ✓ | 30 – 1000 | 0.2 – 0.6 | [ISCA Archive](https://www.isca-archive.org/interspeech_2019/ardaillon19_interspeech.html) | [GitHub](https://github.com/ardaillon/FCN-f0) |
| `fcpe` | ✓ | 33 – 1975 | 0.0 – 0.1 | [arXiv](https://arxiv.org/abs/2509.15140) | [GitHub](https://github.com/CNChTu/FCPE) |
| `harmof0` | ✓ | 28 – 4371 | 0.5 – 0.9 | [arXiv](https://arxiv.org/abs/2205.01019) | [GitHub](https://github.com/WX-Wei/HarmoF0) |
| `harvest` | | 10 – Nyquist | 0.2 – 0.0 | [ISCA Archive](https://www.isca-archive.org/interspeech_2017/morise17b_interspeech.html) | [GitHub](https://github.com/mmorise/World) |
| `pesto` | ✓ | 28 – 3452 | 0.0 – 0.2 | [arXiv](https://arxiv.org/abs/2309.02265) | [GitHub](https://github.com/SonyCSLParis/pesto) |
| `pienet` | ✓ | 50 – 500 | 0.5 – 1.0 | [IEEE Xplore](https://ieeexplore.ieee.org/document/8683041) | [GitHub](https://github.com/mairaksi/PiENet) |
| `pyin` | | 16 – 8000 | | [IEEE Xplore](https://ieeexplore.ieee.org/document/6853678) | [GitHub](https://github.com/librosa/librosa) |
| `rapt` | | 10 – Nyquist | 0.7 – -0.6 | [EDU](https://www.ee.columbia.edu/~dpwe/papers/Talkin95-rapt.pdf) | [GitHub](https://github.com/scottypitcher/tcl-snack) |
| `reaper` | | 10 – Nyquist | 1.6 – -0.5 | | [GitHub](https://github.com/google/reaper) |
| `rmvpe` | ✓ | 32 – 2005 | 0.0 – 0.3 | [arXiv](https://arxiv.org/abs/2306.15412) | [GitHub](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) |
| `straight` | | 40 – Nyquist | | [ISCA Archive](https://www.isca-archive.org/interspeech_2005/kawahara05_interspeech.html) | [GitHub](https://github.com/takenori-y/pylstraight) |
| `swift-f0` | ✓ | 47 – 2093 | 0.1 – 0.5 | [arXiv](https://arxiv.org/abs/2508.18440) | [GitHub](https://github.com/lars76/swift-f0) |
| `swipe` | | 10 – Nyquist | 0.2 – 0.5 | [EDU](https://scholarsmine.mst.edu/cgi/viewcontent.cgi?article=8070&context=ele_comeng_facwork) | [GitHub](https://github.com/kylebgorman/swipe) |
| `yaapt` | | 10 – Nyquist | 0.1 – 1.0 | [IEEE Xplore](https://ieeexplore.ieee.org/document/5743729) | [GitHub](https://github.com/bjbschmitt/AMFM_decompy) |
| `yin` | | 16 – 8000 | 0.3 – 0.6 | [JASA](https://doi.org/10.1121/1.1458024) | [GitHub](https://github.com/librosa/librosa) |

The F0 range is the search range that can be specified.
The voicing threshold is shown as a typical range, where moving it toward the right value gives more unvoiced frames.
`pesto` is not available on Python 3.15 or later, because the pesto-pitch package requires torchaudio, which does not support those versions.
`dio`, `harvest`, `rapt`, `reaper`, and `swipe` are based on the implementations in [SPTK](https://github.com/sp-nitech/SPTK), and their outputs are compatible with the `pitch` command of SPTK.

## Unsupported Algorithms

| Name | Neural | Paper | Source | Reason |
| :--- | :---: | :---: | :---: | :--- |
| Basic Pitch | ✓ | [arXiv](https://arxiv.org/abs/2203.09893) | [GitHub](https://github.com/spotify/basic-pitch) | Designed for polyphonic transcription |
| JDCNet | ✓ | [MDPI](https://doi.org/10.3390/app9071324) | [GitHub](https://github.com/keums/melodyExtraction_JDC) | PyTorch versions are unlicensed |
| Kaldi Pitch | | [IEEE Xplore](https://doi.org/10.1109/ICASSP.2014.6854049) | [GitHub](https://github.com/kaldi-asr/kaldi) | No maintained Python implementation |
| MPM | | [EDU](https://www.cs.otago.ac.nz/research/publications/oucs-2008-03.pdf) | | Similar to `yin` |
| PENN (FCNF0++) | ✓ | [arXiv](https://arxiv.org/abs/2301.12258) | [GitHub](https://github.com/interactiveaudiolab/penn) | Did not outperform `fcn-f0` in our evaluation |
| Praat (Parselmouth) | | [EDU](https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf) | [GitHub](https://github.com/YannickJadoul/Parselmouth) | Licensed under GPL-3.0 |
| SPICE | ✓ | [arXiv](https://arxiv.org/abs/1910.11664) | [Kaggle](https://www.kaggle.com/models/google/spice) | No PyTorch weights available |

## Usage

### Input preparation

Audio files, headerless raw files, and NumPy arrays are accepted.

```python
import numpy as np

import pitch_garden as pg

tracker = pg.Tracker("rapt")

# An audio file readable by soundfile, where multiple channels are averaged.
audio = pg.load_audio("speech.wav")
pitch = tracker(audio)

# A headerless raw file with a data type of SPTK x2x, e.g., 16-bit integers.
raw = pg.RawFormat(sample_rate=16000, data_type="s", num_channels=1)
audio = pg.load_audio("speech.raw", raw)
pitch = tracker(audio)

# A NumPy array with its sample rate.
sample_rate = 16000
waveform = np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate)
pitch = tracker(waveform, sample_rate)
```

### Single algorithm

A `Tracker` holds an algorithm and its own parameters.
The F0 search range given by `fmin` and `fmax` at the creation is the default of the tracker, and can be overridden only for a call by giving `fmin` and/or `fmax` to the call, e.g., if the range is known for each file.
The bound not given at the call is taken from the creation, and changing the range does not reload the algorithm.
The trackers supporting GPUs, e.g., `crepe`, use a GPU if available, which is chosen when the tracker is created and can be changed by `tracker.replace(device="cpu")`.
The output `Pitch` holds the F0 in Hz on a uniform frame grid regardless of the native frame period of the algorithm, and converts it to other formats on demand.

```python
import pitch_garden as pg

audio = pg.load_audio("speech.wav")

tracker = pg.Tracker("rapt", voicing_threshold=0.0, fmin=60.0, fmax=800.0)
pitch = tracker(audio, frame_period=5.0)
print(pitch)
# Pitch(source='rapt', num_frames=100, num_voiced_frames=..., ...)
print(pitch.f0)  # The F0 in Hz as a NumPy array, where unvoiced frames are 0.

# The search range is overridden only for this call, i.e., 80-800 Hz.
pitch = tracker(audio, frame_period=5.0, fmin=80.0)

# The F0 is converted to a NumPy array of another format.
lf0 = pitch.to("log")  # log F0 (unvoiced: -1e10)
period = pitch.to("inverse")  # Fs/F0 in samples (unvoiced: 0)

# The F0 is converted and saved; the pitch object itself is unchanged.
pitch.tofile("speech.lf0", output_format="log", file_type="double")
```

> [!TIP]
> The model weights not included in the packages are downloaded at the first use to `$TORCH_HOME/hub/checkpoints` (by default `~/.cache/torch/hub/checkpoints`), or can be given by the parameter `weights_path`.

### Multiple algorithms and voting

A `Pipeline` runs the trackers in parallel where possible and votes their outputs.
The same algorithm can be used with different parameters by giving different names.
The F0 search range can be given at a call as with `Tracker`, which overrides the ranges of all trackers supporting it only for the call.

```python
import pitch_garden as pg

audio = pg.load_audio("speech.wav")

trackers = [pg.Tracker("rapt"), pg.Tracker("swift-f0")]
with pg.Pipeline(trackers, voter="median") as pipeline:
    result = pipeline(audio, frame_period=5.0)

# The final contour voted from the trackers as a Pitch object
print(result.pitch)
# Pitch(source='median', num_frames=100, num_voiced_frames=..., ...)

# The outputs of the trackers as Pitch objects keyed by the tracker names
for name, pitch in result.each.items():
    print(name, pitch)
# rapt Pitch(source='rapt', num_frames=100, ...)
# swift-f0 Pitch(source='swift-f0', num_frames=100, ...)
```

> [!NOTE]
> If a tracker fails, e.g., on very short audio, it is excluded from voting with a warning and its error message is kept in `result.errors`.
> Give `on_error="raise"` to `Pipeline` to raise the error instead.

### Two-pass tracking

If the F0 range of the speaker is unknown, the trackers are first run with a wide search range, and then rerun with the range estimated from the distribution of the F0 of the first pass.
This works with a single tracker as well as multiple ones, where the estimated range is shared by all trackers.
The first-pass range is given by `TwoPass`, and can be overridden only for a call by giving `fmin` and/or `fmax` to the call, e.g., if a rough range is known for each file.
Each algorithm has its recommended settings chosen in [examples/pitch_benchmark](https://github.com/takenori-y/pitch-garden/tree/master/examples/pitch_benchmark), i.e., whether it is rerun and which pass gives the voicing decisions, which can be overridden by `two_pass_policy` of each tracker.

```python
import pitch_garden as pg
from pitch_garden.ranges.log_normal import LogNormalRangeEstimator

audio = pg.load_audio("speech.wav")

two_pass = pg.TwoPass(estimator=LogNormalRangeEstimator(), fmin=40.0, fmax=2000.0)
with pg.Pipeline([pg.Tracker("rapt")], two_pass=two_pass) as pipeline:
    result = pipeline(audio)

# The estimated range
print(result.f0_range)
# F0Range(fmin=111.96..., fmax=224.08..., num_voiced_frames=..., ...)

# The final contour of the second pass, i.e., the output of the single tracker
print(result.pitch)
# Pitch(source='rapt', num_frames=100, num_voiced_frames=..., ...)

# The result of the first pass, which is of the same type as the final result
print(result.pass1.pitch)
# Pitch(source='rapt', num_frames=100, num_voiced_frames=..., ...)
```

### Command-line interface

The settings are given by a YAML file, and the input and the output directory are given by the command line.
The input is an audio file, a directory searched recursively, or a text file listing audio files line by line.
See [configs/single.yaml](https://github.com/takenori-y/pitch-garden/blob/master/configs/single.yaml) for all settings, [configs/multiple.yaml](https://github.com/takenori-y/pitch-garden/blob/master/configs/multiple.yaml) for multiple trackers, and [configs/two-pass.yaml](https://github.com/takenori-y/pitch-garden/blob/master/configs/two-pass.yaml) for the two-pass tracking.

```sh
pitch-garden configs/multiple.yaml data/wav -o exp/f0
```

The main settings for the above use cases are as follows:

```yaml
trackers:
  - algorithm: rapt
  - algorithm: swift-f0
voter:
  type: median

two_pass:
  scope: group # The unit to estimate the range: utterance, group, or global.
  group_by: {type: dirname, level: 1}
```

With `scope: group`, the range is estimated for each group of files, e.g., each speaker.
The groups are defined by `group_by` in either of the following ways:

- `{type: dirname, level: 1}` groups the files by their directory `level` levels above, where `level: 1` is the directory containing the files.
  For example, the files below are grouped into `spk1` and `spk2` with `level: 1`, and into a single group `data/wav` with `level: 2`.

  ```
  data/wav/spk1/a.wav
  data/wav/spk1/b.wav
  data/wav/spk2/c.wav
  ```

- `{type: map, path: data/speaker.map}` groups the files by a text file, where each line is the file name without the extension followed by the group name.

  ```
  a spk1
  b spk1
  c spk2
  ```

With `scope: utterance` and `scope: global`, the range is estimated for each file and for all files, respectively.

The final contours are saved under `<output_dir>/pitch`, which are the voted ones for multiple trackers.
For multiple trackers, their outputs are also saved under `<output_dir>/trackers/<name>` if `save_each` is enabled or voting is disabled.
For the two-pass tracking, the estimated ranges are saved in `<output_dir>/f0_range.yaml`.

## License

This software is released under the MIT License.
