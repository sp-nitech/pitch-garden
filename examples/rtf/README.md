# RTF

Measurement of the real-time factors (RTFs) of the trackers, i.e., the processing time divided by the duration of the audio.

```sh
./run.sh
```

The script measures the RTF of each tracker and updates the results below.

- The audio is a synthetic signal of 30 seconds at 16 kHz, where 2 seconds of a voice-like harmonic tone with vibrato and 0.5 seconds of silence alternate.
- The F0 is extracted with the frame period of 5 ms by the one-pass tracking with the default parameters.
- The trackers are loaded and run once as warm-up, and then the median of five measurements is used.
- The trackers on CPU are run with a single thread, and those supporting GPUs are also measured on a GPU if available.
- `crepe` is measured only on a GPU because it is too slow on CPU, and the dashes indicate the RTFs not measured.

<!-- results:begin -->

| Tracker | RTF (CPU) | RTF (GPU) |
| :--- | ---: | ---: |
| `crepe` | — | 0.0426 |
| `dio` | 0.0057 | — |
| `fcn-f0` | 0.7988 | 0.0053 |
| `fcpe` | 0.0236 | 0.0003 |
| `harmof0` | 1.0345 | 0.0079 |
| `harvest` | 0.1652 | — |
| `pesto` | 0.1831 | 0.0014 |
| `pienet` | 0.0043 | 0.0002 |
| `pyin` | 0.0285 | — |
| `rapt` | 0.0014 | — |
| `reaper` | 0.0069 | — |
| `rmvpe` | 0.0729 | 0.0017 |
| `straight` | 0.1410 | — |
| `swift-f0` | 0.0068 | — |
| `swipe` | 0.0094 | — |
| `yaapt` | 0.0582 | — |
| `yin` | 0.0054 | — |

- CPU: 12th Gen Intel(R) Core(TM) i7-12700K
- GPU: NVIDIA GeForce RTX 3080 Ti

<!-- results:end -->
