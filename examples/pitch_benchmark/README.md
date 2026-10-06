# pitch-benchmark

Evaluation on the eval set of [pitch-benchmark](https://huggingface.co/datasets/lars1234/pitch-benchmark), a compilation of corpora with F0 labels (CC BY-NC-SA 4.0).

```sh
./run.sh
```

The script downloads the eval set, extracts F0 with each tracker individually, and updates the results below.

- The corpora of musical instruments, i.e., Bach10Synth and URMP, are excluded from all evaluations, and the other eight corpora of speech and singing voice are used.
- The clean audio is used: `identity` of the test split and `level`, which is only scaled in level, of the valid split.
- The F0 search range is fixed to 40–800 Hz, which covers all labels, and the other parameters are the defaults of the trackers.
- The F0 is extracted with the frame period of 4 ms, and the frames at the 16 ms frames of the labels are scored without interpolation.

The metrics are computed on the trusted frames pooled within each corpus with a tolerance of 50 cents, following `mir_eval`, and then averaged over the corpora.

- RPA (Raw Pitch Accuracy): the ratio of reference voiced frames estimated within the tolerance.
- RCA (Raw Chroma Accuracy): RPA ignoring octave errors.
- VDE (Voicing Decision Error): the ratio of frames with wrong voicing decisions.
- OA (Overall Accuracy): the ratio of frames that are correctly unvoiced or voiced with the F0 within the tolerance.

## Single pass

The results on the test split.

<!-- single:begin -->

| Tracker | RPA [%] | RCA [%] | VDE [%] | OA [%] |
| :--- | ---: | ---: | ---: | ---: |
| `dio` | 87.8 | 88.2 | 6.7 | 86.0 |
| `harvest` | 92.3 | 92.4 | 6.7 | 91.8 |
| `rapt` | 83.7 | 84.8 | 7.0 | 83.2 |
| `reaper` | 89.3 | 90.1 | 4.8 | 88.2 |
| `straight` | 90.5 | 90.8 | 8.3 | 88.1 |
| `swift-f0` | 92.7 | 92.8 | 6.2 | 92.3 |
| `swipe` | 88.1 | 88.2 | 8.0 | 87.8 |

<!-- single:end -->

## Settings of the two-pass tracking

The results of the settings of the two-pass tracking on both splits, which are used to choose the recommended settings, i.e., `two_pass_policy` of each algorithm.
Only the corpora of speech are used, which is the most common use case.
The setting improving OA by 0.1 points or more from `single` on both splits is recommended, otherwise the algorithm is not rerun.
The first pass uses the fixed range, and the range of the second pass is exp(μ ± 3σ) of the voiced log F0 of the first pass.

- `single`: the single pass.
- `two-pass-1st`: the two-pass tracking taking the voicing from the first pass.
- `two-pass-2nd`: the two-pass tracking taking the voicing from the second pass.

### Valid split

<!-- comparison_valid:begin -->

| Tracker | Setting | RPA [%] | RCA [%] | VDE [%] | OA [%] |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `dio` | single | 85.1 | 86.3 | 4.8 | 86.0 |
| `dio` | two-pass-1st | 85.2 | 86.3 | 4.8 | 86.0 |
| `dio` | two-pass-2nd | 85.9 | 87.0 | 4.0 | 86.4 |
| `harvest` | single | 91.1 | 91.3 | 6.3 | 91.5 |
| `harvest` | two-pass-1st | 91.1 | 91.3 | 6.3 | 91.6 |
| `harvest` | two-pass-2nd | 91.2 | 91.3 | 6.1 | 91.6 |
| `rapt` | single | 84.5 | 85.1 | 3.4 | 84.9 |
| `rapt` | two-pass-1st | 84.6 | 85.0 | 3.4 | 85.0 |
| `rapt` | two-pass-2nd | 73.8 | 74.2 | 12.0 | 76.9 |
| `reaper` | single | 87.4 | 88.7 | 5.9 | 86.0 |
| `reaper` | two-pass-1st | 87.5 | 88.6 | 5.9 | 86.1 |
| `reaper` | two-pass-2nd | 86.4 | 87.5 | 6.9 | 84.9 |
| `straight` | single | 88.1 | 88.5 | 9.4 | 85.4 |
| `straight` | two-pass-1st | 88.4 | 88.7 | 9.4 | 85.6 |
| `straight` | two-pass-2nd | 88.2 | 88.6 | 9.4 | 85.5 |
| `swift-f0` | single | 93.0 | 93.1 | 4.4 | 93.2 |
| `swift-f0` | two-pass-1st | 93.0 | 93.1 | 4.4 | 93.2 |
| `swift-f0` | two-pass-2nd | 92.5 | 92.6 | 4.9 | 92.7 |
| `swipe` | single | 84.4 | 84.6 | 9.5 | 85.3 |
| `swipe` | two-pass-1st | 83.3 | 83.5 | 9.5 | 84.2 |
| `swipe` | two-pass-2nd | 81.8 | 81.9 | 10.9 | 82.8 |

<!-- comparison_valid:end -->

### Test split

<!-- comparison_test:begin -->

| Tracker | Setting | RPA [%] | RCA [%] | VDE [%] | OA [%] |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `dio` | single | 88.0 | 88.3 | 5.2 | 87.4 |
| `dio` | two-pass-1st | 87.9 | 88.2 | 5.2 | 87.4 |
| `dio` | two-pass-2nd | 88.3 | 88.6 | 4.5 | 87.8 |
| `harvest` | single | 91.6 | 91.7 | 6.6 | 91.8 |
| `harvest` | two-pass-1st | 91.6 | 91.6 | 6.6 | 91.7 |
| `harvest` | two-pass-2nd | 91.9 | 92.0 | 6.1 | 92.0 |
| `rapt` | single | 82.2 | 83.2 | 6.9 | 82.5 |
| `rapt` | two-pass-1st | 82.5 | 83.1 | 6.9 | 82.7 |
| `rapt` | two-pass-2nd | 76.4 | 77.0 | 11.4 | 78.8 |
| `reaper` | single | 88.7 | 89.3 | 4.6 | 87.9 |
| `reaper` | two-pass-1st | 88.7 | 89.3 | 4.6 | 87.9 |
| `reaper` | two-pass-2nd | 89.1 | 89.7 | 3.9 | 88.2 |
| `straight` | single | 89.5 | 89.8 | 8.5 | 87.5 |
| `straight` | two-pass-1st | 89.2 | 89.7 | 8.5 | 87.2 |
| `straight` | two-pass-2nd | 89.3 | 89.7 | 8.2 | 87.3 |
| `swift-f0` | single | 92.0 | 92.1 | 6.3 | 92.0 |
| `swift-f0` | two-pass-1st | 92.0 | 92.1 | 6.3 | 92.0 |
| `swift-f0` | two-pass-2nd | 91.7 | 91.8 | 6.5 | 91.8 |
| `swipe` | single | 86.8 | 86.9 | 8.0 | 87.2 |
| `swipe` | two-pass-1st | 85.8 | 86.0 | 8.0 | 86.3 |
| `swipe` | two-pass-2nd | 85.4 | 85.5 | 8.4 | 85.9 |

<!-- comparison_test:end -->
