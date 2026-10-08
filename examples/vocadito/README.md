# Vocadito

Evaluation on [Vocadito](https://zenodo.org/records/5578807), a dataset of 40 excerpts of solo singing voice in 7 languages with manually annotated F0 (CC BY 4.0).

```sh
./run.sh
```

The script downloads the dataset, extracts F0 with each tracker individually, and updates the results below.
The F0 search range is fixed to 40–800 Hz, and the other parameters are the defaults of the trackers.
The F0 is extracted with the frame period of the reference, i.e., 256 samples at 44.1 kHz.

The metrics are computed on the pooled frames of all excerpts with a tolerance of 50 cents, following `mir_eval`.

- RPA (Raw Pitch Accuracy): the ratio of reference voiced frames estimated within the tolerance.
- RCA (Raw Chroma Accuracy): RPA ignoring octave errors.
- VDE (Voicing Decision Error): the ratio of frames with wrong voicing decisions.
- OA (Overall Accuracy): the ratio of frames that are correctly unvoiced or voiced with the F0 within the tolerance.

<!-- results:begin -->

| Tracker | RPA [%] | RCA [%] | VDE [%] | OA [%] |
| :--- | ---: | ---: | ---: | ---: |
| `crepe` | 98.8 | 98.9 | 12.8 | 86.7 |
| `dio` | 85.8 | 86.4 | 15.7 | 77.7 |
| `fcn-f0` | 97.0 | 97.4 | 12.3 | 86.2 |
| `fcpe` | 98.9 | 99.0 | 10.6 | 88.7 |
| `harmof0` | 96.2 | 96.5 | 10.3 | 88.3 |
| `harvest` | 96.5 | 96.7 | 7.2 | 92.1 |
| `pesto` | 98.3 | 98.9 | 10.5 | 88.5 |
| `pienet` | 90.7 | 96.1 | 6.0 | 88.6 |
| `pyin` | 96.8 | 97.0 | 12.3 | 85.9 |
| `rapt` | 95.6 | 96.9 | 7.0 | 90.4 |
| `reaper` | 95.6 | 96.1 | 9.9 | 87.9 |
| `rmvpe` | 98.4 | 98.5 | 10.0 | 89.0 |
| `straight` | 97.4 | 97.4 | 6.7 | 92.3 |
| `swift-f0` | 97.1 | 97.2 | 5.2 | 93.9 |
| `swipe` | 97.0 | 97.1 | 6.8 | 92.6 |
| `yaapt` | 97.1 | 97.2 | 6.4 | 92.2 |
| `yin` | 95.4 | 96.6 | 14.4 | 82.7 |

<!-- results:end -->
