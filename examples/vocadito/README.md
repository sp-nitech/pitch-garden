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
| `dio` | 85.8 | 86.4 | 15.7 | 77.7 |
| `harvest` | 96.5 | 96.7 | 7.2 | 92.1 |
| `rapt` | 95.6 | 96.9 | 7.0 | 90.4 |
| `reaper` | 95.6 | 96.1 | 9.9 | 87.9 |
| `straight` | 97.4 | 97.4 | 6.7 | 92.3 |
| `swift-f0` | 97.1 | 97.2 | 5.2 | 93.9 |
| `swipe` | 97.0 | 97.1 | 6.8 | 92.6 |

<!-- results:end -->
