# PTDB-TUG

Evaluation on [PTDB-TUG](https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html), a database of 4,718 English sentences read by 10 female and 10 male native speakers with laryngograph signals (ODbL 1.0).

```sh
./run.sh
```

The script downloads the database, extracts F0 from the microphone signals with each tracker individually, and updates the results below.

- The reference F0 provided in the database, which is extracted from the high-pass filtered laryngograph signals by RAPT every 10 ms, is used as it is, although it is partly questionable.
- The first reference frame is regarded as at 16 ms, which is chosen by comparing the reference with the trackers.
- The F0 search range is fixed to 40–800 Hz, and the other parameters are the defaults of the trackers.
- The trackers are run under their default policies of the two-pass tracking, which are chosen on the speech of [pitch-benchmark](../pitch_benchmark), where the first pass uses the fixed search range.
- The F0 is extracted with the frame period of 2 ms, and the frames at the reference frames are scored without interpolation.
- `swift-f0` is trained on PTDB-TUG, and thus its result is biased in favor of it. The training data of `pienet` is unknown.
- `dio` regards most of the silent segments, which are very quiet in this database, as voiced with the F0 near the lower bound of the search range. Its own second pass does not narrow the range because the range is estimated from its output including those frames.

The metrics are computed on the pooled frames of all sentences with a tolerance of 50 cents, following `mir_eval`.

- RPA (Raw Pitch Accuracy): the ratio of reference voiced frames estimated within the tolerance.
- RCA (Raw Chroma Accuracy): RPA ignoring octave errors.
- VDE (Voicing Decision Error): the ratio of frames with wrong voicing decisions.
- OA (Overall Accuracy): the ratio of frames that are correctly unvoiced or voiced with the F0 within the tolerance.

<!-- results:begin -->

| Tracker | RPA [%] | RCA [%] | VDE [%] | OA [%] |
| :--- | ---: | ---: | ---: | ---: |
| `swift-f0` | 85.2 | 86.4 | 4.0 | 94.5 |
| `fcpe` | 92.2 | 92.9 | 5.2 | 93.4 |
| `rmvpe` | 92.2 | 92.9 | 5.4 | 93.2 |
| `pienet` | 83.0 | 84.3 | 4.7 | 93.1 |
| `swipe` | 77.6 | 78.5 | 5.5 | 92.7 |
| `rapt` | 77.8 | 79.2 | 5.1 | 91.8 |
| `reaper` | 78.6 | 79.5 | 4.9 | 91.4 |
| `harmof0` | 80.6 | 81.6 | 6.6 | 90.9 |
| `pesto` | 81.2 | 84.2 | 5.7 | 90.8 |
| `harvest` | 81.1 | 81.8 | 7.6 | 90.5 |
| `straight` | 82.7 | 84.5 | 6.5 | 90.2 |
| `crepe` | 84.9 | 86.4 | 9.5 | 88.3 |
| `pyin` | 74.7 | 75.6 | 8.8 | 87.2 |
| `yaapt` | 67.7 | 69.3 | 6.1 | 86.9 |
| `yin` | 75.3 | 77.5 | 10.0 | 85.3 |
| `fcn-f0` | 85.9 | 87.2 | 15.3 | 82.5 |
| `dio` | 80.0 | 82.0 | 56.5 | 39.2 |

<!-- results:end -->
