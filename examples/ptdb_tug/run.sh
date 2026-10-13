#!/bin/bash
# Evaluate the trackers on PTDB-TUG, a speech database with laryngograph signals.
# Run this script in the environment where pitch-garden[all] is installed.

set -euo pipefail

cd "$(dirname "$0")"

url="https://www2.spsc.tugraz.at/databases/PTDB-TUG/SPEECH_DATA_ZIPPED.zip"
md5="e590798aa0befaf7b458d3c38cda4030"
db="downloads/SPEECH DATA"
exp=exp
nj=8        # The number of worker processes for the trackers running on CPU.
device=auto # The device of the neural trackers, where auto prefers a GPU.

# Download the dataset, where only the microphone signals and the references are
# extracted.
if [ ! -d "$db" ]; then
    mkdir -p downloads
    wget -q "$url" -O downloads/SPEECH_DATA_ZIPPED.zip
    echo "$md5  downloads/SPEECH_DATA_ZIPPED.zip" | md5sum -c --quiet
    # Python is used instead of unzip, which is not always installed.
    python -c '
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
z.extractall(sys.argv[2], [n for n in z.namelist() if "/MIC/" in n or "/REF/" in n])
' downloads/SPEECH_DATA_ZIPPED.zip downloads
    rm downloads/SPEECH_DATA_ZIPPED.zip
fi

# Extract F0 with each tracker.
# The algorithms given as the arguments, e.g., ./run.sh pesto, or all the registered
# ones. The results of the other algorithms in exp are kept and scored as well.
algorithms="$*"
if [ -z "$algorithms" ]; then
    algorithms=$(python -c "from pitch_garden.algorithms import ALGORITHM_MAP; print(*ALGORITHM_MAP)")
fi
for algorithm in $algorithms; do
    echo "Extracting F0 with $algorithm..."
    python local/extract.py "$db" "$exp" --algorithm "$algorithm" \
        --num_workers "$nj" --device "$device"
done

# Score the trackers and update the results in README.md.
python local/score.py "$db" "$exp" --readme README.md

echo "run.sh: successfully finished"
