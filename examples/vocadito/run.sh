#!/bin/bash
# Evaluate the trackers on Vocadito, a dataset of solo singing voice.
# Run this script in the environment where pitch-garden[all] is installed.

set -euo pipefail

cd "$(dirname "$0")"

url="https://zenodo.org/records/5578807/files/vocadito.zip?download=1"
md5="dea40fd18f14d899643c4ba221b33a46"
db=downloads/vocadito
exp=exp
device=auto # The device of the neural trackers, where auto prefers a GPU.

# Download the dataset.
if [ ! -d "$db" ]; then
    mkdir -p downloads
    wget -q "$url" -O downloads/vocadito.zip
    echo "$md5  downloads/vocadito.zip" | md5sum -c --quiet
    # Python is used instead of unzip, which is not always installed.
    python -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
        downloads/vocadito.zip "$db"
    rm -rf "$db/__MACOSX"
    rm downloads/vocadito.zip
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
    python local/extract.py "$db" "$exp" --algorithm "$algorithm" --device "$device"
done

# Score the trackers and update the results in README.md.
python local/score.py "$db/Annotations/F0" "$exp" --readme README.md

echo "run.sh: successfully finished"
