#!/bin/bash
# Evaluate the trackers on Vocadito, a dataset of solo singing voice.
# Run this script in the environment where pitch-garden[all] is installed.

set -euo pipefail

cd "$(dirname "$0")"

url="https://zenodo.org/records/5578807/files/vocadito.zip?download=1"
md5="dea40fd18f14d899643c4ba221b33a46"
db=downloads/vocadito
exp=exp

# Download the dataset.
if [ ! -d "$db" ]; then
    mkdir -p downloads
    wget -q "$url" -O downloads/vocadito.zip
    echo "$md5  downloads/vocadito.zip" | md5sum -c --quiet
    unzip -q downloads/vocadito.zip -x "__MACOSX/*" -d "$db"
    rm downloads/vocadito.zip
fi

# Extract F0 with each tracker.
algorithms="swift-f0 rapt swipe reaper dio harvest straight"
for algorithm in $algorithms; do
    echo "Extracting F0 with $algorithm..."
    python local/extract.py "$db" "$exp" --algorithm "$algorithm"
done

# Score the trackers and update the results in README.md.
python local/score.py "$db/Annotations/F0" "$exp" --readme README.md

echo "run.sh: successfully finished"
