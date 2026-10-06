#!/bin/bash
# Evaluate the trackers on the eval set of pitch-benchmark.
# Run this script in the environment where pitch-garden[all] is installed.

set -euo pipefail

cd "$(dirname "$0")"

# The revision of the dataset on Hugging Face and the checksum of the eval set.
rev="e98642fc2cb82e4dc201421b24f24c523989d344"
url="https://huggingface.co/datasets/lars1234/pitch-benchmark/resolve/$rev/eval.tar"
sha256="f4de960de53c1b8d069ade901b8145c0ccc311bd3760b613bd32474fa8b3a374"
db=downloads/eval
exp=exp
nj=8

# Download the eval set.
if [ ! -d "$db" ]; then
    mkdir -p downloads
    wget -q "$url" -O downloads/eval.tar
    echo "$sha256  downloads/eval.tar" | sha256sum -c --quiet
    tar xf downloads/eval.tar -C downloads
    rm downloads/eval.tar
fi

algorithms="swift-f0 rapt swipe reaper dio harvest straight"

extract() {
    local split=$1
    local setting=$2
    for algorithm in $algorithms; do
        echo "Extracting F0 from $split with $algorithm under $setting..."
        python local/extract.py "$db" "$exp/$split" --algorithm "$algorithm" \
            --setting "$setting" --split "$split" --num_workers "$nj"
    done
}

# Table 1: the single pass on the test split.
extract test single
python local/score.py "$db" "$exp/test" --split test --settings single \
    --readme README.md --section single

# Table 2: the settings of the two-pass tracking on the speech of both splits.
settings="single two-pass-1st two-pass-2nd"
for split in valid test; do
    for setting in $settings; do
        extract "$split" "$setting"
    done
    # shellcheck disable=SC2086
    python local/score.py "$db" "$exp/$split" --split "$split" --settings $settings \
        --speech_only --readme README.md --section "comparison_$split"
done

echo "run.sh: successfully finished"
