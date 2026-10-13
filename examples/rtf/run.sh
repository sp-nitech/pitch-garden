#!/bin/bash
# Measure the real-time factors (RTFs) of the trackers on a synthetic signal.
# Run this script in the environment where pitch-garden[all] is installed.

set -euo pipefail

cd "$(dirname "$0")"

exp=exp
# The trackers measured only on a GPU, which are too slow on CPU, e.g., CREPE
# running a large model at every frame.
gpu_only="crepe"

# The trackers on CPU are measured with a single thread.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# The algorithms given as the arguments, e.g., ./run.sh pesto, or all the registered
# ones. The results of the other algorithms in exp are kept and reported as well.
algorithms="$*"
if [ -z "$algorithms" ]; then
    algorithms=$(python -c "from pitch_garden.algorithms import ALGORITHM_MAP; print(*ALGORITHM_MAP)")
fi
for algorithm in $algorithms; do
    if [[ " $gpu_only " != *" $algorithm "* ]]; then
        echo "Measuring the RTF of $algorithm on CPU..."
        python local/measure.py "$exp" --algorithm "$algorithm" --device cpu
    fi
    # The trackers supporting GPUs are also measured on a GPU if available.
    if python -c "
import sys, torch
from pitch_garden.algorithms import ALGORITHM_MAP
sys.exit(not (ALGORITHM_MAP[sys.argv[1]].supports_gpu and torch.cuda.is_available()))
" "$algorithm" 2>/dev/null; then
        echo "Measuring the RTF of $algorithm on GPU..."
        python local/measure.py "$exp" --algorithm "$algorithm" --device cuda
    fi
done

# Tabulate the RTFs and update the results in README.md.
python local/report.py "$exp" --readme README.md

echo "run.sh: successfully finished"
