#!/usr/bin/env bash
# ==============================================================================
# DustImpact - Server / HPC Execution Launcher
# ==============================================================================
set -e

# 1. Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# 2. Configure headless Matplotlib and OpenMP / BLAS threading
export MPLBACKEND="Agg"
if [ -z "$OMP_NUM_THREADS" ]; then
    export OMP_NUM_THREADS=4
fi

echo "====================================================="
echo "   DUSTIMPACT SERVER RUNNER"
echo "   Python: $(which python3)"
echo "   OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "   MPLBACKEND: $MPLBACKEND"
echo "====================================================="

# 3. Execute main script with passed arguments (defaults to --no-visualize)
if [[ " $* " =~ " --visualize " ]]; then
    python3 main.py "$@"
else
    python3 main.py --no-visualize "$@"
fi
