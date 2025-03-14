#!/bin/bash

# Set JAX to use CPU instead of GPU to avoid memory issues
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu

# Activate the jax-gpu conda environment (assuming same as in run_read_mesh_jax.sh)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax-gpu

# Get number of processes from command line or default to 2
NPROCS=${1:-2}

# Ensure NPROCS is either 2 or 4
if [ "$NPROCS" -ne 2 ] && [ "$NPROCS" -ne 4 ]; then
    echo "Error: Number of processes must be either 2 or 4."
    exit 1
fi

# Run the optimized halo exchange benchmark
echo "Running optimized halo exchange benchmark with $NPROCS MPI processes..."
mpirun -n $NPROCS python test_halo_exchange_optimized.py

echo "Done!"
