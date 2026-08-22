#!/bin/bash
#SBATCH --job-name=ipeps-D12-chi60
#SBATCH --partition=gpu_h200_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=sweep_D12_chi60_%j.out
#SBATCH --error=sweep_D12_chi60_%j.err

# =============================================================================
# D=12, χ=60 sweep with warm-start chaining and targeted 12-point h-sweep.
#
# Memory profile (estimated, verified against the χ=40 production run):
#   Peak ≈ 47 GB on H200 (141 GB) — 3× headroom.
#   H100 (80 GB) would also fit, but H200 is roomier.
#
# Runtime estimate:
#   - Cold first point     : ~25-30 min  (full SU + cold CTMRG)
#   - Warm subsequent points: ~5-8 min each (short SU + warm CTMRG)
#   - 11 warm + 1 cold = ~25 + 11*7 = ~100 min wall time
#   - 12h limit gives 7× buffer.
#
# Resume-safe: re-running picks up cached points from ipeps_tfim_sweep_D12_chi60.jls.
# (Warm chain restarts cold after any skipped/cached point.)
# =============================================================================

set -euo pipefail
echo "=== sweep started at $(date) ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

cd "${SLURM_SUBMIT_DIR:-$HOME/scratch/ipeps_D12_chi40}"   # repo root
echo "PWD: $PWD"
ls -la src/ipeps_tfim_gpu.jl scripts/benchmarks_chi60.jl 2>/dev/null

export PATH=$HOME/julia-1.10.10/bin:$PATH
export JULIA_DEPOT_PATH=/scratch/$USER/.julia
export JULIA_CUDA_MEMORY_POOL=cuda

nvidia-smi

# stdbuf forces line-buffered output so live tail works
stdbuf -oL -eL julia --project=. scripts/benchmarks_chi60.jl

echo "=== sweep ended at $(date) ==="
