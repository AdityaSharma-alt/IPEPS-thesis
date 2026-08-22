#!/bin/bash
#SBATCH --job-name=ipeps-FU-D4-chi20
#SBATCH --partition=gpu_h200_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=fu_sweep_%j.out
#SBATCH --error=fu_sweep_%j.err

# =============================================================================
# Full 22-point Full Update sweep at D=4, χ=20 across the TFIM phase diagram.
# Resume-safe: re-running picks up where it left off via fu_sweep_D4_chi20.jls
#
# Estimated runtime: ~3-4 hours.  Time limit set to 12h for safety.
# =============================================================================

set -euo pipefail
echo "=== FU sweep started at $(date) ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd ~/scratch/ipeps_D12_chi40   # repo root
echo "PWD: $PWD"
ls -la src/ipeps_fu.jl scripts/fu_sweep.jl

export PATH=$HOME/julia-1.10.10/bin:$PATH
export JULIA_DEPOT_PATH=/scratch/$USER/.julia
export JULIA_CUDA_MEMORY_POOL=cuda

nvidia-smi

# stdbuf forces line-buffered output so live tail works
stdbuf -oL -eL julia --project=. scripts/fu_sweep.jl

echo "=== FU sweep ended at $(date) ==="
