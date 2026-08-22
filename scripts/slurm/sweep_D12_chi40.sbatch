#!/bin/bash
#SBATCH --job-name=ipeps-D12-chi40
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=ipeps_D12_chi40_%j.out
#SBATCH --error=ipeps_D12_chi40_%j.err

# =============================================================================
# Run iPEPS TFIM benchmark sweep at D=12, χ=40 on H100.
#
# Memory budget at D=12, χ=40:
#   - 4 double layers:        ≈ 27.5 GiB resident
#   - Working intermediates:  ≈  2– 5 GiB
#   - expect_2site peak:      +13.7 GiB transient
# Total peak: ≈ 45–55 GiB.  Fits H100 80 GB with headroom.
#
# Recommended wall time per h-point:
#   - SU at D=12: ~5–15 min (depends on schedule)
#   - CTMRG at χ=40, 60 max iter: ~30 min (SVDs dominate)
#   - Observables: ~10 min (4 × expect_2site + 2 × expect_1site)
#   - Total per point: ~45–60 min
# 21 h-points: ~16–20 hours.  Bump --time accordingly.
# =============================================================================

export PATH=$HOME/julia-1.10.10/bin:$HOME/.juliaup/bin:$PATH
export JULIA_DEPOT_PATH=/scratch/$USER/.julia
# Important: use the CUDA driver allocator (not the binned pool).  At our
# allocation sizes the binned pool fragments badly.
export JULIA_CUDA_MEMORY_POOL=cuda
# Cap GPU memory used by Julia at 70 GiB to leave headroom for CUDA driver / cuTENSOR plan caches
export JULIA_CUDA_HARD_MEMORY_LIMIT=70000000000

cd $SLURM_SUBMIT_DIR   # repo root (this file lives under scripts/slurm/)

echo "=== iPEPS D=12 χ=40 sweep on $(hostname) at $(date) ==="
nvidia-smi | head -15

julia --project=. scripts/benchmarks.jl

echo "=== sweep ended at $(date) ==="
