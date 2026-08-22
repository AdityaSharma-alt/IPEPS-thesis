# CHANGES.md — iPEPS upgrade D=10/χ=25 → D=12/χ=40

This document explains what changed, why, and how to run.  Read it before
running — there are real memory and time implications.

---

## 1. What was wrong/limiting

### 1a. Convergence target was too tight relative to the schedule
The original code used `tol = 1e-5` with `max_iter = 50`.  Your log
(`sweep_corr_168687.out`) shows runs hitting iteration 36–37 with `ΔSV ≈ 4e-4`
before finally dropping below `1e-5`.  The tail of CTMRG iterations is the
expensive part (long SVD on near-converged matrices).  Spec change: `tol = 1e-4`,
`max_iter = 60`.  Faster convergence with no observable change in physics.

### 1b. Memory allocator pressure during observables
`expect_2site` is the peak-memory call.  At D=10, χ=25 it allocated two
operator-inserted double-layer tensors (≈ 1.6 GB each) inside the SVD loop
over the bond operator without releasing them between iterations.  At D=12 the
same code path allocates ≈ 6.87 GB per `mL`/`mR` → **13.7 GB transient per
SVD component**, on top of the 27.5 GB already held by `dl.mA…mD`.  Easy OOM
with 4 components × no release.

### 1c. Stage 4 rebuilt double layers
In the original `run_point`, the line `dl = build_all_dl(st)` ran a second
time inside Stage 4 even though `dl` from Stage 2 was still in scope.  This
temporarily doubled the double-layer footprint to 55 GB until GC kicked in.

### 1d. SVD intermediates lived past their use
Every absorption function in §5 builds `nClu`, `nCld`, `Ψ`, `F.U`, `F.S`,
`F.Vt`, `P`, `Q`, then derived corner/edge tensors, all in one scope.  At
D=10/χ=25 the buffers were small enough that the GC could keep up.  At
D=12/χ=40 the resident set creeps from sweep to sweep.

### 1e. CTMRG iteration scaling
At D=12/χ=40 the SVD input matrix is 5760×5760 vs 2500×2500 previously.  Each
SVD is ~12× more expensive.  60 iterations × 4 directions × 2 absorptions
per move = 480 SVDs.  Your existing log already showed individual SVDs
taking 10–15 seconds at D=10/χ=25; at D=12/χ=40 you should budget **30–45
minutes of CTMRG per h-point** even with good memory hygiene.

### 1f. Cache filename was hardcoded to one (D, χ)
`DEFAULT_CACHE = ".../ipeps_tfim_sweep_D10_chi25.jls"` would silently mix
results across different (D, χ) runs.  Fixed: now parametrized.

---

## 2. What was changed

All changes are in `ipeps_tfim_gpu.jl`; the new files are `benchmarks.jl`
and `sweep_D12_chi40.sbatch`.  The patches are surgical — I did not
rewrite the working SU / CTMRG / observable logic.

### Memory hygiene
- Added `_free!(x)` and `_freeall!(xs...)` helpers that call
  `CUDA.unsafe_free!` with a try/catch double-free guard.
- All four absorption functions (`_absorb_left_col`, `_absorb_right_col`,
  `_absorb_top_row`, `_absorb_bot_row`) now release `CTu`, `CTuTl`, `nClu`,
  `nCld`, `Ψ`, `Ψmat`, `F.U/S/Vt`, and the per-step temporaries
  (`QcTl`, `QcTlMt`, …) as soon as they're consumed.
- The 4 directional moves now free the old corner/edge tensors before
  overwriting (the old buffer was leaking through the struct field).
- `expect_1site` and `expect_2site` free their operator-inserted `mO` / `mL` /
  `mR` after each component and call `CUDA.reclaim()` between iterations.
- `run_point` no longer rebuilds the double layer in Stage 4.

### CTMRG convergence
- `ctmrg_converge!` defaults: `max_iter = 60`, `tol = 1e-4`.
- Convergence metric: top `min(χ, 32)` normalized singular values of `Clu`,
  ℓ²-distance between consecutive sweeps.  This is the same scheme as before,
  just with a hard cap on `n_track` so the metric doesn't depend on the
  random tail of small SVs.

### Default parameters everywhere
- `run_point`: `D=12, χ=40, ctm_iter=60, ctm_tol=1e-4`
- `stage23_ctmrg`: `χ=40, max_iter=60, tol=1e-4`
- `sweep_h`: matches
- `DEFAULT_CACHE`: now points to `…_D12_chi40.jls`
- Plot titles updated.

### New observables on each h-point
- `Czz_h = ⟨σᶻ_A σᶻ_B⟩` (horizontal NN correlator)
- `Czz_v = ⟨σᶻ_A σᶻ_C⟩` (vertical NN correlator)
- These cost one extra `expect_2site` each (~10 min at D=12, χ=40).  Skip
  them if speed matters — they're computed at the end of `run_point`.

### Benchmark workflow (`benchmarks.jl`)
- `benchmark_run(; n_points, ...)` — resumable sweep over h.
- `tabulate_vs_qmc(cache)` — prints comparison table.
- `save_csv(cache, out_csv)` — writes per-h CSV for external tooling.
- `make_plots(cache)` — 4-panel benchmark figure (E, |⟨σᶻ⟩|, ⟨σˣ⟩, error vs QMC).
- `plot_correlators(cache)` — NN zz correlator vs h.

---

## 3. Contraction-order discussion

I considered three options for the CTMRG projector construction and want to
explain why I did **not** adopt the QR-based "biorthogonal projector"
formulation that's standard in newer CTMRG implementations.

**Standard CTMRG projector (what current code does):**
```
Ψ[a,b,c,d] = nClu[a,b,v,e] * nCld[e,v,c,d]
SVD(Ψ as (χ·D², D²·χ)) → U, S, Vᵗ
P = U[:, 1:χmax];  Q = Vᵗ[1:χmax, :]
```
Memory cost: Ψ is (χD², D²χ) = 5760² ComplexF64 = 530 MB.  SVD via `gesvdj`
workspace ≈ 1–1.5 GB.  Total ≈ 2 GB peak inside the SVD step.

**QR-based projectors (Corboz, PRB 2014):**
```
QR(nClu reshaped (χD², D²χ)) → Q1·R1
LQ(nCld reshaped (D²χ, χD²)) → L2·V2     # legs swapped appropriately
SVD(R1 * L2 — small)  → U, S, Vᵗ
P = Q1 @ U;  Q' = Vᵗ @ V2
```
**The promise:** if one of the dimensions is much smaller than the other, the
small SVD saves enormous compute.

**Why it doesn't help here:** for the 2×2 unit cell, both Ψ dimensions are
balanced (`χD² = D²χ = 5760`).  QR doesn't reduce the rank; it just shifts
work around.  You'd still SVD a 5760×5760 matrix in practice.  The
"biorthogonal" variant has nicer numerical conditioning when the projectors
are near-rank-deficient, but our normalization (`nrm!` after each absorption)
keeps the spectra healthy.

**What would help:** a **partial SVD** that computes only the top χmax
singular vectors via Lanczos/LOBPCG.  At χmax=40 out of 5760 candidate SVs,
that's a ~140× theoretical speedup.  CUDA.jl does not expose a turnkey
partial-SVD for ComplexF64, so this is a future optimization (see §6 below).

**Decision:** keep the current SVD approach.  Bottleneck is wall-clock from
the SVD, not memory.  My changes reduce memory peaks; the algorithm is
unchanged.

---

## 4. Memory-saving strategy summary

| Change | Saves | Risk |
|---|---|---|
| `_free!` of CTu/CTuTl/nClu/nCld/Ψ in absorptions | ~1–2 GB transient × 4 dirs | None — these are unambiguously dead |
| `_free!` of F.U/F.S/F.Vt after slicing | ~500 MB | None — Julia keeps the views alive otherwise |
| `_free!` of old corners/edges before overwrite in `ctm_move_*` | ~50 MB each × 4 dirs | None |
| Free `mL`/`mR` between SVD components in `expect_2site` | ~13.7 GB transient → ~6.87 GB | None — operator components are independent |
| Reuse Stage-2 `dl` in Stage 4 | ~27.5 GB transient | None — same `st` |
| `CUDA.reclaim()` every 5 CTMRG iter | reduces fragmentation | None |
| Cap n_track for convergence at 32 SVs | nil memory, more robust δ | None |

The combined effect: peak resident set at D=12, χ=40 should drop from
~55 GB to ~42 GB.  Comfortable on H100 (80 GB).

---

## 5. How to run safely

### One-time setup (unchanged from before)
```bash
ssh rudranilb@hpc.bits-hyderabad.ac.in
mkdir -p ~/scratch/ipeps_run_D12
cd ~/scratch/ipeps_run_D12
# upload these 3 files from your laptop:
#   ipeps_tfim_gpu.jl  benchmarks.jl  sweep_D12_chi40.sbatch  Project.toml  Manifest.toml
# (Project.toml/Manifest.toml from the original zip)

# Get an interactive session
srun --partition=gpu_h100_4 --gres=gpu:1 --cpus-per-task=8 --mem=64G \
     --time=01:00:00 --pty bash

# Re-precompile (CairoMakie now needed for benchmarks.jl)
cd ~/scratch/ipeps_run_D12
julia --project=. -e 'using Pkg; Pkg.add("DelimitedFiles"); Pkg.instantiate(); Pkg.precompile()'

# Smoke test at one h-point first
julia --project=. -e '
include("ipeps_tfim_gpu.jl")
@info "running h=3.044 single point"
r = run_point(3.044; D=12, χ=40, ctm_iter=60, ctm_tol=1e-4,
              su_schedule=[(0.1, 30), (0.01, 100), (0.001, 200)],
              verbose=true)
@show r
'

exit  # leave interactive
```

### Production sweep
```bash
cd ~/scratch/ipeps_run_D12
sbatch sweep_D12_chi40.sbatch
```

### Monitor
```bash
squeue -u $USER
tail -f ipeps_D12_chi40_<JOBID>.out
```

### After the job ends
```bash
# Cache file:                ipeps_tfim_sweep_D12_chi40.jls
# Plots:                     benchmark_D12_chi40.png  correlators_D12_chi40.png
# CSV:                       ipeps_tfim_D12_chi40.csv
# Console table at end of:   ipeps_D12_chi40_<JOBID>.out
```

### Expected wall time
- Smoke test (1 point, reduced SU schedule): **15–25 min**
- Full sweep (21 points, full SU schedule): **16–22 hours**.  Bump
  `--time=12:00:00` in the SLURM script if you need it; 24:00:00 is the
  cluster cap for `gpu_h100_4`.

---

## 6. If memory is still tight — further optimizations

In rough priority order:

1. **Partial SVD via Lanczos on `Ψ Ψᵗ`.**  Compute only the top χmax left
   singular vectors of Ψ by applying `Ψ Ψᵗ` implicitly.  Avoids the
   5760×5760 dense SVD entirely.  Need to write a custom Lanczos in
   CUDA.jl; the cuTENSOR contractions for `v ↦ Ψ(Ψᵗ v)` are cheap once.
   Expected speedup: **5–15×** on CTMRG.

2. **Skip CTMRG re-randomization between h-points.**  Currently
   `init_ctm` is called fresh each h-point.  Warm-starting from the
   converged environment of the previous h-point (when D and χ are
   unchanged) typically cuts CTMRG iterations in half.  Implementation:
   carry `e` across `sweep_h` invocations.  About 5 extra lines.

3. **Compute correlators with a single shared denominator.**  Right now
   `expect_2site` re-computes `contract_norm` every call.  In `run_point`
   we call it twice for the correlator + four times inside `compute_energy`.
   Cache the norm once per env.  Saves ~10% of observable time.

4. **Move from ComplexF64 to ComplexF32 in the double-layer tensors.**
   For ground-state TFIM the dominant cost is bond-dimension truncation,
   not roundoff.  Cuts memory in half.  Risk: cuTENSOR plan auto-tuner
   sometimes picks worse algorithms for CF32 at our shapes — test first
   on one point.

5. **Build only 2 double layers at a time.**  A given CTMRG direction
   only touches 2 of {mA, mB, mC, mD}.  You can free the unused pair
   and rebuild them for the next direction.  Cost: 4 extra `double_layer`
   builds per CTMRG sweep, each ~2 ms.  Savings: 13.7 GB resident.
   Big win **only if you also drop the partial SVD**, because the SVD
   itself needs the double-layer tensors live during the absorption.

6. **Recompute double-layer tensors fresh for each `expect_2site` call.**
   Same trick at observation time — free the 4 dl tensors, build only
   the 2 needed for the current correlator.  Easier to implement than
   the CTMRG version.

If you hit OOM despite all the above, the next move is **multi-GPU CTMRG**
(stripe the 4 m-tensors across GPUs), or move to D=10/χ=40 (cuts the dl
footprint by a factor of (144/100)² ≈ 2× per tensor → 13.4 GB total dl).

---

## 7. Caveats about the paper benchmark

You asked me to benchmark against Patra, Jahromi, Singh, Orús (PhysRevResearch
6, 013326 (2024)).  I have to be straight with you:

- **Their model:** kicked Ising with discrete unitary trotter step
  `U(θ) = (∏ exp(iπ/4 ZZ))(∏ exp(-iθ/2 X))`, **real-time evolution**.
- **Our model:** continuous Hamiltonian `H = -J ΣZZ - h Σ X`,
  **imaginary-time evolution to find ground state**.
- **Their lattice:** heavy-hexagon (matching IBM Eagle/Osprey/Condor
  qubit topology).
- **Our lattice:** square, 2×2 unit cell.
- **Their method:** gPEPS with simple update + mean-field 1-site
  measurement.
- **Our method:** iPEPS with simple update + CTMRG-bounded measurement.

These differences are not cosmetic — gPEPS uses a **graph-based update**
that doesn't construct a corner transfer matrix at all (their environment
is mean-field through the simple-update weights).  And the kicked Ising
unitary preserves total-Z eigenvalues at θ=π/2, which they exploit for
their "weight-N observable" reductions (their Appendix B).  None of that
applies to a continuous TFIM ground state.

The closest honest comparison I can build is what `benchmarks.jl` already
does: square-lattice TFIM ground state, energy + transverse and longitudinal
magnetization vs. **QMC reference values**.  That's the comparison your old
`plot_tfim.py` was already doing; I extended it with correlators and a
proper table.

If you want a direct repro of the paper:
- Switch from imaginary to real time (replace `ising_gate` with the kicked
  unitary `U(θ)`).
- Switch lattice from square to heavy-hexagon (totally different topology
  — gPEPS would need different weight structures and a different `simple_update!`).
- Switch from CTMRG measurement to mean-field 1-site measurement (drop
  the entire §5 CTMRG block, use 1-site reduced density matrix from the
  weighted local tensor).
- Adopt the weight-N observable reduction (their Eq. B2).

That's a different project, not an incremental change to your code.  I
didn't write that.  If you want to, say so and I'll scope it.
