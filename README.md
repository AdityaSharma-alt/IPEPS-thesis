# iPEPS for the 2D Transverse-Field Ising Model

A minimal, research-grade Julia implementation of **infinite Projected Entangled Pair States (iPEPS)** for the 2D transverse-field Ising model on the square lattice.

$$H = -J \sum_{\langle ij \rangle} \sigma^z_i \sigma^z_j \;-\; h \sum_i \sigma^x_i$$

The quantum critical point is at **h_c / J ≈ 3.044** (Blöte & Deng 2002).

-----

## What this code does

It finds the ground state of the infinite 2D TFIM by variationally optimizing an iPEPS wave function, then measures the ground-state energy, magnetizations, and entanglement entropy. The full pipeline runs identically for bond dimensions D = 2, 3, 4 with parameters automatically scaled to fit in memory:

```
Simple Update  →  CTMRG  →  Full Update  →  CTMRG  →  Observables
```

-----

## Features

|Feature      |Detail                                                                   |
|-------------|-------------------------------------------------------------------------|
|Unit cell    |2×2 (tensors A, B, C, D) with period-2 translational invariance          |
|CTMRG        |4 explicit directional moves — LEFT, UP, RIGHT, DOWN — no rotation tricks|
|Simple Update|Imaginary-time TEBD with diagonal bond weights on all 4 bond types       |
|Full Update  |Exact variational compression via **N · A′ = W** solved with ALS         |
|All 4 bonds  |AB, CD (horizontal) and AC, BD (vertical) each fully updated             |
|Memory safety|`auto_params(D)` tunes χ and rank cap so FU never runs out of memory     |
|Validation   |Bundled QMC reference energies at h = 2.0, 2.5, 3.0, 3.044, 3.5, 4.0     |

-----

## Requirements

Julia 1.9+ with one external package:

```julia
using Pkg
Pkg.add("TensorOperations")
```

`LinearAlgebra` and `Printf` are Julia standard library.

-----

## Quick start

```julia
include("ipeps_tfim.jl")

# Single run: D=2, near the critical point, one Full Update sweep
r = run_ipeps_2x2(D=2, h=3.0, J=1.0, n_fu=1)

println("E/site = ", r.E)   # compare to QMC: -3.2802
println("⟨σˣ⟩   = ", r.mx)
println("S_ent  = ", r.S)
```

```julia
# Convergence study over D = 2, 3, 4 (energy must decrease with D)
convergence_study(h=3.0, n_fu=1)
```

```julia
# Check what parameters will be used before running
print_params()
```

```
Auto-tuned parameters:
  D    χ    r_max  nals   peak_mem
  -----------------------------------
  2    40   4      2      1.0e+07
  3    40   6      2      3.1e+07
  4    30   8      2      7.4e+07
```

-----

## Physics

### iPEPS

An iPEPS tiles the infinite 2D lattice with a periodic unit cell. Each site carries a rank-5 tensor with one physical leg (dimension `d = 2` for spin-1/2) and four virtual legs (dimension `D`) connecting to neighbours:

```
  A ─── B ─── A ─── B
  │      │     │     │
  C ─── D ─── C ─── D
  │      │     │     │
  A ─── B ─── A ─── B
```

Tensor leg order: `[phys, up, right, down, left]`. Increasing `D` increases the expressiveness of the ansatz and variational energy must decrease monotonically.

### CTM environment

The environment tensors — 4 corners (χ × χ) and 8 edges (χ × D² × χ) — represent the infinite lattice outside the unit cell:

```
  Clu ── Tu1 ── Tu2 ── Cru
   │      │      │      │
  Tl1 ──  A  ──  B  ── Tr1
   │      │      │      │
  Tl2 ──  C  ──  D  ── Tr2
   │      │      │      │
  Cld ── Td1 ── Td2 ── Crd
```

The CTMRG algorithm converges these tensors by iteratively absorbing rows and columns of the lattice and truncating back to bond dimension χ using SVD projectors.

### Simple Update

Applies imaginary-time gates `exp(−dτ H_bond)` one bond at a time. The environment is approximated by diagonal weight matrices λ on each virtual bond, making this fast and memory-light. Used as a warm-start before the Full Update.

### Full Update — N · A′ = W

After applying the gate to produce the full evolved tensor Θ (never truncated), the compressed tensors A′ and B′ are found by minimizing the compression error:

```
∂K/∂A′* = 0   ⟹   N · vec(A′) = vec(W)
```

- **N** is the environment norm matrix with site A removed (hole construction via `build_env_A`)
- **W** is the overlap with Θ from the full CTM environment (via `build_W_A_h`)
- Solved as `(N + 1e-8 I) \ W` with damping `A′ ← 0.5 A′_new + 0.5 A′_old`
- ALS alternates between optimizing A′ and B′ with a CTMRG refresh between each

-----

## Code map

|Section|What it contains                                                                                      |
|-------|------------------------------------------------------------------------------------------------------|
|§1     |Model constants, data structures (`iPEPS`, `Weights`, `CTMEnv`, `DoubleLayers`)                       |
|§2     |Initialisation and normalization helpers                                                              |
|§3     |`auto_params(D)` — adaptive χ, r_max, nals selection                                                  |
|§4     |`double_layer`, `double_layer_op`, `build_all_dl`                                                     |
|§5     |Low-level CTM absorbers: `_absorb_left_col`, `_absorb_right_col`, `_absorb_top_row`, `_absorb_bot_row`|
|§6     |`ctm_move_left/right/up/down!`, `ctmrg_sweep!`, `ctmrg_converge!`                                     |
|§7     |`contract_2x2`, `expect_1site`, `expect_2site`, `compute_energy`, `entanglement_entropy`              |
|§8     |`simple_update!` with correct weight handling for all four bond types                                 |
|§9     |Hole-construction environment builders: `build_env_A/B/C/D`, `build_N_mat`                            |
|§10    |Memory-safe overlap builders: `build_W_A_h`, `build_W_B_h` with rank-capped SVD                       |
|§11    |`_als_solve`, `full_update_AB/CD/AC/BD!`                                                              |
|§12    |`full_update_sweep!` over all four bonds                                                              |
|§13    |`run_ipeps_2x2` — main driver                                                                         |
|§14    |`convergence_study`                                                                                   |

-----

## Expected output

Running `convergence_study(h=3.0, n_fu=1)` should produce something like:

```
D   χ    r_max  method  E/site           ⟨σᶻ⟩      ⟨σˣ⟩      S         sec
──────────────────────────────────────────────────────────────────────────
2   40   4      SU+FU   -3.27XXXXXX     0.XXXXXX  0.XXXXXX  X.XXXXXX  XX
3   40   6      SU+FU   -3.28XXXXXX     ...
4   30   8      SU+FU   -3.29XXXXXX     ...

QMC reference (h=3.0): E/site = -3.2802
✓ Energy is monotonically decreasing ✓
```

Energy decreases with D toward the QMC reference value, confirming the variational principle is being respected. At h ≈ 3.044 (the critical point) the entanglement entropy S peaks and convergence with D is slowest due to the diverging correlation length.

-----

## Tuning

All parameters except `h`, `J`, and `n_fu` are set automatically. To override:

```julia
# More SU steps for a better starting point
r = run_ipeps_2x2(D=3, h=3.044,
      su_schedule=[(0.1,100),(0.01,200),(0.001,300)],
      n_fu=2, damp=0.5)

# Looser CTMRG inside ALS (faster, less accurate)
# Edit the max_iter/tol values in full_update_AB! directly
```

The `damp = 0.5` parameter is important for stability — do not set it to 1.0 (no damping) unless the ALS is already close to convergence.

-----

## References

1. Verstraete & Cirac (2004) — original iPEPS proposal
2. Orús & Vidal, PRB **80**, 094403 (2009) — CTM for iPEPS
3. Corboz, SciPost Phys. Lect. Notes **7** (2019) — full iPEPS review; this code follows this reference most closely
4. Phien, Vidal & McCulloch, PRB **92**, 035142 (2015) — Full Update algorithm
5. Blöte & Deng, PRE **66**, 066110 (2002) — 2D Ising quantum critical point h_c ≈ 3.044

-----

## License

MIT
# ipeps
