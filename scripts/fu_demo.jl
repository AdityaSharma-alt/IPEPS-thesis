# =============================================================================
# fu_demo.jl  —  Smoke test + small benchmark for the all-bonds Full Update.
#
# Two stages:
#   STAGE 1: tiny D=2, χ=4 smoke test that exercises EVERY bond environment
#            once and checks for NaN/Inf in N.  This is the cheap way to catch
#            leg-convention bugs in fu_env_{AB,CD,AC,BD} BEFORE running at
#            real parameters.  Takes ~1 minute.
#   STAGE 2: SU-only vs SU+FU comparison at D=4, χ=20 on TFIM h sweep around
#            the critical point.  Takes ~5-10 minutes.
#
# Usage:
#   julia --project=. fu_demo.jl           # both stages
#   julia --project=. fu_demo.jl smoke     # smoke test only
#   julia --project=. fu_demo.jl bench     # benchmark only (skips smoke)
# =============================================================================

include(joinpath(@__DIR__, "..", "src", "ipeps_fu.jl"))

using Printf, Serialization

mode = length(ARGS) >= 1 ? ARGS[1] : "all"

# =============================================================================
# STAGE 1 — Smoke test: build env for each bond, check for NaN/Inf, verify N
# has hermitian structure under the (1,2,3,4)↔(2,1,4,3) swap.
# =============================================================================

function smoke_test()
    println("\n" * "="^70)
    println(" SMOKE TEST  —  D=2, χ=4, random state, exercise all 4 bond envs ")
    println("="^70)

    D = 2; chi = 4
    st = init_ipeps(2, D; seed=1)
    dl = build_all_dl(st)
    e  = init_ctm(chi, D^2)

    # A few CTMRG iterations to make the env semi-physical (not random noise)
    ctmrg_converge!(e, dl, chi; max_iter=20, tol=1e-3, verbose=false)
    println("CTMRG converged.")
    @printf("GPU after CTMRG: %s\n\n", gpumem_str())

    bond_specs = [
        ("AB", :A, :B, split_right, split_left, double_Q_right, double_Q_left, fu_env_AB),
        ("CD", :C, :D, split_right, split_left, double_Q_right, double_Q_left, fu_env_CD),
        ("AC", :A, :C, split_down,  split_up,   double_Q_down,  double_Q_up,   fu_env_AC),
        ("BD", :B, :D, split_down,  split_up,   double_Q_down,  double_Q_up,   fu_env_BD),
    ]

    all_ok = true
    for (name, sL, sR, splL, splR, QL_fn, QR_fn, env_fn) in bond_specs
        AL = getfield(st, sL); AR = getfield(st, sR)
        Q_L, R_L = splL(AL)
        Q_R, R_R = splR(AR)
        QQ_L = QL_fn(Q_L); QQ_R = QR_fn(Q_R)
        N    = env_fn(e, dl, QQ_L, QQ_R)
        N_cpu = Array(N)

        has_nan = any(isnan, N_cpu)
        has_inf = any(isinf, N_cpu)

        # Check hermiticity:  N[a,a',b,b'] should equal conj(N[a',a,b',b])
        N_sym = permutedims(conj.(N_cpu), (2,1,4,3))
        herm_err = norm(N_cpu - N_sym) / max(norm(N_cpu), 1e-30)

        # Reshape N to a (k_L*k_R) × (k_L*k_R) matrix and check eigenvalues
        k_L = size(N_cpu, 1); k_R = size(N_cpu, 3)
        N_mat = reshape(permutedims(N_cpu, (1,3, 2,4)), k_L*k_R, k_L*k_R)
        N_mat_h = (N_mat + N_mat') / 2
        evals = real.(LinearAlgebra.eigvals(N_mat_h))
        min_ev = minimum(evals); max_ev = maximum(evals)
        cond_num = abs(max_ev) / max(abs(min_ev), 1e-30)
        positive = all(>=(-1e-8 * abs(max_ev)), evals)  # allow small negative noise

        status = (has_nan || has_inf || !positive) ? "FAIL" : "OK"
        all_ok &= (status == "OK")
        @printf("  bond %s  size(N)=(%d,%d,%d,%d)  ", name, size(N_cpu)...)
        @printf("|nan|=%d  |inf|=%d  herm_err=%.1e  λ_min=%+.3e  λ_max=%.3e  cond=%.1e  [%s]\n",
                Int(has_nan), Int(has_inf), herm_err, min_ev, max_ev, cond_num, status)
    end

    println()
    if all_ok
        println("✓ all bond envs PASS basic checks (no NaN/Inf, positive-semidefinite)")
    else
        println("✗ at least one bond env FAILED — fix before running benchmark!")
    end
    return all_ok
end

# =============================================================================
# STAGE 2 — SU-only vs SU+FU benchmark at D=4, χ=20
# =============================================================================

function bench()
    println("\n" * "="^70)
    println(" BENCHMARK  —  D=4, χ=20, SU-only vs SU+FU (all 4 bonds) ")
    println("="^70)

    h_points = [2.0, 3.0, 3.044, 3.5]   # h_c ≈ 3.044, so test below/at/above

    results = Dict{Float64, Any}()
    for h in h_points
        @printf("\n>>> h = %.4f  (h_c ≈ 3.044) <<<\n", h)
        r = run_point_fu_full(h;
                              D=4, chi=20,
                              n_fu_sweeps=8,
                              dtau_fu=0.01,
                              ctm_iter=60, ctm_tol=1e-4,
                              accept_dE=1e-3,
                              verbose=false)
        results[h] = r
        CUDA.reclaim()
    end

    # ---- Summary table ----
    println("\n" * "="^80)
    println(" SUMMARY  —  D=4, χ=20 ")
    println("="^80)
    @printf("%-7s | %-26s | %-26s\n",
            "h", "POST-SU", "POST-FU (all bonds)")
    @printf("%-7s | %-26s | %-26s\n",
            "", "E       |mz|   mx     S", "E       |mz|   mx     S")
    println("-"^80)
    for h in h_points
        r = results[h]
        @printf("%6.4f | %+.4f  %.4f %.4f %.4f | %+.4f  %.4f %.4f %.4f\n",
                h,
                r.E_init, abs(r.mz_init), r.mx_init, r.S_init,
                r.E,      abs(r.mz),      r.mx,      r.S)
    end

    println("\n" * "="^80)
    println(" PER-SWEEP HISTORY  (E, S) ")
    println("="^80)
    for h in h_points
        r = results[h]
        @printf("\nh = %.4f:\n", h)
        @printf("  sweep |    E         |mz|    mx       S        ΔS_vs_SU\n")
        for hist in r.history
            ΔS = hist.S - r.S_init
            @printf("  %5d | %+.6f  %.4f  %.4f  %.4f  %+.4f\n",
                    hist.sweep, hist.E, abs(hist.mz), hist.mx, hist.S, ΔS)
        end
    end

    serialize("fu_demo_results.jls", results)
    println("\nSaved to fu_demo_results.jls")
    return results
end

# =============================================================================
# Main
# =============================================================================

if mode == "smoke"
    smoke_test()
elseif mode == "bench"
    bench()
else
    ok = smoke_test()
    if !ok
        println("\n⚠ Smoke test failed; skipping benchmark.  Fix env code first.")
        exit(1)
    end
    bench()
end
