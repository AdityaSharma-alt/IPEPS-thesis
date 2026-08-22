# =============================================================================
# fu_validate.jl — single-point FU at h_c with 80 sweeps to confirm:
#   1. ΔE grows monotonically (more sweeps → bigger improvement)
#   2. ALS converges sensibly each sweep
#   3. Magnetizations stay stable, don't blow up
#   4. Boundary entropy behaves sensibly (large near h_c, no negative-noise junk)
#
# Run BEFORE committing to the 22-point production sweep.
# ~10 minutes on H200.
# =============================================================================

include(joinpath(@__DIR__, "..", "src", "ipeps_fu.jl"))
using Printf, Serialization

println("\n" * "="^70)
println(" FU validation  —  h=3.044 (h_c)  D=4  χ=20  80 sweeps  dτ=0.005 ")
println("="^70)

r = run_point_fu_full(3.044;
                      D=4, chi=20,
                      n_fu_sweeps=80,
                      dtau_fu=0.005,
                      ctm_iter=60, ctm_tol=1e-4,
                      accept_dE=2e-3,
                      verbose=false)

# Compact summary table
println("\n" * "="^70)
println(" PER-SWEEP TRAJECTORY ")
println("="^70)
@printf("%6s  %12s  %10s  %8s  %8s  %8s  %8s\n",
        "sweep", "E", "ΔE_vs_SU", "|mz|", "mx", "S_c", "S_b")
println("-"^70)
for hist in r.history
    ΔE = hist.E - r.E_init
    @printf("%6d  %+.8f  %+.4e  %.5f  %.5f  %.4f  %.4f\n",
            hist.sweep, hist.E, ΔE, abs(hist.mz), hist.mx, hist.S, hist.Sb)
end

println("\n" * "="^70)
println(" SUMMARY ")
println("="^70)
@printf("  SU baseline:   E=%.6f  |mz|=%.4f  mx=%.4f  S_b=%.4f\n",
        r.E_init, abs(r.mz_init), r.mx_init, r.Sb_init)
@printf("  After FU(80):  E=%.6f  |mz|=%.4f  mx=%.4f  S_b=%.4f\n",
        r.E, abs(r.mz), r.mx, r.Sb)
@printf("  ΔE total: %+.6f  (%.4f%% improvement)\n",
        r.E - r.E_init, 100*(r.E - r.E_init)/abs(r.E_init))
@printf("  ΔS_b    : %+.4f  (FU should INCREASE boundary entropy near h_c)\n",
        r.Sb - r.Sb_init)

serialize("fu_validate_result.jls", r)
println("\nSaved to fu_validate_result.jls")

# Decision hint:
if r.E < r.E_init - 1e-3
    println("\n✓ Validation looks good: ΔE < -1e-3.  Safe to proceed to 22-point sweep.")
else
    println("\n⚠ ΔE = $(round(r.E - r.E_init, sigdigits=3))  is small.")
    println("  Either FU is genuinely marginal at these settings,")
    println("  or there's a bug.  Check the trajectory above for monotonicity.")
end
