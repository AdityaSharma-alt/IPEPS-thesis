# =============================================================================
# fu_sweep.jl — full 22-point FU sweep across the TFIM phase diagram.
#
# Mirrors the structure of benchmarks.jl (which does SU-only sweeps for the
# big-D production runs).  This script does SU warmup + FU on every h point,
# saves results to JLS + CSV, and is resume-safe via a per-point cache.
#
# Settings (chosen as a compromise between accuracy and ~24h budget):
#   D = 4, χ = 20
#   n_fu_sweeps = 60
#   dτ_fu = 0.005    (total τ_FU = 0.30)
#   22 h points from 1.5 to 4.5 (denser near h_c)
#
# Estimated time per point: 5-8 minutes → ~2-3 hours total for 22 points.
# =============================================================================

include(joinpath(@__DIR__, "..", "src", "ipeps_fu.jl"))
using Serialization, Printf, DelimitedFiles

# -- h points (22 total, denser near h_c) --
h_points = sort(vcat(
    range(1.5, 2.8, length=6),                # broad sweep below
    [2.9, 2.95, 3.0, 3.02, 3.044, 3.07, 3.1], # dense around h_c
    range(3.2, 4.5, length=9),                # broad sweep above
))

cache_path = joinpath(@__DIR__, "..", "analysis", "data", "fu_sweep_D4_chi20.jls")
csv_path   = joinpath(@__DIR__, "..", "analysis", "data", "fu_sweep_D4_chi20.csv")

# Load existing cache if present
results = if isfile(cache_path)
    println("Loading existing cache from $cache_path")
    deserialize(cache_path)
else
    Dict{Float64, NamedTuple}()
end

println("\n" * "="^70)
println(" FU SWEEP  D=4 χ=20  $(length(h_points)) points  dτ=0.005  60 sweeps ")
println("="^70)
println("Already done: $(length(results)) / $(length(h_points))")

t_start = time()
for (i, h) in enumerate(h_points)
    h_key = round(h, digits=4)
    if haskey(results, h_key)
        @printf("[%2d/%2d] h=%.4f  SKIP (cached)\n", i, length(h_points), h_key)
        continue
    end

    @printf("\n[%2d/%2d] h=%.4f  starting  (elapsed: %.0fs)\n",
            i, length(h_points), h_key, time() - t_start)

    try
        r = run_point_fu_full(h_key;
                              D=4, chi=20,
                              n_fu_sweeps=60,
                              dtau_fu=0.005,
                              ctm_iter=60, ctm_tol=1e-4,
                              accept_dE=2e-3,
                              verbose=false)
        results[h_key] = r
        # Save cache after every point
        serialize(cache_path, results)
        @printf("  → cached.  E_SU=%.6f → E_FU=%.6f  ΔE=%+.4e\n",
                r.E_init, r.E, r.E - r.E_init)
    catch err
        @warn "h=$h_key FAILED" exception=(err, catch_backtrace())
    end
    CUDA.reclaim()
end

# -- Dump to CSV --
println("\n" * "="^70)
println(" Writing CSV: $csv_path ")
println("="^70)
sorted_h = sort(collect(keys(results)))
rows = []
push!(rows, ["h", "E_SU", "E_FU", "dE", "mz_SU", "mz_FU", "mx_SU", "mx_FU",
             "Sc_SU", "Sc_FU", "Sb_SU", "Sb_FU"])
for h in sorted_h
    r = results[h]
    push!(rows, [
        h,
        r.E_init, r.E, r.E - r.E_init,
        abs(r.mz_init), abs(r.mz),
        r.mx_init, r.mx,
        r.S_init, r.S,
        r.Sb_init, r.Sb,
    ])
end
open(csv_path, "w") do f
    for row in rows
        println(f, join(row, ","))
    end
end

# -- Print compact summary table --
println("\n" * "="^90)
println(" FU SWEEP SUMMARY ")
println("="^90)
@printf("%6s | %10s %10s %10s | %7s %7s | %7s %7s | %7s\n",
        "h", "E_SU", "E_FU", "ΔE", "|mz|_SU", "|mz|_FU", "S_b_SU", "S_b_FU", "ΔS_b")
println("-"^90)
for h in sorted_h
    r = results[h]
    @printf("%6.4f | %+.6f %+.6f %+.4e | %.5f %.5f | %.4f %.4f | %+.4f\n",
            h, r.E_init, r.E, r.E - r.E_init,
            abs(r.mz_init), abs(r.mz), r.Sb_init, r.Sb, r.Sb - r.Sb_init)
end

@printf("\nDone.  Total time: %.0fs = %.2fh\n",
        time() - t_start, (time() - t_start)/3600)
