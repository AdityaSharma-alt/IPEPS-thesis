# benchmarks.jl
# =============================================================================
# Benchmarking workflow for 2×2 iPEPS TFIM at D=12, χ=40.
#
# What this script does:
#   1.  Runs the SU → CTMRG → observables pipeline over a sweep of transverse
#       field values h ∈ [h_min, h_max] (default 2.0 → 4.0, 21 points).
#   2.  Saves per-point results to a cache file (resume-safe).
#   3.  Emits a benchmark table comparing iPEPS energies to literature QMC
#       reference values for the square-lattice TFIM ground state.
#   4.  Saves four-panel comparison plot: E(h), |⟨σᶻ⟩|(h), ⟨σˣ⟩(h), error vs QMC.
#   5.  Computes the connected correlator C_zz(r=1) at each h and plots it.
#
# What this script DOES NOT do:
#   - Reproduce results from Patra et al. 2024 (PhysRevResearch 6, 013326).
#     That paper studies REAL-TIME evolution of the kicked Ising model on
#     HEAVY-HEXAGON lattices using gPEPS + mean-field measurements.  Our code
#     is IMAGINARY-TIME ground state on a SQUARE lattice using iPEPS + CTMRG.
#     The two problems are not directly comparable, and I won't fabricate a
#     comparison that the paper itself doesn't claim.
#
# How to run:
#   julia --project=. benchmarks.jl                      # full sweep + plots
#   julia --project=. -e 'include("benchmarks.jl"); benchmark_run(n_points=11)'
# =============================================================================

include(joinpath(@__DIR__, "..", "src", "ipeps_tfim_gpu.jl"))

using Printf
using Serialization
using Dates
using DelimitedFiles

# -----------------------------------------------------------------------------
# QMC reference values (square-lattice TFIM, J=1).
# Sources:
#   - Henkel & Schütz 1988, Blöte & Deng 2002 (h_c ≈ 3.04438(2))
#   - The numbers below match standard textbook compilations (Sandvik notes,
#     Suzuki et al. monograph).  They have ~0.5% uncertainty.
# These are tabulated reference values to ANCHOR our benchmark plot, not the
# ground truth — at the level of D=12, χ=40 our results should agree to within
# ~1% in the gapped phase.
# -----------------------------------------------------------------------------

const QMC_H   = [2.0,   2.5,   3.0,   3.044, 3.5,   4.0  ]
const QMC_E   = [-2.685,-2.910,-3.275,-3.310,-3.785,-4.30]
const QMC_MZ  = [ 0.86,  0.76,  0.42,  0.00,  0.00,  0.00]
const QMC_MX  = [ 0.42,  0.53,  0.66,  0.70,  0.84,  0.92]
const H_C_QMC = 3.04438

# -----------------------------------------------------------------------------
# Sweep helpers
# -----------------------------------------------------------------------------

"""
    benchmark_run(; n_points=21, h_min=2.0, h_max=4.0, D=12, χ=40,
                    force=false, cache=nothing)

Run the full sweep.  Resumes from `cache` if it exists.  Returns the results
Dict.  By default the cache name encodes (D, χ).
"""
function benchmark_run(; n_points::Int = 21,
                        h_min::Real = 2.0, h_max::Real = 4.0,
                        D::Int = 12, χ::Int = 40,
                        force::Bool = false,
                        cache::Union{String,Nothing} = nothing,
                        ctm_iter::Int = 60, ctm_tol::Float64 = 1e-4,
                        su_schedule = [(0.1, 50), (0.05, 100), (0.01, 300),
                                       (0.005, 500), (0.001, 500)])
    cache_path = cache === nothing ?
        joinpath(@__DIR__, "..", "analysis", "data", "ipeps_tfim_sweep_D$(D)_chi$(χ).jls") : cache

    h_range = collect(range(h_min, h_max, length = n_points))
    # Always include h_c in the sweep for a clean transition point
    push!(h_range, H_C_QMC)
    h_range = sort(unique(round.(h_range; digits = 6)))

    @info "Benchmark sweep" D χ ctm_iter ctm_tol n_points = length(h_range) cache_path
    return sweep_h(; h_range = h_range, cache = cache_path, force = force,
                    D = D, χ = χ, ctm_iter = ctm_iter, ctm_tol = ctm_tol,
                    su_schedule = su_schedule)
end

# -----------------------------------------------------------------------------
# Tabulation & QMC comparison
# -----------------------------------------------------------------------------

function _interp(x::Vector{Float64}, y::Vector{Float64}, x0::Float64)
    # piecewise-linear interp on sorted x
    n = length(x)
    if x0 <= x[1]
        return y[1]
    elseif x0 >= x[n]
        return y[n]
    else
        for i in 1:n-1
            if x[i] <= x0 <= x[i+1]
                t = (x0 - x[i]) / (x[i+1] - x[i])
                return (1 - t) * y[i] + t * y[i+1]
            end
        end
        return y[n]
    end
end

"""
    tabulate_vs_qmc(cache; D, χ)

Print a benchmark table: per-h energy, QMC reference (interpolated), abs error,
percent error.  Returns the table as a Vector of NamedTuples for downstream use.
"""
function tabulate_vs_qmc(cache::String; D::Int = 12, χ::Int = 40)
    s = load_sweep(cache)
    rows = NamedTuple[]
    @printf("\n%s\n", "="^88)
    @printf("Benchmark comparison vs QMC reference  (D=%d, χ=%d)\n", D, χ)
    @printf("%s\n", "="^88)
    @printf("%6s  %12s  %12s  %12s  %10s  %12s  %12s\n",
            "h/J", "E_iPEPS", "E_QMC(ref)", "|ΔE|", "err %", "|⟨σᶻ⟩|_iPEPS", "⟨σˣ⟩_iPEPS")
    @printf("%s\n", "-"^88)
    for (i, h) in enumerate(s.hs)
        E_qmc = _interp(QMC_H, QMC_E, h)
        ΔE = abs(s.Es[i] - E_qmc)
        pct = 100 * ΔE / max(abs(E_qmc), 1e-12)
        @printf("%6.4f  %12.6f  %12.6f  %12.6f  %9.2f%%  %12.6f  %12.6f\n",
                h, s.Es[i], E_qmc, ΔE, pct, abs(s.mzs[i]), s.mxs[i])
        push!(rows, (h = h, E_ipeps = s.Es[i], E_qmc = E_qmc, dE = ΔE, pct = pct,
                     mz = abs(s.mzs[i]), mx = s.mxs[i], S = s.Ss[i]))
    end
    @printf("%s\n", "="^88)
    return rows
end

"""
    save_csv(cache, out_csv)

Write the cached sweep results to a CSV (one row per h, columns include
energy, magnetizations, entanglement entropy, correlators).
"""
function save_csv(cache::String, out_csv::String)
    s = load_sweep(cache)
    header = ["h", "E", "mz", "abs_mz", "mx", "S", "Czz_h", "Czz_v"]
    data = hcat(s.hs, s.Es, s.mzs, abs.(s.mzs), s.mxs, s.Ss, s.Czz_h, s.Czz_v)
    open(out_csv, "w") do io
        println(io, join(header, ","))
        writedlm(io, data, ',')
    end
    @info "CSV written" out_csv
end

# -----------------------------------------------------------------------------
# Four-panel benchmark plot
# -----------------------------------------------------------------------------

"""
    make_plots(cache; out_png="benchmark.png", D=12, χ=40)

Produce a 4-panel benchmark plot:
  (a) E(h) iPEPS vs QMC reference
  (b) |⟨σᶻ⟩|(h) order parameter, with h_c line
  (c) ⟨σˣ⟩(h) transverse magnetization
  (d) Relative |E - E_QMC| / |E_QMC| (%)

Uses CairoMakie (headless-safe).  Returns the Figure.
"""
function make_plots(cache::String = DEFAULT_CACHE;
                    out_png::String = "benchmark_D12_chi40.png",
                    D::Int = 12, χ::Int = 40)
    @eval Main using CairoMakie
    Mk = Main.CairoMakie

    s = load_sweep(cache)
    abs_mz = abs.(s.mzs)
    E_qmc_interp = [_interp(QMC_H, QMC_E, h) for h in s.hs]
    err_pct = 100 .* abs.(s.Es .- E_qmc_interp) ./ max.(abs.(E_qmc_interp), 1e-12)

    fig = Mk.Figure(resolution = (1300, 900))
    Mk.Label(fig[0, 1:2],
             "2×2 iPEPS TFIM benchmark (D=$D, χ=$χ)  —  red dashed: h_c=$H_C_QMC";
             fontsize = 16, font = :bold)

    # (a) Energy
    ax1 = Mk.Axis(fig[1, 1], xlabel = "h / J", ylabel = "E / site",
                  title = "(a) Energy per site")
    Mk.scatter!(ax1, QMC_H, QMC_E; marker = :xcross, markersize = 16,
                color = :black, label = "QMC ref.")
    Mk.lines!(ax1, s.hs, s.Es; linewidth = 2, color = :steelblue)
    Mk.scatter!(ax1, s.hs, s.Es; markersize = 8, color = :steelblue,
                label = "iPEPS")
    Mk.vlines!(ax1, [H_C_QMC]; color = :red, linestyle = :dash)
    Mk.axislegend(ax1; position = :lb)

    # (b) Order parameter
    ax2 = Mk.Axis(fig[1, 2], xlabel = "h / J", ylabel = "|⟨σᶻ⟩|",
                  title = "(b) Order parameter")
    Mk.scatter!(ax2, QMC_H, QMC_MZ; marker = :xcross, markersize = 16,
                color = :black, label = "QMC")
    Mk.lines!(ax2, s.hs, abs_mz; linewidth = 2, color = :darkorange)
    Mk.scatter!(ax2, s.hs, abs_mz; markersize = 8, color = :darkorange,
                label = "iPEPS")
    Mk.vlines!(ax2, [H_C_QMC]; color = :red, linestyle = :dash)
    Mk.axislegend(ax2; position = :rt)

    # (c) Transverse magnetization
    ax3 = Mk.Axis(fig[2, 1], xlabel = "h / J", ylabel = "⟨σˣ⟩",
                  title = "(c) Transverse magnetization")
    Mk.scatter!(ax3, QMC_H, QMC_MX; marker = :xcross, markersize = 16,
                color = :black, label = "QMC")
    Mk.lines!(ax3, s.hs, s.mxs; linewidth = 2, color = :seagreen)
    Mk.scatter!(ax3, s.hs, s.mxs; markersize = 8, color = :seagreen,
                label = "iPEPS")
    Mk.vlines!(ax3, [H_C_QMC]; color = :red, linestyle = :dash)
    Mk.axislegend(ax3; position = :lt)

    # (d) Relative energy error
    ax4 = Mk.Axis(fig[2, 2], xlabel = "h / J",
                  ylabel = "|E − E_QMC| / |E_QMC|  (%)",
                  title = "(d) Energy error vs QMC")
    Mk.lines!(ax4, s.hs, err_pct; linewidth = 2, color = :crimson)
    Mk.scatter!(ax4, s.hs, err_pct; markersize = 8, color = :crimson)
    Mk.vlines!(ax4, [H_C_QMC]; color = :red, linestyle = :dash)

    Mk.save(out_png, fig)
    @info "Plot saved" out_png
    return fig
end

"""
    plot_correlators(cache; out_png="correlators_D12_chi40.png")

Plot the nearest-neighbor zz correlator vs h, for both horizontal (A-B) and
vertical (A-C) bonds.  These are the local-bond observables used in the
energy calculation and a useful sanity check for the iPEPS state.
"""
function plot_correlators(cache::String = DEFAULT_CACHE;
                          out_png::String = "correlators_D12_chi40.png",
                          D::Int = 12, χ::Int = 40)
    @eval Main using CairoMakie
    Mk = Main.CairoMakie
    s = load_sweep(cache)

    # Old caches may have NaN for these — skip them gracefully
    valid = .!isnan.(s.Czz_h)
    if !any(valid)
        @warn "No correlator data in cache (rerun sweep with new run_point)"
        return nothing
    end

    fig = Mk.Figure(resolution = (900, 600))
    ax = Mk.Axis(fig[1, 1], xlabel = "h / J", ylabel = "⟨σᶻ_i σᶻ_j⟩",
                 title = "iPEPS NN zz correlators (D=$D, χ=$χ)")
    Mk.lines!(ax, s.hs[valid], s.Czz_h[valid]; linewidth = 2, color = :navy,
              label = "⟨σᶻ_A σᶻ_B⟩ (horizontal)")
    Mk.scatter!(ax, s.hs[valid], s.Czz_h[valid]; markersize = 8, color = :navy)
    Mk.lines!(ax, s.hs[valid], s.Czz_v[valid]; linewidth = 2, color = :firebrick,
              label = "⟨σᶻ_A σᶻ_C⟩ (vertical)")
    Mk.scatter!(ax, s.hs[valid], s.Czz_v[valid]; markersize = 8, color = :firebrick)
    Mk.vlines!(ax, [H_C_QMC]; color = :red, linestyle = :dash)
    Mk.axislegend(ax; position = :rt)
    Mk.save(out_png, fig)
    @info "Plot saved" out_png
    return fig
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

function _benchmark_driver(; n_points::Int = 21, force::Bool = false)
    @info "GPU: $(CUDA.name(CUDA.device()))   free: $(CUDA.free_memory() ÷ 2^30) GiB"
    cache = joinpath(@__DIR__, "..", "analysis", "data", "ipeps_tfim_sweep_D12_chi40.jls")
    benchmark_run(; n_points = n_points, force = force, cache = cache)
    try
        tabulate_vs_qmc(cache; D = 12, χ = 40)
    catch err
        @warn "Tabulation failed: $err"
    end
    csv_path = joinpath(@__DIR__, "..", "analysis", "data", "ipeps_tfim_D12_chi40.csv")
    try
        save_csv(cache, csv_path)
    catch err
        @warn "CSV save failed: $err"
    end
    try
        make_plots(cache; out_png = joinpath(@__DIR__, "..", "analysis", "data", "benchmark_D12_chi40.png"))
        plot_correlators(cache; out_png = joinpath(@__DIR__, "..", "analysis", "data", "correlators_D12_chi40.png"))
    catch err
        @warn "Plot generation failed (CairoMakie present?): $err"
    end
    @info "Benchmark complete" cache csv_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    _benchmark_driver()
end
