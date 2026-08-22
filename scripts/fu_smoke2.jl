# =============================================================================
# fu_smoke2.jl — meaningful smoke test for the all-bonds FU.
#
# Does a real SU warmup + CTMRG converge first, THEN runs one FU update on
# each of the 4 bonds and checks the energy change.  This actually exercises
# the variational update — unlike fu_demo.jl smoke which only checked N
# properties using random tensors (which N had no reason to be physical with).
#
# Pass criteria per bond:
#   - ALS produces no NaN/Inf
#   - energy stays finite
#   - |dE| < 0.5  (FU at dτ=0.01 shouldn't move energy by more than ~0.1
#     in a single bond update)
# =============================================================================

include(joinpath(@__DIR__, "..", "src", "ipeps_fu.jl"))

println("\n" * "="^70)
println(" FU smoke test 2  —  SU warmup, CTMRG converge, then 1× FU per bond ")
println("="^70)

D = 2; chi = 8
h = 3.0   # near critical — most interesting regime
J = 1.0
dtau = 0.01

println("\n-- SU warmup --")
st = init_ipeps(2, D; seed=42)
w  = init_weights(D)
simple_update!(st, w, J, h; D_max=D,
               schedule=[(0.1, 50), (0.01, 100)],
               verbose=false)

println("\n-- CTMRG converge --")
dl = build_all_dl(st)
e  = init_ctm(chi, D^2)
ctmrg_converge!(e, dl, chi; max_iter=40, tol=1e-5, verbose=false)

E0  = compute_energy(e, st, dl, J, h)
mz0 = expect_1site_avg(e, st, dl, σz_gpu())
mx0 = expect_1site_avg(e, st, dl, σx_gpu())
S0  = corner_entropy(e)
@printf("\nInitial (POST-SU):  E=%.6f  |mz|=%.4f  mx=%.4f  S=%.4f\n",
        E0, abs(mz0), mx0, S0)

# Now check each bond's N for sane structure on the PHYSICAL (not random) env
println("\n-- N tensor properties on physical env --")
bond_specs = [
    ("AB", :A, :B, split_right, split_left, double_Q_right, double_Q_left, fu_env_AB),
    ("CD", :C, :D, split_right, split_left, double_Q_right, double_Q_left, fu_env_CD),
    ("AC", :A, :C, split_down,  split_up,   double_Q_down,  double_Q_up,   fu_env_AC),
    ("BD", :B, :D, split_down,  split_up,   double_Q_down,  double_Q_up,   fu_env_BD),
]

for (name, sL, sR, splL, splR, QL_fn, QR_fn, env_fn) in bond_specs
    AL = getfield(st, sL); AR = getfield(st, sR)
    Q_L, R_L = splL(AL)
    Q_R, R_R = splR(AR)
    QQ_L = QL_fn(Q_L); QQ_R = QR_fn(Q_R)
    N    = env_fn(e, dl, QQ_L, QQ_R)
    N_cpu = Array(N)
    # Symmetrize and check spectrum
    N_sym = (N_cpu .+ permutedims(conj.(N_cpu), (2,1,4,3))) ./ 2
    herm_err = norm(N_cpu - N_sym) / max(norm(N_cpu), 1e-30)
    k_L = size(N_cpu, 1); k_R = size(N_cpu, 3)
    N_mat = reshape(permutedims(N_sym, (1,3, 2,4)), k_L*k_R, k_L*k_R)
    N_mat_h = (N_mat + N_mat') / 2
    evals = real.(LinearAlgebra.eigvals(N_mat_h))
    min_ev = minimum(evals); max_ev = maximum(evals)
    @printf("  bond %s  k_L=%d k_R=%d  herm_err=%.2e  λ_max=%.3e  λ_min/λ_max=%+.2e\n",
            name, k_L, k_R, herm_err, max_ev,
            min_ev / max(abs(max_ev), 1e-30))
end

# The actual test: run ONE FU step on each bond
println("\n-- Per-bond FU update (single sweep) --")
gate = ising_gate(J, h, dtau)
for (name, _, _, _, _, _, _, _) in bond_specs
    A_save = copy(st.A); B_save = copy(st.B)
    C_save = copy(st.C); D_save = copy(st.D)

    fn = name == "AB" ? fu_update_AB! :
         name == "CD" ? fu_update_CD! :
         name == "AC" ? fu_update_AC! : fu_update_BD!

    diag = fn(st, e, dl, gate; verbose=false)
    nan_or_skip = get(diag, :reason, nothing) === :nan

    # Recompute energy with refreshed dl + CTMRG
    dl_new = build_all_dl(st)
    e_new  = init_ctm(chi, D^2)
    ctmrg_converge!(e_new, dl_new, chi; max_iter=30, tol=1e-5, verbose=false)
    E_new  = compute_energy(e_new, st, dl_new, J, h)
    finite = isfinite(E_new) && !nan_or_skip
    sensible = finite && abs(E_new - E0) < 0.5

    status = sensible ? "OK" : "FAIL"
    @printf("  bond %s  ALS_converged=%-5s  E=%.6f  ΔE=%+.4f  [%s]\n",
            name, string(diag.converged), E_new, E_new - E0, status)

    # Restore state for next bond test
    st.A .= A_save; st.B .= B_save; st.C .= C_save; st.D .= D_save
end

println("\nSmoke test 2 done.")
