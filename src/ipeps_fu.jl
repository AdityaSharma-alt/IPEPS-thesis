# =============================================================================
# ipeps_fu.jl — REAL ALS-based Full Update for the 2×2 iPEPS TFIM, ALL FOUR BONDS.
# Hard-restricted to D ≤ 5, χ ≤ 30.
#
# Replaces the original ipeps_fu.jl which only updated the AB bond.  Adds proper
# vertical-bond support and a driver that loops over all 4 bonds + multiple
# Trotter sweeps with a sensible acceptance tolerance.
#
# Leg conventions (deduced from ipeps_tfim_gpu.jl and verified against working
# absorption code):
#   PEPS tensor:  A[p, U, R, Dn, L]    p = physical, then up/right/down/left
#   Double-layer: m[U, R, Dn, L] each D² (combined ket+bra)
#   CTM corners:  Clu(d,r) Cru(d,l) Cld(u,r) Crd(u,l)
#   CTM edges:    Tu(l,d,r) Td(l,u,r) Tl(u,r,d) Tr(u,l,d)
#
# The four bonds of the 2×2 cell:
#       AB (horizontal, top row)        A.R  ↔ B.L
#       CD (horizontal, bottom row)     C.R  ↔ D.L
#       AC (vertical,   left column)    A.Dn ↔ C.U
#       BD (vertical,   right column)   B.Dn ↔ D.U
#
# CAVEAT — this code is UNTESTED in the cloud environment where it was written.
# The leg conventions in the vertical-bond builders (fu_env_AC, fu_env_BD) are
# the highest-risk part.  Run the smoke test in fu_demo.jl FIRST at D=2, χ=4 to
# catch any leg bugs before doing real runs.
# =============================================================================

include("ipeps_tfim_gpu.jl")

using LinearAlgebra: pinv, svd, Hermitian, cholesky, I, tr, opnorm, norm, eigvals
using Printf

const FU_D_MAX   = 5
const FU_CHI_MAX = 30

# =============================================================================
# §A   Reduced-tensor splits — one per (site, bond-direction) combination.
#
#   split_right(A) — split A on its R leg, used when A is on the LEFT side of a
#                    horizontal bond (e.g. A in AB, C in CD)
#   split_left (B) — split B on its L leg, used when B is on the RIGHT side of a
#                    horizontal bond (e.g. B in AB, D in CD)
#   split_down (A) — split A on its Dn leg, for A in vertical AC bond, or B in BD
#   split_up   (C) — split C on its U leg,  for C in vertical AC bond, or D in BD
# =============================================================================

function split_right(A::CuArray{T,5}) where T
    d, U, R, Dn, L = size(A)
    Aperm = permutedims(A, (2,5,4, 1,3))   # (U, L, Dn, p, R)
    Amat  = reshape(Aperm, U*L*Dn, d*R)
    F     = svd(Amat)
    k     = min(size(F.U, 2), size(F.Vt, 1))
    Q_A   = reshape(F.U[:, 1:k], U, L, Dn, k)
    SV    = (F.S[1:k]) .* F.Vt[1:k, :]
    R_A   = reshape(SV, k, d, R)
    return Q_A, R_A
end

function split_left(B::CuArray{T,5}) where T
    d, U, R, Dn, L = size(B)
    Bperm = permutedims(B, (2,3,4, 1,5))   # (U, R, Dn, p, L)
    Bmat  = reshape(Bperm, U*R*Dn, d*L)
    F     = svd(Bmat)
    k     = min(size(F.U, 2), size(F.Vt, 1))
    Q_B   = reshape(F.U[:, 1:k], U, R, Dn, k)
    SV    = (F.S[1:k]) .* F.Vt[1:k, :]
    R_B   = reshape(SV, k, d, L)
    return Q_B, R_B
end

function split_down(A::CuArray{T,5}) where T
    d, U, R, Dn, L = size(A)
    Aperm = permutedims(A, (2,3,5, 1,4))   # (U, R, L, p, Dn)
    Amat  = reshape(Aperm, U*R*L, d*Dn)
    F     = svd(Amat)
    k     = min(size(F.U, 2), size(F.Vt, 1))
    Q_A   = reshape(F.U[:, 1:k], U, R, L, k)
    SV    = (F.S[1:k]) .* F.Vt[1:k, :]
    R_A   = reshape(SV, k, d, Dn)
    return Q_A, R_A
end

function split_up(C::CuArray{T,5}) where T
    d, U, R, Dn, L = size(C)
    Cperm = permutedims(C, (3,4,5, 1,2))   # (R, Dn, L, p, U)
    Cmat  = reshape(Cperm, R*Dn*L, d*U)
    F     = svd(Cmat)
    k     = min(size(F.U, 2), size(F.Vt, 1))
    Q_C   = reshape(F.U[:, 1:k], R, Dn, L, k)
    SV    = (F.S[1:k]) .* F.Vt[1:k, :]
    R_C   = reshape(SV, k, d, U)
    return Q_C, R_C
end

# -----------------------------------------------------------------------------
# Merges that re-form the full PEPS tensor [p, U, R, Dn, L] from (Q, R).

function merge_right(Q::CuArray{T,4}, Rmat::CuArray{T,3}) where T
    @tensor M[U, L, Dn, p, R] := Q[U, L, Dn, k] * Rmat[k, p, R]
    return permutedims(M, (4, 1, 5, 3, 2))   # → (p, U, R, Dn, L)
end

function merge_left(Q::CuArray{T,4}, Rmat::CuArray{T,3}) where T
    @tensor M[U, R, Dn, p, L] := Q[U, R, Dn, k] * Rmat[k, p, L]
    return permutedims(M, (4, 1, 2, 3, 5))   # → (p, U, R, Dn, L)
end

function merge_down(Q::CuArray{T,4}, Rmat::CuArray{T,3}) where T
    @tensor M[U, R, L, p, Dn] := Q[U, R, L, k] * Rmat[k, p, Dn]
    return permutedims(M, (4, 1, 2, 5, 3))   # → (p, U, R, Dn, L)
end

function merge_up(Q::CuArray{T,4}, Rmat::CuArray{T,3}) where T
    @tensor M[R, Dn, L, p, U] := Q[R, Dn, L, k] * Rmat[k, p, U]
    return permutedims(M, (4, 5, 1, 2, 3))   # → (p, U, R, Dn, L)
end

# =============================================================================
# §B   Double-Q tensors (ket × bra) with the surviving bond legs kept open.
#
# For a Q with 4 legs (3 env-facing D-legs + 1 bond k-leg), QQ has 5 legs:
# three D² env-facing legs (combined ket+bra) plus k, k*.
# =============================================================================

function double_Q_right(Q::CuArray{T,4}) where T   # Q from split_right: (U, L, Dn, k)
    U, L, Dn, k = size(Q)
    @tensor QQ[Uk,Ub,Lk,Lb,Dk,Db,ka,kap] :=
        Q[Uk, Lk, Dk, ka] * conj(Q[Ub, Lb, Db, kap])
    return reshape(QQ, U*U, L*L, Dn*Dn, k, k)
end

function double_Q_left(Q::CuArray{T,4}) where T    # Q from split_left:  (U, R, Dn, k)
    U, R, Dn, k = size(Q)
    @tensor QQ[Uk,Ub,Rk,Rb,Dk,Db,kb,kbp] :=
        Q[Uk, Rk, Dk, kb] * conj(Q[Ub, Rb, Db, kbp])
    return reshape(QQ, U*U, R*R, Dn*Dn, k, k)
end

function double_Q_down(Q::CuArray{T,4}) where T    # Q from split_down:  (U, R, L, k)
    U, R, L, k = size(Q)
    @tensor QQ[Uk,Ub,Rk,Rb,Lk,Lb,ka,kap] :=
        Q[Uk, Rk, Lk, ka] * conj(Q[Ub, Rb, Lb, kap])
    return reshape(QQ, U*U, R*R, L*L, k, k)
end

function double_Q_up(Q::CuArray{T,4}) where T      # Q from split_up:    (R, Dn, L, k)
    R, Dn, L, k = size(Q)
    @tensor QQ[Rk,Rb,Dk,Db,Lk,Lb,kc,kcp] :=
        Q[Rk, Dk, Lk, kc] * conj(Q[Rb, Db, Lb, kcp])
    return reshape(QQ, R*R, Dn*Dn, L*L, k, k)
end

# =============================================================================
# §C   Bond environments N(k_a, k_a*, k_b, k_b*) for each of the 4 bonds.
#
# Each function takes the CTM env, the unchanged double-layers of the OTHER
# two sites (e.g. for AB bond: dl.mC, dl.mD), and the double-Q tensors of the
# active two sites.  Returns N as a 4-leg tensor.
# =============================================================================

# -- AB bond (horizontal, top row) --------------------------------------------
#    Active: A (Q_A right-split), B (Q_B left-split)
#    Frozen: dl.mC (bottom-left), dl.mD (bottom-right)
function fu_env_AB(e::CTMEnv, dl::DoubleLayers,
                   QQ_A::CuArray{T,5}, QQ_B::CuArray{T,5}) where T
    @tensor CT[a, UA, c]       := e.Clu[a, b] * e.Tu1[b, UA, c]
    @tensor CTL[UA, c, LA, f]  := CT[a, UA, c] * e.Tl1[a, LA, f]
    @tensor CTLA[c, DA, ka, kap, f] := CTL[UA, c, LA, f] * QQ_A[UA, LA, DA, ka, kap]
    @tensor CldT[g, DC, i]     := e.Cld[g, h] * e.Td1[h, DC, i]
    @tensor CldTL[f, LC, DC, i]:= CldT[g, DC, i] * e.Tl2[f, LC, g]
    @tensor CldM[f, UC, RC, i] := CldTL[f, LC, DC, i] * dl.mC[UC, RC, DC, LC]
    @tensor LeftHalf[c, RC, i, ka, kap] := CTLA[c, DA, ka, kap, f] * CldM[f, DA, RC, i]

    @tensor CTr[c, UB, e_]     := e.Cru[d, e_] * e.Tu2[c, UB, d]
    @tensor CTrR[c, UB, RB, l] := CTr[c, UB, e_] * e.Tr1[e_, RB, l]
    @tensor CTrRB[c, DB, kb, kbp, l] := CTrR[c, UB, RB, l] * QQ_B[UB, RB, DB, kb, kbp]
    @tensor CrdT[i, DD, k]     := e.Td2[i, DD, j] * e.Crd[j, k]
    @tensor CrdTR[l, RD, DD, i]:= CrdT[i, DD, k] * e.Tr2[l, RD, k]
    @tensor CrdTRD[l, UD, LD, i]:= CrdTR[l, RD, DD, i] * dl.mD[UD, RD, DD, LD]
    @tensor RightHalf[c, LD, i, kb, kbp] := CTrRB[c, DB, kb, kbp, l] * CrdTRD[l, DB, LD, i]

    @tensor N[ka, kap, kb, kbp] := LeftHalf[c, R_, i, ka, kap] * RightHalf[c, R_, i, kb, kbp]
    return N
end

# -- CD bond (horizontal, bottom row) -----------------------------------------
#    Active: C (Q_C right-split), D (Q_D left-split)
#    Frozen: dl.mA (top-left), dl.mB (top-right)
function fu_env_CD(e::CTMEnv, dl::DoubleLayers,
                   QQ_C::CuArray{T,5}, QQ_D::CuArray{T,5}) where T
    @tensor CT[a, UA, c]        := e.Clu[a, b] * e.Tu1[b, UA, c]
    @tensor CTL[UA, c, LA, f]   := CT[a, UA, c] * e.Tl1[a, LA, f]
    @tensor CTLA[c, DA, RA, f]  := CTL[UA, c, LA, f] * dl.mA[UA, RA, DA, LA]
    @tensor CldT[g, DC, i]      := e.Cld[g, h] * e.Td1[h, DC, i]
    @tensor CldTL[f, LC, DC, i] := CldT[g, DC, i] * e.Tl2[f, LC, g]
    @tensor CldM[f, UC, kc, kcp, i] := CldTL[f, LC, DC, i] * QQ_C[UC, DC, LC, kc, kcp]
    @tensor LeftHalf[c, RA, i, kc, kcp] := CTLA[c, DA, RA, f] * CldM[f, DA, kc, kcp, i]

    @tensor CTr[c, UB, e_]      := e.Cru[d, e_] * e.Tu2[c, UB, d]
    @tensor CTrR[c, UB, RB, l]  := CTr[c, UB, e_] * e.Tr1[e_, RB, l]
    @tensor CTrRB[c, DB, LB, l] := CTrR[c, UB, RB, l] * dl.mB[UB, RB, DB, LB]
    @tensor CrdT[i, DD, k]      := e.Td2[i, DD, j] * e.Crd[j, k]
    @tensor CrdTR[l, RD, DD, i] := CrdT[i, DD, k] * e.Tr2[l, RD, k]
    @tensor CrdTRD[l, UD, kd, kdp, i] := CrdTR[l, RD, DD, i] * QQ_D[UD, RD, DD, kd, kdp]
    @tensor RightHalf[c, LB, i, kd, kdp] := CTrRB[c, DB, LB, l] * CrdTRD[l, DB, kd, kdp, i]

    @tensor N[kc, kcp, kd, kdp] := LeftHalf[c, R_, i, kc, kcp] * RightHalf[c, R_, i, kd, kdp]
    return N
end

# -- AC bond (vertical, left column) ------------------------------------------
#    Active: A (Q_A down-split), C (Q_C up-split)
#    Frozen: dl.mB (top-right), dl.mD (bot-right)
function fu_env_AC(e::CTMEnv, dl::DoubleLayers,
                   QQ_A::CuArray{T,5}, QQ_C::CuArray{T,5}) where T
    @tensor CT[a, UA, c]        := e.Clu[a, b] * e.Tu1[b, UA, c]
    @tensor CTL[UA, c, LA, f]   := CT[a, UA, c] * e.Tl1[a, LA, f]
    @tensor CTLA[c, RA, ka, kap, f] := CTL[UA, c, LA, f] * QQ_A[UA, RA, LA, ka, kap]
    @tensor CTr[c, UB, e_]      := e.Cru[d, e_] * e.Tu2[c, UB, d]
    @tensor CTrR[c, UB, RB, l]  := CTr[c, UB, e_] * e.Tr1[e_, RB, l]
    @tensor CTrRB[c, DB, LB, l] := CTrR[c, UB, RB, l] * dl.mB[UB, RB, DB, LB]
    @tensor TopHalf[f, ka, kap, DB, l] := CTLA[c, RA, ka, kap, f] * CTrRB[c, DB, RA, l]

    @tensor CldT[g, DC, i]      := e.Cld[g, h] * e.Td1[h, DC, i]
    @tensor CldTL[f, LC, DC, i] := CldT[g, DC, i] * e.Tl2[f, LC, g]
    @tensor CldTLC[f, RC, kc, kcp, i] := CldTL[f, LC, DC, i] * QQ_C[RC, DC, LC, kc, kcp]
    @tensor CrdT[i, DD, k]      := e.Td2[i, DD, j] * e.Crd[j, k]
    @tensor CrdTR[l, RD, DD, i] := CrdT[i, DD, k] * e.Tr2[l, RD, k]
    @tensor CrdTRD[l, UD, LD, i]:= CrdTR[l, RD, DD, i] * dl.mD[UD, RD, DD, LD]
    @tensor BotHalf[f, kc, kcp, UD, l] := CldTLC[f, RC, kc, kcp, i] * CrdTRD[l, UD, RC, i]

    @tensor N[ka, kap, kc, kcp] := TopHalf[f, ka, kap, DB, l] * BotHalf[f, kc, kcp, DB, l]
    return N
end

# -- BD bond (vertical, right column) -----------------------------------------
#    Active: B (Q_B down-split), D (Q_D up-split)
#    Frozen: dl.mA (top-left), dl.mC (bot-left)
function fu_env_BD(e::CTMEnv, dl::DoubleLayers,
                   QQ_B::CuArray{T,5}, QQ_D::CuArray{T,5}) where T
    @tensor CT[a, UA, c]        := e.Clu[a, b] * e.Tu1[b, UA, c]
    @tensor CTL[UA, c, LA, f]   := CT[a, UA, c] * e.Tl1[a, LA, f]
    @tensor CTLA[c, DA, RA, f]  := CTL[UA, c, LA, f] * dl.mA[UA, RA, DA, LA]
    @tensor CTr[c, UB, e_]      := e.Cru[d, e_] * e.Tu2[c, UB, d]
    @tensor CTrR[c, UB, RB, l]  := CTr[c, UB, e_] * e.Tr1[e_, RB, l]
    @tensor CTrRB[c, LB, kb, kbp, l] := CTrR[c, UB, RB, l] * QQ_B[UB, RB, LB, kb, kbp]
    @tensor TopHalf[f, DA, kb, kbp, l] := CTLA[c, DA, RA, f] * CTrRB[c, RA, kb, kbp, l]

    @tensor CldT[g, DC, i]      := e.Cld[g, h] * e.Td1[h, DC, i]
    @tensor CldTL[f, LC, DC, i] := CldT[g, DC, i] * e.Tl2[f, LC, g]
    @tensor CldM[f, UC, RC, i]  := CldTL[f, LC, DC, i] * dl.mC[UC, RC, DC, LC]
    @tensor CrdT[i, DD, k]      := e.Td2[i, DD, j] * e.Crd[j, k]
    @tensor CrdTR[l, RD, DD, i] := CrdT[i, DD, k] * e.Tr2[l, RD, k]
    @tensor CrdTRD[l, LD, kd, kdp, i] := CrdTR[l, RD, DD, i] * QQ_D[RD, DD, LD, kd, kdp]
    @tensor BotHalf[f, UC, kd, kdp, l] := CldM[f, UC, RC, i] * CrdTRD[l, RC, kd, kdp, i]

    @tensor N[kb, kbp, kd, kdp] := TopHalf[f, UC, kb, kbp, l] * BotHalf[f, UC, kd, kdp, l]
    return N
end

# =============================================================================
# §D   ALS solver — regularized cholesky with pinv fallback.  CPU.
# =============================================================================

function als_solve(M::AbstractMatrix, W::AbstractVector;
                   reg_start::Real=1e-10, max_reg::Real=1e-3)
    n   = size(M, 1)
    Mh  = (M + M') / 2
    trM = real(tr(Mh)) / n
    reg = reg_start
    while reg <= max_reg
        Mreg = Mh + reg * trM * I
        try
            C = cholesky(Hermitian(Mreg))
            X = C \ W
            if !any(isnan, X) && !any(isinf, X)
                return X, (method=:cholesky, reg=reg)
            end
        catch
        end
        reg *= 10
    end
    X = pinv(Mh + max_reg * trM * I) * W
    return X, (method=:pinv, reg=max_reg)
end

# =============================================================================
# §E   Generic per-bond FU update.  Solves N·A = W via ALS.
# =============================================================================

function _fu_update_bond!(st::iPEPS, e::CTMEnv, dl::DoubleLayers,
                          gate::CuArray{T,4}, sL::Symbol, sR::Symbol;
                          split_L::Function, split_R::Function,
                          merge_L::Function, merge_R::Function,
                          QQ_L_fn::Function, QQ_R_fn::Function,
                          env_fn::Function,
                          max_als::Int=30, tol_als::Real=1e-7,
                          verbose::Bool=true) where T
    AL = getfield(st, sL); AR = getfield(st, sR)
    Q_L, R_L_init = split_L(AL)
    Q_R, R_R_init = split_R(AR)
    @assert size(R_L_init, 3) == size(R_R_init, 3) "$(sL)$(sR) bond dim mismatch"

    # Apply the gate to (R_L, R_R) → T_target
    @tensor T_target[a, dL, dR, b] :=
        gate[dL, dR, dLp, dRp] * R_L_init[a, dLp, m] * R_R_init[b, dRp, m]

    QQ_L = QQ_L_fn(Q_L); QQ_R = QQ_R_fn(Q_R)
    N    = env_fn(e, dl, QQ_L, QQ_R)

    # CPU ALS (small matrices at D ≤ 5)
    N_cpu = Array(N)
    T_cpu = Array(T_target)
    R_L   = Array(R_L_init)
    R_R   = Array(R_R_init)

    k_a   = size(R_L, 1); k_b = size(R_R, 1)
    d     = size(R_L, 2); R_dim = size(R_L, 3)

    # Symmetrize N for hermiticity stability
    N_cpu = (N_cpu .+ permutedims(conj.(N_cpu), (2, 1, 4, 3))) ./ 2

    log_lines = String[]
    converged = false

    for iter in 1:max_als
        prev_R_L = copy(R_L); prev_R_R = copy(R_R)

        @tensor tmpA[a, ap, m, mp] :=
            N_cpu[a, ap, b, bp] * R_R[b, dR, m] * conj(R_R[bp, dR, mp])
        M_A_mat = zeros(ComplexF64, k_a*d*R_dim, k_a*d*R_dim)
        for dL in 1:d, a in 1:k_a, m in 1:R_dim, ap in 1:k_a, mp in 1:R_dim
            row = (a-1)*d*R_dim + (dL-1)*R_dim + m
            col = (ap-1)*d*R_dim + (dL-1)*R_dim + mp
            M_A_mat[row, col] = tmpA[a, ap, m, mp]
        end
        @tensor W_A_t[a, dL, m] :=
            N_cpu[a, ap, b, bp] * T_cpu[ap, dL, dR, b] * conj(R_R[bp, dR, m])
        W_A_vec = zeros(ComplexF64, k_a*d*R_dim)
        for a in 1:k_a, dL in 1:d, m in 1:R_dim
            W_A_vec[(a-1)*d*R_dim + (dL-1)*R_dim + m] = W_A_t[a, dL, m]
        end
        x, infoA = als_solve(M_A_mat, W_A_vec)
        R_L_new = zeros(ComplexF64, k_a, d, R_dim)
        for a in 1:k_a, dL in 1:d, m in 1:R_dim
            R_L_new[a, dL, m] = x[(a-1)*d*R_dim + (dL-1)*R_dim + m]
        end
        nrmA = norm(R_L_new); R_L_new ./= max(nrmA, 1e-12)
        R_L = R_L_new

        @tensor tmpB[b, bp, m, mp] :=
            N_cpu[a, ap, b, bp] * R_L[a, dL, m] * conj(R_L[ap, dL, mp])
        M_B_mat = zeros(ComplexF64, k_b*d*R_dim, k_b*d*R_dim)
        for dR in 1:d, b in 1:k_b, m in 1:R_dim, bp in 1:k_b, mp in 1:R_dim
            row = (b-1)*d*R_dim + (dR-1)*R_dim + m
            col = (bp-1)*d*R_dim + (dR-1)*R_dim + mp
            M_B_mat[row, col] = tmpB[b, bp, m, mp]
        end
        @tensor W_B_t[b, dR, m] :=
            N_cpu[a, ap, b, bp] * T_cpu[a, dL, dR, bp] * conj(R_L[ap, dL, m])
        W_B_vec = zeros(ComplexF64, k_b*d*R_dim)
        for b in 1:k_b, dR in 1:d, m in 1:R_dim
            W_B_vec[(b-1)*d*R_dim + (dR-1)*R_dim + m] = W_B_t[b, dR, m]
        end
        y, infoB = als_solve(M_B_mat, W_B_vec)
        R_R_new = zeros(ComplexF64, k_b, d, R_dim)
        for b in 1:k_b, dR in 1:d, m in 1:R_dim
            R_R_new[b, dR, m] = y[(b-1)*d*R_dim + (dR-1)*R_dim + m]
        end
        nrmB = norm(R_R_new); R_R_new ./= max(nrmB, 1e-12)
        R_R = R_R_new

        dA_diff = norm(R_L - prev_R_L) / max(norm(R_L), 1e-12)
        dB_diff = norm(R_R - prev_R_R) / max(norm(R_R), 1e-12)
        push!(log_lines, @sprintf("  ALS %2d  dL=%.2e  dR=%.2e  reg_L=%.0e  reg_R=%.0e",
                                  iter, dA_diff, dB_diff, infoA.reg, infoB.reg))
        if max(dA_diff, dB_diff) < tol_als
            converged = true
            verbose && println("  ALS converged at iter $iter")
            break
        end
    end
    if verbose
        for line in log_lines; println(line); end
    end

    # Recompress (R_L · R_R) back to bond dim R_dim
    @tensor RR[a, dL, dR, b] := R_L[a, dL, m] * R_R[b, dR, m]
    Mat   = reshape(RR, k_a*d, d*k_b)
    F     = svd(Mat)
    keep  = min(size(F.U, 2), R_dim)
    sqs   = sqrt.(F.S[1:keep])
    U_kept    = F.U[:, 1:keep] .* sqs'
    R_L_final = reshape(U_kept, k_a, d, keep)
    Vt_kept   = sqs .* F.Vt[1:keep, :]
    R_R_temp  = reshape(Vt_kept, keep, d, k_b)
    R_R_final = permutedims(R_R_temp, (3, 2, 1))

    R_L_gpu = CuArray(R_L_final)
    R_R_gpu = CuArray(R_R_final)
    AL_new  = merge_L(Q_L, R_L_gpu)
    AR_new  = merge_R(Q_R, R_R_gpu)
    AL_new ./= norm(AL_new); AR_new ./= norm(AR_new)

    if any(isnan, Array(AL_new)) || any(isnan, Array(AR_new))
        @warn "FU $(sL)$(sR) produced NaN — bond skipped, state unchanged"
        return (converged=false, reason=:nan, log=log_lines, kept_sv=Float64[])
    end
    setfield!(st, sL, AL_new)
    setfield!(st, sR, AR_new)
    return (converged=converged, n_als=length(log_lines),
            kept_sv=Array(F.S[1:keep]), log=log_lines)
end

# Per-bond wrappers -----------------------------------------------------------

fu_update_AB!(st, e, dl, gate; kw...) = _fu_update_bond!(
    st, e, dl, gate, :A, :B;
    split_L=split_right, split_R=split_left,
    merge_L=merge_right, merge_R=merge_left,
    QQ_L_fn=double_Q_right, QQ_R_fn=double_Q_left,
    env_fn=fu_env_AB, kw...)

fu_update_CD!(st, e, dl, gate; kw...) = _fu_update_bond!(
    st, e, dl, gate, :C, :D;
    split_L=split_right, split_R=split_left,
    merge_L=merge_right, merge_R=merge_left,
    QQ_L_fn=double_Q_right, QQ_R_fn=double_Q_left,
    env_fn=fu_env_CD, kw...)

fu_update_AC!(st, e, dl, gate; kw...) = _fu_update_bond!(
    st, e, dl, gate, :A, :C;
    split_L=split_down, split_R=split_up,
    merge_L=merge_down, merge_R=merge_up,
    QQ_L_fn=double_Q_down, QQ_R_fn=double_Q_up,
    env_fn=fu_env_AC, kw...)

fu_update_BD!(st, e, dl, gate; kw...) = _fu_update_bond!(
    st, e, dl, gate, :B, :D;
    split_L=split_down, split_R=split_up,
    merge_L=merge_down, merge_R=merge_up,
    QQ_L_fn=double_Q_down, QQ_R_fn=double_Q_up,
    env_fn=fu_env_BD, kw...)

# =============================================================================
# §F   Entropy from CTM corner spectrum.
#
# After FU the SU bond weights w.λ_AB are stale.  A better diagnostic is the
# entanglement entropy from diagonalizing the (Hermitianized) upper-left corner
# Clu: this is the entanglement of half the infinite environment crossing the
# corner.  Defines a Shannon entropy on the squared, normalized spectrum.
# =============================================================================

"""
    corner_entropy(e)

Entanglement entropy from the CTM upper-left corner.  Uses singular values
(not eigenvalues) for robustness — eigvals on a near-singular non-Hermitian
matrix produces negative noise that contaminates entropy.  svdvals returns
non-negative magnitudes regardless.

For a converged CTM, Clu represents (quarter-plane RDM)^{1/2}; the squared,
normalized singular values are the entanglement spectrum across the corner.
"""
function corner_entropy(e::CTMEnv)
    s = LinearAlgebra.svdvals(Array(e.Clu))
    p = s .^ 2
    Σ = sum(p)
    Σ > 0 || return 0.0
    p ./= Σ
    -sum(x -> x > 1e-30 ? x*log(x) : 0.0, p)
end

"""
    boundary_entropy(e)

Alternative diagnostic: entropy from the Tu1 boundary tensor's bond spectrum.
Slice Tu1[χ, D², χ] as a (χD²) × χ matrix and take svdvals; the squared
normalized spectrum is the entropy across a vertical cut of the lattice.
More physical than the corner: this is the cylinder entropy in the χ-bond
direction, the standard CTM entanglement-spectrum diagnostic.
"""
function boundary_entropy(e::CTMEnv)
    T = Array(e.Tu1)
    χ, D2, _ = size(T)
    M = reshape(T, χ * D2, χ)
    s = LinearAlgebra.svdvals(M)
    p = s .^ 2
    Σ = sum(p)
    Σ > 0 || return 0.0
    p ./= Σ
    -sum(x -> x > 1e-30 ? x*log(x) : 0.0, p)
end

# =============================================================================
# §G   Run-point driver: SU warmup → multiple FU sweeps over ALL 4 bonds.
#
# Key differences from original `run_point_fu` (which only updated AB):
#   1. FU touches all 4 bonds per sweep (AB, CD, AC, BD).
#   2. Acceptance tolerance is `accept_dE` (default +1e-3, was +1e-10).
#      Single-step Trotter error is O(dτ²)~1e-4 so we must allow small ΔE.
#   3. Tracks corner_entropy (not stale SU λ_AB) per sweep.
#   4. Re-converges CTMRG once per sweep (after all 4 bonds updated), not
#      between bonds.  Trades a bit of strict variationality for ~4× speed.
# =============================================================================

"""
    run_point_fu_full(h; D=4, chi=20, J=1.0, n_fu_sweeps=10, dtau_fu=0.01, ...)

Run SU warmup, then n_fu_sweeps of FU over all 4 bonds (AB, CD, AC, BD).
Log E, |mz|, mx, S after each sweep.  Returns a NamedTuple with the full
history of observables.
"""
function run_point_fu_full(h::Real;
                           D::Int=4, chi::Int=20, J::Real=1.0,
                           n_fu_sweeps::Int=10, dtau_fu::Float64=0.01,
                           su_schedule = [(0.1, 50), (0.05, 100), (0.01, 200)],
                           ctm_iter::Int=50, ctm_tol::Float64=1e-4,
                           accept_dE::Float64=1e-3,
                           verbose::Bool=true, seed::Int=42)
    @assert D    <= FU_D_MAX   "FU restricted to D <= $FU_D_MAX"
    @assert chi  <= FU_CHI_MAX "FU restricted to chi <= $FU_CHI_MAX"

    @printf("\n[run_point_fu_full] h=%.4f D=%d χ=%d  n_fu=%d  dτ=%.4f\n",
            h, D, chi, n_fu_sweeps, dtau_fu)

    println("\n-- SU warmup --")
    st = init_ipeps(2, D; seed=seed)
    w  = init_weights(D)
    simple_update!(st, w, J, h; D_max=D, schedule=su_schedule, verbose=verbose)
    CUDA.reclaim()

    println("\n-- CTMRG initial converge --")
    dl = build_all_dl(st)
    e  = init_ctm(chi, D^2)
    ctmrg_converge!(e, dl, chi; max_iter=ctm_iter, tol=ctm_tol, verbose=verbose)

    E_init  = compute_energy(e, st, dl, J, h)
    mz_init = expect_1site_avg(e, st, dl, σz_gpu())
    mx_init = expect_1site_avg(e, st, dl, σx_gpu())
    S_init  = corner_entropy(e)
    Sb_init = boundary_entropy(e)
    @printf("\n  POST-SU :  E=%.6f  |mz|=%.4f  mx=%.4f  S_corner=%.4f  S_bnd=%.4f\n",
            E_init, abs(mz_init), mx_init, S_init, Sb_init)

    gate    = ising_gate(J, h, dtau_fu)
    history = NamedTuple[]
    push!(history, (sweep=0, E=E_init, mz=mz_init, mx=mx_init, S=S_init, Sb=Sb_init))
    E_best  = E_init
    A_save  = copy(st.A); B_save = copy(st.B)
    C_save  = copy(st.C); D_save = copy(st.D)

    for sw in 1:n_fu_sweeps
        @printf("\n-- FU sweep %d / %d  (dτ=%.4f) --\n", sw, n_fu_sweeps, dtau_fu)
        A_save .= st.A; B_save .= st.B; C_save .= st.C; D_save .= st.D
        bond_nan = false

        for (bond_name, bond_fn) in [("AB", fu_update_AB!), ("CD", fu_update_CD!),
                                      ("AC", fu_update_AC!), ("BD", fu_update_BD!)]
            verbose && @printf("  bond %s\n", bond_name)
            diag = bond_fn(st, e, dl, gate; verbose=false)
            if get(diag, :reason, nothing) === :nan
                @warn "  bond $bond_name produced NaN; reverting full sweep $sw"
                bond_nan = true
                break
            end
        end

        if bond_nan
            st.A .= A_save; st.B .= B_save; st.C .= C_save; st.D .= D_save
            continue
        end

        dl = build_all_dl(st)
        ctmrg_converge!(e, dl, chi; max_iter=30, tol=ctm_tol, verbose=false)

        E_new  = compute_energy(e, st, dl, J, h)
        mz_new = expect_1site_avg(e, st, dl, σz_gpu())
        mx_new = expect_1site_avg(e, st, dl, σx_gpu())
        S_new  = corner_entropy(e)
        Sb_new = boundary_entropy(e)
        dE     = E_new - E_best
        @printf("    E=%.6f  ΔE=%+.3e  |mz|=%.4f  mx=%.4f  S_c=%.4f  S_b=%.4f\n",
                E_new, dE, abs(mz_new), mx_new, S_new, Sb_new)
        push!(history, (sweep=sw, E=E_new, mz=mz_new, mx=mx_new, S=S_new, Sb=Sb_new))

        if dE > accept_dE
            @printf("    ΔE > %.0e — reverting this sweep\n", accept_dE)
            st.A .= A_save; st.B .= B_save; st.C .= C_save; st.D .= D_save
            dl = build_all_dl(st)
            ctmrg_converge!(e, dl, chi; max_iter=30, tol=ctm_tol, verbose=false)
        elseif E_new < E_best
            E_best = E_new
        end
    end

    dl = build_all_dl(st)
    E  = compute_energy(e, st, dl, J, h)
    mz = expect_1site_avg(e, st, dl, σz_gpu())
    mx = expect_1site_avg(e, st, dl, σx_gpu())
    S  = corner_entropy(e)
    Sb = boundary_entropy(e)

    @printf("\n-- FINAL  h=%.4f  D=%d  χ=%d --\n", h, D, chi)
    @printf("  POST-SU:        E=%.6f  |mz|=%.4f  mx=%.4f  S_c=%.4f  S_b=%.4f\n",
            E_init, abs(mz_init), mx_init, S_init, Sb_init)
    @printf("  POST-FU(full):  E=%.6f  |mz|=%.4f  mx=%.4f  S_c=%.4f  S_b=%.4f\n",
            E, abs(mz), mx, S, Sb)
    @printf("  ΔE total = %+.6e   ΔS_b = %+.4f\n", E - E_init, Sb - Sb_init)

    return (h=Float64(h), E=Float64(E), mz=Float64(mz), mx=Float64(mx),
            S=Float64(S), Sb=Float64(Sb),
            E_init=Float64(E_init), mz_init=Float64(mz_init),
            mx_init=Float64(mx_init), S_init=Float64(S_init),
            Sb_init=Float64(Sb_init),
            history=history,
            D=D, chi=chi, n_fu_sweeps=n_fu_sweeps, dtau_fu=dtau_fu)
end

println("ipeps_fu.jl (all-bonds FU) loaded.")
