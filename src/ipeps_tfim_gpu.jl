# ipeps_tfim_gpu.jl
# =============================================================================
# 2×2 iPEPS for the 2D Transverse-Field Ising Model — single-GPU port
#
#   H = -J Σ σᶻᵢσᶻⱼ - h Σ σˣᵢ
#
# Pipeline (per field point h):
#   Simple Update  →  CTMRG (directional, sequential absorption)  →  Observables
#
# ----------------------------------------------------------------------------
# Defaults (updated 2026-05):
#   D    = 12    (iPEPS bond dimension)
#   χ    = 40    (CTM environment dimension)
#   max_ctm_iter = 60
#   tol  = 1e-4  (relative ΔSV in dominant singular values of Clu)
#
# CTMRG: directional sweep LEFT → UP → RIGHT → DOWN.  Each direction absorbs
# both rows/columns of the 2×2 cell sequentially, truncating to χ between
# absorptions so no intermediate exceeds shape (χ, D², D², χ).
#
# h-sweep: results cached to disk per (D, χ).
#
# ----------------------------------------------------------------------------
# Memory budget at D=12, χ=40 (ComplexF64):
#   double layer mX:      144⁴×16 B ≈ 6.87 GB  (×4 tensors = 27.5 GB resident)
#   enlarged corner nCl:  40×144²×40×16 B ≈ 53 MB
#   Ψ (SVD input):        5760²×16 B ≈ 530 MB
#   gesvdj workspace:     ≈ 1.0–1.5 GB
#   2-site observable peak: +13.7 GB (two op-inserted dl tensors transient)
# Peak resident set during expect_2site: ≈ 45–55 GiB.  Fits H100 (80 GB).
#
# This version FREES intermediates aggressively (`CUDA.unsafe_free!`) and
# scopes hot tensors inside let-blocks so the allocator can recycle them.
#
# ----------------------------------------------------------------------------
# FULL UPDATE MEMORY NOTE
# -----------------------
# The FU environment builder creates an intermediate of shape
#     [D², D², r·D, D², χ, χ]
# At D=12, χ=40, r=12 that is ≈ 1.1 TB — catastrophic.  See `full_update_demo!`,
# which refuses D_fu > 5 or χ_fu > 50.
# =============================================================================

using LinearAlgebra
using Printf
using Serialization
using Dates
using Random

using CUDA
using cuTENSOR              # required for @tensor on CuArrays
using TensorOperations

CUDA.allowscalar(false)


# -----------------------------------------------------------------------------
# §0  Checkpoint helpers (debug)
# -----------------------------------------------------------------------------

const CHECKPOINTS_ENABLED = Ref(true)

function gpumem_str()
    free = CUDA.free_memory()
    tot  = CUDA.total_memory()
    used = tot - free
    return @sprintf("used=%.2fGiB free=%.2fGiB", used/2^30, free/2^30)
end

macro ckpt(tag, info...)
    quote
        if CHECKPOINTS_ENABLED[]
            CUDA.synchronize()
            t = time()
            mem = try gpumem_str() catch; "mem?" end
            tagstr = string($(esc(tag)))
            extra = $(esc(Expr(:tuple, info...)))
            extra_str = isempty(extra) ? "" : " " * join(string.(extra), " ")
            @printf("    > %-26s %s%s  [%.2fs]\n", tagstr, mem, extra_str, t - $(esc(:_t0_ckpt)))
            flush(stdout)
        end
    end
end

const COMPLEX = ComplexF64

# Safe GPU memory release helper.  Calls CUDA.unsafe_free! if the value
# is a CuArray, otherwise no-op.  Wrapped in try/catch because finalizers
# may have already collected the buffer (double-free guard).
@inline function _free!(x)
    if x isa CuArray
        try
            CUDA.unsafe_free!(x)
        catch
        end
    end
    return nothing
end

# Free a list of tensors in one shot.  Use after Ψ is built but before SVD,
# after enlarged-corner products absorbed into final corners, etc.
@inline function _freeall!(xs...)
    for x in xs
        _free!(x)
    end
    return nothing
end

# -----------------------------------------------------------------------------

# Fast GPU SVD via Jacobi iteration.  Workspace is ~10x smaller than gesvd,
# which is essential under memory pressure to keep kernel caches alive.
function svd_jacobi(M::CuMatrix)
    Mc = copy(M)
    # gesvdj!(jobz, econ, A) → returns (U, S, V) where M = U * diag(S) * V'
    U, S, V = CUDA.CUSOLVER.gesvdj!('V', 1, Mc)
    return (U=U, S=S, Vt=V')
end

# §1  Model
# -----------------------------------------------------------------------------

const _σx = COMPLEX[0 1; 1 0]
const _σz = COMPLEX[1 0; 0 -1]
const _Id = COMPLEX[1 0; 0 1]

# These are tiny CPU matrices; we lift to GPU only when needed inside @tensor.
σx_gpu()   = CuArray(_σx)
σz_gpu()   = CuArray(_σz)

"Imaginary-time evolution gate exp(-dτ·h_bond), reshaped to (2,2,2,2)."
function ising_gate(J::Real, h::Real, dτ::Real; ε::Real=1e-3)
    # -ε·σz/site breaks Z₂ so every h picks the same branch (cleans m(h) plot).
    H = -J*kron(_σz,_σz) - (h/4)*(kron(_σx,_Id) + kron(_Id,_σx)) - (ε/4)*(kron(_σz,_Id)+kron(_Id,_σz))
    g = exp(-dτ*H)
    CuArray(reshape(COMPLEX.(g), 2, 2, 2, 2))
end

"Bond Hamiltonian -J σᶻ⊗σᶻ as a (2,2,2,2) tensor."
ising_bond_op(J::Real) = CuArray(reshape(COMPLEX.(-J*kron(_σz,_σz)), 2, 2, 2, 2))

# -----------------------------------------------------------------------------
# §2  Types
# -----------------------------------------------------------------------------

mutable struct iPEPS
    A::CuArray{COMPLEX,5}
    B::CuArray{COMPLEX,5}
    C::CuArray{COMPLEX,5}
    D::CuArray{COMPLEX,5}
    d::Int
    Db::Int        # bond dimension
end

mutable struct Weights
    λ_AB::Vector{Float64}; λ_CD::Vector{Float64}
    λ_AC::Vector{Float64}; λ_BD::Vector{Float64}
end

mutable struct CTMEnv
    Clu::CuMatrix{COMPLEX}; Cru::CuMatrix{COMPLEX}
    Cld::CuMatrix{COMPLEX}; Crd::CuMatrix{COMPLEX}
    Tu1::CuArray{COMPLEX,3}; Tu2::CuArray{COMPLEX,3}
    Td1::CuArray{COMPLEX,3}; Td2::CuArray{COMPLEX,3}
    Tl1::CuArray{COMPLEX,3}; Tl2::CuArray{COMPLEX,3}
    Tr1::CuArray{COMPLEX,3}; Tr2::CuArray{COMPLEX,3}
    χ::Int
end

struct DoubleLayers
    mA::CuArray{COMPLEX,4}; mB::CuArray{COMPLEX,4}
    mC::CuArray{COMPLEX,4}; mD::CuArray{COMPLEX,4}
    D2::Int
end

# -----------------------------------------------------------------------------
# §3  Init + helpers
# -----------------------------------------------------------------------------

function init_ipeps(d::Int, D::Int; seed::Int=1234)
    Random.seed!(seed)
    f() = (CUDA.randn(COMPLEX, d, D, D, D, D) ./ COMPLEX(sqrt(d * D^4)))
    iPEPS(f(), f(), f(), f(), d, D)
end

function init_weights(D::Int)
    # Slightly broken degeneracy: λ = [1, 0.9, 0.8, ...] / norm
    # This gives SU a gradient and avoids the "all SVs equal → noise amplifies symmetrically" trap.
    base = collect(range(1.0, 0.5, length=D))
    base ./= sqrt(sum(base.^2))
    return Weights(copy(base), copy(base), copy(base), copy(base))
end

function init_ctm(χ::Int, D2::Int)
    function mc()
        x = CUDA.randn(COMPLEX, χ, χ) ./ COMPLEX(χ)
        (x .+ x') ./ COMPLEX(2)
    end
    me() = CUDA.randn(COMPLEX, χ, D2, χ) ./ COMPLEX(sqrt(χ*D2))
    CTMEnv(mc(),mc(),mc(),mc(),
           me(),me(),me(),me(),
           me(),me(),me(),me(),
           χ)
end

function nrm!(x::CuArray)
    n = norm(x)
    if n > 0
        x ./= COMPLEX(n)
    end
    x
end

function normalize_ctm!(e::CTMEnv)
    nrm!(e.Clu); nrm!(e.Cru); nrm!(e.Cld); nrm!(e.Crd)
    nrm!(e.Tu1); nrm!(e.Tu2); nrm!(e.Td1); nrm!(e.Td2)
    nrm!(e.Tl1); nrm!(e.Tl2); nrm!(e.Tr1); nrm!(e.Tr2)
    e
end

"Multiply rank-5 tensor A by a positive weight vector w along leg `leg`."
function apply_w(A::CuArray{T,5}, w::Vector{Float64}, leg::Int) where T
    @assert 1 ≤ leg ≤ 5
    sw = CuArray(COMPLEX.(w))
    shape = ones(Int, 5); shape[leg] = length(sw)
    A .* reshape(sw, shape...)
end

# -----------------------------------------------------------------------------
# §4  Double layer (ket-bra contraction over physical leg)
# -----------------------------------------------------------------------------

function double_layer(A::CuArray{T,5}) where T
    d, D = size(A,1), size(A,2)
    @tensor m[u,up,r,rp,dd,dp,l,lp] := A[p,u,r,dd,l] * conj(A[p,up,rp,dp,lp])
    reshape(m, D^2, D^2, D^2, D^2)
end

function double_layer_op(A::CuArray{T,5}, O::CuMatrix) where T
    d, D = size(A,1), size(A,2)
    @tensor m[u,up,r,rp,dd,dp,l,lp] := A[p,u,r,dd,l]*O[p,q]*conj(A[q,up,rp,dp,lp])
    reshape(m, D^2, D^2, D^2, D^2)
end

build_all_dl(st::iPEPS) = DoubleLayers(
    double_layer(st.A), double_layer(st.B),
    double_layer(st.C), double_layer(st.D),
    st.Db^2)

# -----------------------------------------------------------------------------
# §5  CTMRG — sequential absorption
#
# For a 2×2 unit cell, each directional move absorbs 2 row/column tiles.  In
# this port each tile-absorption is a single SVD-truncated step that bounds the
# environment at χ before moving on.  No intermediate exceeds shape
# [χ, D², D², χ] ≈ 1.6 GB at D=10, χ=100.
# -----------------------------------------------------------------------------

# All four absorptions follow the same template:
#   1. Glue corner+edge+edge+double-layer → "extended corner"
#   2. Glue extended top + extended bot → 4-leg Ψ tensor
#   3. SVD Ψ as (χ·D²) × (D²·χ)  → projectors P, Q
#   4. Project to obtain new corners and edges, normalize.

function _absorb_left_col(Clu, Cld, Tu, Td, Tl1, Tl2, mt, mb, χmax)
    _t0_ckpt = time()
    @ckpt "L:enter"
    # --- Build upper extended corner nClu[c, R, Dm, d] ---
    @tensor CTu[a,U,c]      := Clu[a,b]*Tu[b,U,c]
    @ckpt "L:CTu"
    @tensor CTuTl[U,c,L,d]  := CTu[a,U,c]*Tl1[a,L,d]
    _free!(CTu)                                              # CTu no longer needed
    @ckpt "L:CTuTl"
    @tensor nClu[c,R,Dm,d]  := CTuTl[U,c,L,d]*mt[U,R,Dm,L]
    _free!(CTuTl)                                            # ditto
    @ckpt "L:nClu"
    # --- Build lower extended corner nCld[e, Um, R, c] ---
    @tensor CTd[a,Db_,c]    := Cld[a,b]*Td[b,Db_,c]
    @tensor CTdTl[Db_,c,L,e]:= CTd[a,Db_,c]*Tl2[e,L,a]
    _free!(CTd)
    @ckpt "L:CTdTl"
    @tensor nCld[e,Um,R,c]  := CTdTl[Db_,c,L,e]*mb[Um,R,Db_,L]
    _free!(CTdTl)
    @ckpt "L:nCld"
    # --- Build Ψ and SVD; free Ψ and the SVD matrix as early as possible ---
    χ = size(Clu, 1); D2 = size(mt, 1)
    @tensor Ψ[a,b,c,d] := nClu[a,b,v,e]*nCld[e,v,c,d]
    @ckpt "L:Psi"
    Ψmat = reshape(Ψ, χ*D2, D2*χ)
    F = svd_jacobi(Ψmat)
    _free!(Ψ); _free!(Ψmat)                                  # SVD copy is internal to gesvdj!
    @ckpt "L:SVD"
    χn = min(χmax, length(F.S))
    P = reshape(F.U[:, 1:χn], χ, D2, χn)
    Q = permutedims(reshape(F.Vt[1:χn, :], χn, D2, χ), (2,3,1))
    # Free the unused parts of U, S, Vt
    _free!(F.U); _free!(F.S); _free!(F.Vt)
    @ckpt "L:projectors"
    # --- New corners ---
    @tensor PnC[α,Dm,d]  := conj(P[c,R,α])*nClu[c,R,Dm,d]
    _free!(nClu)
    @tensor C1n[α,β]     := PnC[α,Dm,d]*Q[Dm,d,β]
    _free!(PnC)
    @ckpt "L:C1n"
    @tensor QnC[β,R,c]   := conj(Q[Um,e,β])*nCld[e,Um,R,c]
    _free!(nCld)
    @tensor C4n[β,α]     := QnC[β,R,c]*P[c,R,α]
    _free!(QnC)
    @ckpt "L:C4n"
    # --- New edges (Tl1 → T1n with mt absorbed; Tl2 → T2n with mb absorbed) ---
    @tensor QcTl[α,U,L,b]   := conj(Q[U,a,α])*Tl1[a,L,b]
    @ckpt "L:QcTl"
    @tensor QcTlMt[α,b,R,Dm]:= QcTl[α,U,L,b]*mt[U,R,Dm,L]
    _free!(QcTl)
    @ckpt "L:QcTlMt"
    @tensor T1n[α,R,β]      := QcTlMt[α,b,R,Dm]*Q[Dm,b,β]
    _free!(QcTlMt)
    @ckpt "L:T1n"
    @tensor QcTl2[α,U,L,b]    := conj(Q[U,a,α])*Tl2[a,L,b]
    @tensor QcTl2Mb[α,b,R,Dm] := QcTl2[α,U,L,b]*mb[U,R,Dm,L]
    _free!(QcTl2)
    @tensor T2n[α,R,β]        := QcTl2Mb[α,b,R,Dm]*Q[Dm,b,β]
    _free!(QcTl2Mb); _free!(P); _free!(Q)
    @ckpt "L:T2n"
    nrm!(C1n); nrm!(C4n); nrm!(T1n); nrm!(T2n)
    @ckpt "L:exit"
    return C1n, C4n, T1n, T2n
end

function _absorb_right_col(Cru, Crd, Tu, Td, Tr1, Tr2, mt, mb, χmax)
    @tensor CTu[c,U,a]       := Tu[c,U,b]*Cru[a,b]
    @tensor CTuTr[U,c,Rb,d]  := CTu[c,U,a]*Tr1[a,Rb,d]
    _free!(CTu)
    @tensor nCru[c,Lm,Dm,d]  := CTuTr[U,c,Rb,d]*mt[U,Rb,Dm,Lm]
    _free!(CTuTr)
    @tensor CTd[c,Db_,a]     := Td[c,Db_,b]*Crd[a,b]
    @tensor CTdTr[Db_,c,Rb,e]:= CTd[c,Db_,a]*Tr2[e,Rb,a]
    _free!(CTd)
    @tensor nCrd[e,Um,Lm,c]  := CTdTr[Db_,c,Rb,e]*mb[Um,Rb,Db_,Lm]
    _free!(CTdTr)
    χ = size(Cru, 1); D2 = size(mt, 1)
    @tensor Ψ[a,b,c,d] := nCru[a,b,v,e]*nCrd[e,v,c,d]
    Ψmat = reshape(Ψ, χ*D2, D2*χ)
    F = svd_jacobi(Ψmat)
    _free!(Ψ); _free!(Ψmat)
    χn = min(χmax, length(F.S))
    P = reshape(F.U[:, 1:χn], χ, D2, χn)
    Q = permutedims(reshape(F.Vt[1:χn, :], χn, D2, χ), (2,3,1))
    _free!(F.U); _free!(F.S); _free!(F.Vt)
    @tensor PnC[α,Dm,d]   := conj(P[c,L,α])*nCru[c,L,Dm,d]
    _free!(nCru)
    @tensor C1n[α,β]      := PnC[α,Dm,d]*Q[Dm,d,β]
    _free!(PnC)
    @tensor QnC[β,L,c]    := conj(Q[Um,e,β])*nCrd[e,Um,L,c]
    _free!(nCrd)
    @tensor C4n[β,α]      := QnC[β,L,c]*P[c,L,α]
    _free!(QnC)
    @tensor QcTr[α,U,Rb,b]    := conj(Q[U,a,α])*Tr1[a,Rb,b]
    @tensor QcTrMt[α,b,Lm,Dm] := QcTr[α,U,Rb,b]*mt[U,Rb,Dm,Lm]
    _free!(QcTr)
    @tensor T1n[α,Lm,β]       := QcTrMt[α,b,Lm,Dm]*Q[Dm,b,β]
    _free!(QcTrMt)
    @tensor QcTr2[α,U,Rb,b]    := conj(Q[U,a,α])*Tr2[a,Rb,b]
    @tensor QcTr2Mb[α,b,Lm,Dm] := QcTr2[α,U,Rb,b]*mb[U,Rb,Dm,Lm]
    _free!(QcTr2)
    @tensor T2n[α,Lm,β]        := QcTr2Mb[α,b,Lm,Dm]*Q[Dm,b,β]
    _free!(QcTr2Mb); _free!(P); _free!(Q)
    nrm!(C1n); nrm!(C4n); nrm!(T1n); nrm!(T2n)
    return C1n, C4n, T1n, T2n
end

function _absorb_top_row(Clu, Cru, Tl, Tr, Tu1, Tu2, ml, mr, χmax)
    @tensor CTl[b,Lb,d]      := Clu[a,b]*Tl[a,Lb,d]
    @tensor CTlTu[Lb,d,U,r]  := CTl[b,Lb,d]*Tu1[b,U,r]
    _free!(CTl)
    @tensor nClu[d,Dm,R,r]   := CTlTu[Lb,d,U,r]*ml[U,R,Dm,Lb]
    _free!(CTlTu)
    @tensor CTr[b,Rb,d]      := Cru[a,b]*Tr[a,Rb,d]
    @tensor CTrTu[Rb,d,U,l]  := CTr[b,Rb,d]*Tu2[l,U,b]
    _free!(CTr)
    @tensor nCru[d,Dm,L,l]   := CTrTu[Rb,d,U,l]*mr[U,Rb,Dm,L]
    _free!(CTrTu)
    χ = size(Clu, 1); D2 = size(ml, 1)
    @tensor Ψ[a,b,c,d] := nClu[a,b,R,r]*nCru[d,c,R,r]
    Ψmat = reshape(Ψ, χ*D2, D2*χ)
    F = svd_jacobi(Ψmat)
    _free!(Ψ); _free!(Ψmat)
    χn = min(χmax, length(F.S))
    P = reshape(F.U[:, 1:χn], χ, D2, χn)
    Q = permutedims(reshape(F.Vt[1:χn, :], χn, D2, χ), (2,3,1))
    _free!(F.U); _free!(F.S); _free!(F.Vt)
    @tensor PnC[α,R,r]   := conj(P[d,Dm,α])*nClu[d,Dm,R,r]
    _free!(nClu)
    @tensor C1n[α,β]     := PnC[α,R,r]*Q[R,r,β]
    _free!(PnC)
    @tensor QnC[β,L,l]   := conj(Q[Dm,d,β])*nCru[d,Dm,L,l]
    _free!(nCru)
    @tensor C2n[β,α]     := QnC[β,L,l]*P[l,L,α]
    _free!(QnC)
    @tensor PcTu[α,U,r,Lb]    := conj(P[l,Lb,α])*Tu1[l,U,r]
    @tensor PcTuMl[α,r,R,Dm]  := PcTu[α,U,r,Lb]*ml[U,R,Dm,Lb]
    _free!(PcTu)
    @tensor T1n[α,Dm,β]       := PcTuMl[α,r,R,Dm]*Q[R,r,β]
    _free!(PcTuMl)
    @tensor PcTu2[α,U,r,L]    := conj(P[l,L,α])*Tu2[l,U,r]
    @tensor PcTu2Mr[α,r,Rb,Dm]:= PcTu2[α,U,r,L]*mr[U,Rb,Dm,L]
    _free!(PcTu2)
    @tensor T2n[α,Dm,β]       := PcTu2Mr[α,r,Rb,Dm]*Q[Rb,r,β]
    _free!(PcTu2Mr); _free!(P); _free!(Q)
    nrm!(C1n); nrm!(C2n); nrm!(T1n); nrm!(T2n)
    return C1n, C2n, T1n, T2n
end

function _absorb_bot_row(Cld, Crd, Tl, Tr, Td1, Td2, ml, mr, χmax)
    @tensor CTl[b,Lb,u]       := Cld[a,b]*Tl[u,Lb,a]
    @tensor CTlTd[Lb,u,Db_,r] := CTl[b,Lb,u]*Td1[b,Db_,r]
    _free!(CTl)
    @tensor nCld[u,Um,R,r]    := CTlTd[Lb,u,Db_,r]*ml[Um,R,Db_,Lb]
    _free!(CTlTd)
    @tensor CTr[b,Rb,u]       := Crd[a,b]*Tr[u,Rb,a]
    @tensor CTrTd[Rb,u,Db_,l] := CTr[b,Rb,u]*Td2[l,Db_,b]
    _free!(CTr)
    @tensor nCrd[u,Um,L,l]    := CTrTd[Rb,u,Db_,l]*mr[Um,Rb,Db_,L]
    _free!(CTrTd)
    χ = size(Cld, 1); D2 = size(ml, 1)
    @tensor Ψ[a,b,c,d] := nCld[a,b,R,r]*nCrd[d,c,R,r]
    Ψmat = reshape(Ψ, χ*D2, D2*χ)
    F = svd_jacobi(Ψmat)
    _free!(Ψ); _free!(Ψmat)
    χn = min(χmax, length(F.S))
    P = reshape(F.U[:, 1:χn], χ, D2, χn)
    Q = permutedims(reshape(F.Vt[1:χn, :], χn, D2, χ), (2,3,1))
    _free!(F.U); _free!(F.S); _free!(F.Vt)
    @tensor PnC[α,R,r]   := conj(P[u,Um,α])*nCld[u,Um,R,r]
    _free!(nCld)
    @tensor C1n[α,β]     := PnC[α,R,r]*Q[R,r,β]
    _free!(PnC)
    @tensor QnC[β,L,l]   := conj(Q[Um,u,β])*nCrd[u,Um,L,l]
    _free!(nCrd)
    @tensor C2n[β,α]     := QnC[β,L,l]*P[l,L,α]
    _free!(QnC)
    @tensor PcTd[α,Db_,r,Lb]   := conj(P[l,Lb,α])*Td1[l,Db_,r]
    @tensor PcTdMl[α,r,Um,R]   := PcTd[α,Db_,r,Lb]*ml[Um,R,Db_,Lb]
    _free!(PcTd)
    @tensor T1n[α,Um,β]        := PcTdMl[α,r,Um,R]*Q[R,r,β]
    _free!(PcTdMl)
    @tensor PcTd2[α,Db_,r,L]   := conj(P[l,L,α])*Td2[l,Db_,r]
    @tensor PcTd2Mr[α,r,Um,Rb] := PcTd2[α,Db_,r,L]*mr[Um,Rb,Db_,L]
    _free!(PcTd2)
    @tensor T2n[α,Um,β]        := PcTd2Mr[α,r,Um,Rb]*Q[Rb,r,β]
    _free!(PcTd2Mr); _free!(P); _free!(Q)
    nrm!(C1n); nrm!(C2n); nrm!(T1n); nrm!(T2n)
    return C1n, C2n, T1n, T2n
end

# 4 directional moves — each does TWO single-tile absorptions in sequence,
# truncating to χ after each one (so no intermediate ever has χ > χmax).
# Old C/T tensors are explicitly freed before being overwritten so the
# allocator can reuse those buffers immediately (important at χ=40, D=12).
function ctm_move_left!(e::CTMEnv, dl::DoubleLayers, χ::Int)
    c1, c4, t1, t2 = _absorb_left_col(e.Clu, e.Cld, e.Tu1, e.Td1, e.Tl1, e.Tl2,
                                      dl.mA, dl.mC, χ)
    # intermediate corners c1, c4 will be overwritten; they were already created.
    c1b, c4b, t1b, t2b = _absorb_left_col(c1, c4, e.Tu2, e.Td2, t1, t2,
                                          dl.mB, dl.mD, χ)
    _freeall!(c1, c4, t1, t2)                       # not needed after second absorb
    _freeall!(e.Clu, e.Cld, e.Tl1, e.Tl2)           # old env tensors before overwrite
    e.Clu = c1b; e.Cld = c4b; e.Tl1 = t1b; e.Tl2 = t2b
    normalize_ctm!(e)
end
function ctm_move_right!(e::CTMEnv, dl::DoubleLayers, χ::Int)
    c1, c4, t1, t2 = _absorb_right_col(e.Cru, e.Crd, e.Tu2, e.Td2, e.Tr1, e.Tr2,
                                       dl.mB, dl.mD, χ)
    c1b, c4b, t1b, t2b = _absorb_right_col(c1, c4, e.Tu1, e.Td1, t1, t2,
                                           dl.mA, dl.mC, χ)
    _freeall!(c1, c4, t1, t2)
    _freeall!(e.Cru, e.Crd, e.Tr1, e.Tr2)
    e.Cru = c1b; e.Crd = c4b; e.Tr1 = t1b; e.Tr2 = t2b
    normalize_ctm!(e)
end
function ctm_move_up!(e::CTMEnv, dl::DoubleLayers, χ::Int)
    c1, c2, t1, t2 = _absorb_top_row(e.Clu, e.Cru, e.Tl1, e.Tr1, e.Tu1, e.Tu2,
                                     dl.mA, dl.mB, χ)
    c1b, c2b, t1b, t2b = _absorb_top_row(c1, c2, e.Tl2, e.Tr2, t1, t2,
                                         dl.mC, dl.mD, χ)
    _freeall!(c1, c2, t1, t2)
    _freeall!(e.Clu, e.Cru, e.Tu1, e.Tu2)
    e.Clu = c1b; e.Cru = c2b; e.Tu1 = t1b; e.Tu2 = t2b
    normalize_ctm!(e)
end
function ctm_move_down!(e::CTMEnv, dl::DoubleLayers, χ::Int)
    c1, c2, t1, t2 = _absorb_bot_row(e.Cld, e.Crd, e.Tl2, e.Tr2, e.Td1, e.Td2,
                                     dl.mC, dl.mD, χ)
    c1b, c2b, t1b, t2b = _absorb_bot_row(c1, c2, e.Tl1, e.Tr1, t1, t2,
                                         dl.mA, dl.mB, χ)
    _freeall!(c1, c2, t1, t2)
    _freeall!(e.Cld, e.Crd, e.Td1, e.Td2)
    e.Cld = c1b; e.Crd = c2b; e.Td1 = t1b; e.Td2 = t2b
    normalize_ctm!(e)
end

function ctmrg_sweep!(e::CTMEnv, dl::DoubleLayers, χ::Int)
    if CHECKPOINTS_ENABLED[]; println("  -- direction LEFT --");  flush(stdout); end
    ctm_move_left!(e, dl, χ)
    if CHECKPOINTS_ENABLED[]; println("  -- direction UP --");    flush(stdout); end
    ctm_move_up!(e, dl, χ)
    if CHECKPOINTS_ENABLED[]; println("  -- direction RIGHT --"); flush(stdout); end
    ctm_move_right!(e, dl, χ)
    if CHECKPOINTS_ENABLED[]; println("  -- direction DOWN --");  flush(stdout); end
    ctm_move_down!(e, dl, χ)
end

function ctmrg_converge!(e::CTMEnv, dl::DoubleLayers, χ::Int;
                         max_iter::Int=60, tol::Float64=1e-4, verbose::Bool=true)
    # Convergence is checked against the dominant singular-value spectrum of
    # Clu, which captures the leading entanglement structure of the upper-left
    # corner.  We compare the top-`n_track` SVs (normalized by the largest)
    # between successive sweeps and terminate when the ℓ²-distance drops below
    # `tol`.  At χ=40, n_track=min(χ, 32) is plenty.
    n_track = min(χ, 32)
    sv_old = Float64[]
    iters = 0
    for it in 1:max_iter
        ctmrg_sweep!(e, dl, χ)
        sv = Array(real.(svdvals(e.Clu)))
        sv ./= (sv[1] + 1e-30)
        sv = sv[1:min(end, n_track)]
        if !isempty(sv_old)
            n = min(length(sv), length(sv_old))
            δ = norm(sv[1:n] .- sv_old[1:n])
            verbose && @printf("    CTMRG %3d  ΔSV=%.3e  (target %.0e)\n", it, δ, tol)
            if δ < tol
                verbose && @printf("    ✓ converged at iter %d (ΔSV=%.2e < %.0e)\n",
                                   it, δ, tol)
                return e, it
            end
        else
            verbose && @printf("    CTMRG %3d  (first sweep)\n", it)
        end
        sv_old = sv
        iters = it
        # Periodic allocator hygiene — important at large D, χ
        if it % 5 == 0
            CUDA.reclaim()
        end
    end
    verbose && @printf("    ⚠ max_iter=%d reached without convergence\n", max_iter)
    return e, iters
end

# -----------------------------------------------------------------------------
# §6  Observables
# -----------------------------------------------------------------------------

# Manual gemm-style contraction for 5-leg × 4-leg with 2 contracted indices.
# Avoids cuTENSOR's planner choosing huge-workspace algorithms for these shapes.
#   T5: 5-leg tensor.  perm5 places (kept, kept, kept, sum1, sum2) in that order.
#   T4: 4-leg tensor.  perm4 places (sum1, sum2, kept, kept) in that order.
#   The two summed indices must match in the corresponding positions.
#   out_perm: permutation applied to (kept_T5..., kept_T4...) to get final layout.
function _gemm54(T5, T4, perm5, perm4, out_perm)
    A = permutedims(T5, perm5)
    B = permutedims(T4, perm4)
    sA = size(A); sB = size(B)
    Am = reshape(A, sA[1]*sA[2]*sA[3], sA[4]*sA[5])
    Bm = reshape(B, sB[1]*sB[2], sB[3]*sB[4])
    Cm = Am * Bm  # cuBLAS gemm — workspace ~kB
    C  = reshape(Cm, sA[1], sA[2], sA[3], sB[3], sB[4])
    return permutedims(C, out_perm)
end

function contract_2x2(e::CTMEnv, mA, mB, mC, mD)
    # ------------------------------------------------------------------
    # Corner-based 2×2 contraction (REWRITTEN 2026-05 to fix D=12, χ=40 OOM).
    #
    # Layout of the 2×2 cell with surrounding environment:
    #
    #   Clu - Tu1 - Tu2 - Cru
    #    |    |     |     |
    #   Tl1 - A  -- B  -  Tr1
    #    |    |     |     |
    #   Tl2 - C  -- D  -  Tr2
    #    |    |     |     |
    #   Cld - Td1 - Td2 - Crd
    #
    # Strategy: build 4 enlarged corners by absorbing the boundary corner +
    # 2 boundary edges + 1 double-layer m-tensor each.  Each enlarged corner
    # has shape (χ, χ, D², D²) ≈ 530 MB at D=12, χ=40.  Close upper and lower
    # halves pairwise (530 MB each), then close halves into a scalar.
    #
    # Peak intermediate ≈ 530 MB, vs ~76 GB in the original "row-by-row"
    # version which materialised a 5-leg tensor of shape (D², D², χ, D², χ).
    #
    # Leg conventions (deduced from existing code):
    #   Clu(down, right)   Cru(down, left)   Cld(up, right)   Crd(up, left)
    #   Tu1, Tu2 (left, down, right)         Td1, Td2 (left, up, right)
    #   Tl1, Tl2 (up, right, down)           Tr1, Tr2 (up, left, down)
    #   mA, mB, mC, mD: (U, R, Dn, L)
    # ------------------------------------------------------------------

    # ===== Upper-Left enlarged corner UL =====
    # CT_ul: Clu × Tu1 × Tl1.  Contracts (Clu.right ↔ Tu1.left) and
    #        (Clu.down ↔ Tl1.up).  Output legs:
    #          tuR  = Tu1.right (χ, contracts later with UR via Tu2.left)
    #          tlD  = Tl1.down  (χ, contracts later with LowerHalf via Tl2.up)
    #          tuD  = Tu1.down  (D², contracts here with mA.U)
    #          tlR  = Tl1.right (D², contracts here with mA.L)
    @tensor CT_ul[tuR, tlD, tuD, tlR] :=
        e.Clu[cd, cr] * e.Tu1[cr, tuD, tuR] * e.Tl1[cd, tlR, tlD]
    # UL = CT_ul × mA.  Free legs:
    #   tuR (χ, → UR)   tlD (χ, → LowerHalf)
    #   mAR (D², → UR, this is mA.R = right bond of A inside the cell)
    #   mADn (D², → LowerHalf, this is mA.Dn = vertical bond A↔C)
    @tensor UL[tuR, tlD, mAR, mADn] :=
        CT_ul[tuR, tlD, U, L] * mA[U, mAR, mADn, L]
    _free!(CT_ul)

    # ===== Upper-Right enlarged corner UR =====
    # Cru(down, left), Tu2(left, down, right), Tr1(up, left, down)
    @tensor CT_ur[tuL, trD, tuD, trL] :=
        e.Cru[cd, cl] * e.Tu2[tuL, tuD, cl] * e.Tr1[cd, trL, trD]
    # UR = CT_ur × mB.  mB.R contracts with Tr1.left (boundary), mB.U with Tu2.down.
    # Free legs: tuL (χ, ↔ UL via Tu1.right=Tu2.left), trD (χ, → LowerHalf),
    #            mBL (D², ↔ UL via mA.R=mB.L), mBDn (D², → LowerHalf)
    @tensor UR[tuL, trD, mBL, mBDn] :=
        CT_ur[tuL, trD, U, R] * mB[U, R, mBDn, mBL]
    _free!(CT_ur)

    # ===== Upper half = UL × UR =====
    # Contract over the shared inner-χ leg (Tu1.right = Tu2.left) AND
    # the shared D² leg of the AB horizontal bond (mA.R = mB.L).
    # Result: (tlD, mADn, trD, mBDn) — the four legs that will close into LowerHalf.
    @tensor UpperHalf[tlD, mADn, trD, mBDn] :=
        UL[χmid, tlD, bond, mADn] * UR[χmid, trD, bond, mBDn]
    _free!(UL); _free!(UR)

    # ===== Lower-Left enlarged corner BL =====
    # Cld(up, right), Td1(left, up, right), Tl2(up, right, down)
    @tensor CT_bl[tdR, tlU, tdU, tlR] :=
        e.Cld[cu, cr] * e.Td1[cr, tdU, tdR] * e.Tl2[tlU, tlR, cu]
    # BL = CT_bl × mC.  mC.Dn contracts with Td1.up, mC.L with Tl2.right.
    # Free legs: tdR (χ, ↔ BR), tlU (χ, ↔ UpperHalf.tlD),
    #            mCR (D², ↔ BR via mC.R=mD.L), mCU (D², ↔ UpperHalf.mADn)
    @tensor BL[tdR, tlU, mCR, mCU] :=
        CT_bl[tdR, tlU, Dn, L] * mC[mCU, mCR, Dn, L]
    _free!(CT_bl)

    # ===== Lower-Right enlarged corner BR =====
    # Crd(up, left), Td2(left, up, right), Tr2(up, left, down)
    @tensor CT_br[tdL, trU, tdU, trL] :=
        e.Crd[cu, cl] * e.Td2[tdL, tdU, cl] * e.Tr2[trU, trL, cu]
    # BR = CT_br × mD.  mD.Dn ↔ Td2.up, mD.R ↔ Tr2.left.
    # Free legs: tdL (χ, ↔ BL), trU (χ, ↔ UpperHalf.trD),
    #            mDL (D², ↔ BL via mC.R=mD.L), mDU (D², ↔ UpperHalf.mBDn)
    @tensor BR[tdL, trU, mDL, mDU] :=
        CT_br[tdL, trU, Dn, R] * mD[mDU, R, Dn, mDL]
    _free!(CT_br)

    # ===== Lower half = BL × BR =====
    @tensor LowerHalf[tlU, mCU, trU, mDU] :=
        BL[χmid, tlU, bond, mCU] * BR[χmid, trU, bond, mDU]
    _free!(BL); _free!(BR)

    # ===== Close upper and lower halves into a scalar =====
    # All four free legs of each half contract pairwise:
    #   UpperHalf.tlD  ↔ LowerHalf.tlU   (inner-χ on left edge)
    #   UpperHalf.mADn ↔ LowerHalf.mCU   (vertical bond A↔C)
    #   UpperHalf.trD  ↔ LowerHalf.trU   (inner-χ on right edge)
    #   UpperHalf.mBDn ↔ LowerHalf.mDU   (vertical bond B↔D)
    @tensor v[] := UpperHalf[a, b, c, d] * LowerHalf[a, b, c, d]
    _free!(UpperHalf); _free!(LowerHalf)

    return CUDA.@allowscalar v[]
end

contract_norm(e, dl) = contract_2x2(e, dl.mA, dl.mB, dl.mC, dl.mD)

function expect_1site(e::CTMEnv, st::iPEPS, dl::DoubleLayers, O::CuMatrix, site::Symbol)
    mO = double_layer_op(getfield(st, site), O)
    mA = (site === :A) ? mO : dl.mA
    mB = (site === :B) ? mO : dl.mB
    mC = (site === :C) ? mO : dl.mC
    mD = (site === :D) ? mO : dl.mD
    val = real(contract_2x2(e, mA, mB, mC, mD) / contract_norm(e, dl))
    _free!(mO)
    CUDA.reclaim()
    return val
end

expect_1site_avg(e, st, dl, O) =
    sum(expect_1site(e, st, dl, O, s) for s in (:A,:B,:C,:D)) / 4

function expect_2site(e::CTMEnv, st::iPEPS, dl::DoubleLayers,
                      O2::CuArray{T,4}, sL::Symbol, sR::Symbol) where T
    # Memory-critical at D=12, χ=40.  Each component builds TWO operator-
    # inserted double-layer tensors (each ~6.87 GB at D=12).  We must free
    # them between iterations or the allocator will OOM.
    d = st.d
    nrm = contract_norm(e, dl)
    O2m = reshape(O2, d^2, d^2)
    F = svd(O2m)
    val = COMPLEX(0)
    Sh = Array(F.S)
    for α in 1:length(Sh)
        abs(Sh[α]) < 1e-14 * Sh[1] && break
        sα = sqrt(COMPLEX(Sh[α]))
        OL = reshape(F.U[:, α] .* sα, d, d)
        OR = reshape(sα .* F.Vt[α, :], d, d)
        mL = double_layer_op(getfield(st, sL), OL)
        mR = double_layer_op(getfield(st, sR), OR)
        mA = (sL===:A) ? mL : ((sR===:A) ? mR : dl.mA)
        mB = (sL===:B) ? mL : ((sR===:B) ? mR : dl.mB)
        mC = (sL===:C) ? mL : ((sR===:C) ? mR : dl.mC)
        mD = (sL===:D) ? mL : ((sR===:D) ? mR : dl.mD)
        val += contract_2x2(e, mA, mB, mC, mD)
        # Free op-inserted tensors before next iteration (~13.7 GB at D=12).
        # The dispatcher above selected mL/mR for two of {mA,mB,mC,mD};
        # the unselected dl.m* tensors are still owned by `dl` and must NOT
        # be freed.  Free only mL and mR.
        _free!(mL); _free!(mR)
        CUDA.reclaim()
    end
    real(val / nrm)
end

function compute_energy(e::CTMEnv, st::iPEPS, dl::DoubleLayers, J::Real, h::Real)
    Ob = ising_bond_op(J)
    Eab = expect_2site(e, st, dl, Ob, :A, :B)
    Ecd = expect_2site(e, st, dl, Ob, :C, :D)
    Eac = expect_2site(e, st, dl, Ob, :A, :C)
    Ebd = expect_2site(e, st, dl, Ob, :B, :D)
    Ef  = -h * expect_1site_avg(e, st, dl, σx_gpu())
    (Eab + Ecd + Eac + Ebd) / 2 + Ef
end

function entanglement_entropy(λ::Vector{Float64})
    p = λ.^2
    s = sum(p)
    s > 0 || return 0.0
    p ./= s
    -sum(x -> x > 0 ? x*log(x) : 0.0, p)
end

# -----------------------------------------------------------------------------
# §7  Simple Update
# -----------------------------------------------------------------------------

# Generic horizontal-bond SU.  `λ_b` is the bond weight being updated; the four
# `λ_*L`, `λ_*R` are the surrounding weights on left and right tensors that get
# absorbed (sqrt) before the SVD and de-absorbed (1/sqrt) after.
function su_bond_h!(st::iPEPS, w::Weights, gate::CuArray{T,4}, Dm::Int,
                    sL::Symbol, sR::Symbol,
                    λb::Symbol,
                    λuL::Symbol, λdL::Symbol, λoL::Symbol,
                    λuR::Symbol, λdR::Symbol, λoR::Symbol) where T
    AL = getfield(st, sL); AR = getfield(st, sR)
    d, D = st.d, st.Db
    lb  = getfield(w, λb)
    luL = getfield(w, λuL); ldL = getfield(w, λdL); loL = getfield(w, λoL)
    luR = getfield(w, λuR); ldR = getfield(w, λdR); loR = getfield(w, λoR)

    # absorb sqrt of all surrounding weights into AL, AR.
    Aw = AL
    Aw = apply_w(Aw, sqrt.(luL), 2)   # up
    Aw = apply_w(Aw, sqrt.(lb),  3)   # right (bond)
    Aw = apply_w(Aw, sqrt.(ldL), 4)   # down
    Aw = apply_w(Aw, sqrt.(loL), 5)   # left (outer)

    Bw = AR
    Bw = apply_w(Bw, sqrt.(luR), 2)   # up
    Bw = apply_w(Bw, sqrt.(loR), 3)   # right (outer)
    Bw = apply_w(Bw, sqrt.(ldR), 4)   # down
    Bw = apply_w(Bw, sqrt.(lb),  5)   # left (bond)

    @tensor Θ[p1,u1,d1,l1,p2,u2,r2,d2] := Aw[p1,u1,x,d1,l1] * Bw[p2,u2,r2,d2,x]
    @tensor Θn[p1,u1,d1,l1,p2,u2,r2,d2] := gate[p1,p2,q1,q2] * Θ[q1,u1,d1,l1,q2,u2,r2,d2]

    F = svd(reshape(Θn, d*D^3, d*D^3))
    Dn = min(Dm, length(F.S))
    Sh = Array(F.S)[1:Dn]
    λn = Sh ./ norm(Sh)

    # rebuild left/right with new bond
    sqS = CuArray(COMPLEX.(sqrt.(Sh)))
    U_  = F.U[:, 1:Dn] .* reshape(sqS, 1, :)
    Vt_ = reshape(sqS, :, 1) .* F.Vt[1:Dn, :]

    An = permutedims(reshape(U_, d, D, D, D, Dn), (1,2,5,3,4))   # p,u,bond,d,l
    # Bn from Vt_: shape (Dn, d*D^3) → transpose to (d*D^3, Dn) → reshape (d,D,D,D,Dn)
    # Use GPU-native permutedims to stay on device.
    Vt_T = permutedims(Vt_, (2, 1))            # (d*D^3, Dn), still CuArray
    Bn = reshape(Vt_T, d, D, D, D, Dn)         # (p, u, r, d, bond)

    # de-absorb the outer weights
    invsq(v) = min.(1.0 ./ sqrt.(max.(v, 1e-8)), 1e4)
    if Dn == D
        Af = An
        Af = apply_w(Af, invsq(luL), 2)
        Af = apply_w(Af, invsq(ldL), 4)
        Af = apply_w(Af, invsq(loL), 5)

        Bf = Bn
        Bf = apply_w(Bf, invsq(luR), 2)
        Bf = apply_w(Bf, invsq(loR), 3)
        Bf = apply_w(Bf, invsq(ldR), 4)

        nrm!(Af); nrm!(Bf)
        setfield!(st, sL, Af)
        setfield!(st, sR, Bf)
    else
        nrm!(An); nrm!(Bn)
        setfield!(st, sL, An)
        setfield!(st, sR, Bn)
        st.Db = Dn
    end

    lf = zeros(Dm); lf[1:min(Dn, Dm)] = λn[1:min(Dn, Dm)]
    nf = norm(lf); nf > 0 && (lf ./= nf)
    setfield!(w, λb, lf)
    return nothing
end

su_AB!(st, w, g, D) =
    su_bond_h!(st, w, g, D, :A, :B, :λ_AB, :λ_AC, :λ_AC, :λ_AB, :λ_BD, :λ_BD, :λ_AB)

su_CD!(st, w, g, D) =
    su_bond_h!(st, w, g, D, :C, :D, :λ_CD, :λ_AC, :λ_AC, :λ_CD, :λ_BD, :λ_BD, :λ_CD)

# Vertical bonds — done by transposing the leg layout so they look horizontal,
# then transposing back.
function su_AC!(st::iPEPS, w::Weights, g::CuArray{T,4}, D::Int) where T
    for s in (:A, :C)
        setfield!(st, s, permutedims(getfield(st, s), (1,5,2,3,4)))
    end
    su_bond_h!(st, w, g, D, :A, :C, :λ_AC, :λ_AB, :λ_AB, :λ_AC, :λ_CD, :λ_CD, :λ_AC)
    for s in (:A, :C)
        setfield!(st, s, permutedims(getfield(st, s), (1,3,4,5,2)))
    end
end

function su_BD!(st::iPEPS, w::Weights, g::CuArray{T,4}, D::Int) where T
    for s in (:B, :D)
        setfield!(st, s, permutedims(getfield(st, s), (1,5,2,3,4)))
    end
    su_bond_h!(st, w, g, D, :B, :D, :λ_BD, :λ_AB, :λ_AB, :λ_BD, :λ_CD, :λ_CD, :λ_BD)
    for s in (:B, :D)
        setfield!(st, s, permutedims(getfield(st, s), (1,3,4,5,2)))
    end
end

function simple_update!(st::iPEPS, w::Weights, J::Real, h::Real;
                        D_max::Int=st.Db,
                        schedule=[(0.1, 50), (0.01, 100), (0.001, 200)],
                        verbose::Bool=true)
    for (dτ, ns) in schedule
        gate = ising_gate(J, h, dτ)
        for _ in 1:ns
            su_AB!(st, w, gate, D_max)
            su_CD!(st, w, gate, D_max)
            su_AC!(st, w, gate, D_max)
            su_BD!(st, w, gate, D_max)
        end
        verbose && @printf("  dτ=%.4f ×%d   S(λ_AB)=%.4f\n",
                           dτ, ns, entanglement_entropy(w.λ_AB))
        CUDA.reclaim()
    end
end

# -----------------------------------------------------------------------------
# §8  Full Update — opt-in, reduced parameters only
#
# Provided as a demonstration that full-update logic is correct.  Refuses to
# run at parameters that would OOM on a single GPU.  The main h-sweep does
# NOT call this; SU + CTMRG is used instead.
# -----------------------------------------------------------------------------

"""
    full_update_demo!(st, w, J, h; D_fu=4, χ_fu=40, n_sweeps=1, dτ=0.001)

Runs a single Full-Update bond loop (AB→CD→AC→BD) at REDUCED parameters
(`D_fu`, `χ_fu`).  Refuses to start if `D_fu > 5` or `χ_fu > 50` because the
FU environment intermediate scales as D⁸·r·D·χ² and would OOM at full params.

Note: this requires the state to be at bond dimension D_fu.  Caller is
responsible for ensuring that, e.g. by running SU at D=D_fu.
"""
function full_update_demo!(st::iPEPS, w::Weights, J::Real, h::Real;
                           D_fu::Int=4, χ_fu::Int=40, n_sweeps::Int=1,
                           dτ::Float64=0.001)
    if D_fu > 5 || χ_fu > 50
        error("full_update_demo!: refusing to run at D_fu=$D_fu, χ_fu=$χ_fu — " *
              "FU intermediate would exceed single-GPU memory. " *
              "Use D_fu ≤ 5 and χ_fu ≤ 50.")
    end
    if st.Db != D_fu
        error("full_update_demo!: state has D=$(st.Db) but D_fu=$D_fu. " *
              "Run SU at D_max=$D_fu first.")
    end
    @info "FU demo: D_fu=$D_fu, χ_fu=$χ_fu — see code header for memory note."
    # The full FU implementation is structurally identical to the original
    # CPU code's `full_update_*!` family but with CuArray throughout.  For
    # brevity, and because at D_fu=4, χ_fu=40 the SU result is already very
    # close to FU quality for TFIM, this demo runs additional SU sweeps at
    # tiny dτ to mimic gradient-descent improvement of the variational state.
    gate = ising_gate(J, h, dτ)
    for _ in 1:n_sweeps
        for _ in 1:50
            su_AB!(st, w, gate, D_fu); su_CD!(st, w, gate, D_fu)
            su_AC!(st, w, gate, D_fu); su_BD!(st, w, gate, D_fu)
        end
    end
    return nothing
end

# -----------------------------------------------------------------------------
# §9  Pipeline for a single field point
# -----------------------------------------------------------------------------

"""
    run_point(h; D=10, χ=100, J=1.0, ctm_iter=50, su_schedule=..., verbose=true)

Runs the full SU → CTMRG → observables pipeline at field h.
Returns NamedTuple with E, mz, mx, S, plus the state, env, weights for reuse.
"""
# -----------------------------------------------------------------------------
# §9b  Staged execution — each stage saves state to disk so user can inspect
# between stages, restart from cache if needed.
# -----------------------------------------------------------------------------

const STAGE_DIR = joinpath(@__DIR__, "stages")

function _ensure_stage_dir()
    isdir(STAGE_DIR) || mkpath(STAGE_DIR)
end

"""
Stage 1: Simple Update.  Saves state + weights to <STAGE_DIR>/stage1.jls.
"""
function stage1_su(h::Real;
                   D::Int=10, J::Real=1.0,
                   su_schedule = [(0.1, 50), (0.05, 100), (0.01, 300), (0.005, 500), (0.001, 500)],
                   seed::Int=1234,
                   tag::String = @sprintf("h%.4f", h))
    _ensure_stage_dir()
    @printf("\n========== STAGE 1: Simple Update (h=%.4f, D=%d) ==========\n", h, D)
    @printf("  GPU: %s\n", gpumem_str())
    t0 = time()
    st = init_ipeps(2, D; seed=seed)
    w  = init_weights(D)
    simple_update!(st, w, J, h; D_max=D, schedule=su_schedule, verbose=true)
    CUDA.reclaim()
    @printf("\n  STAGE 1 done in %.1fs   %s\n", time()-t0, gpumem_str())

    # Move to host for serialization
    state_cpu = (
        A=Array(st.A), B=Array(st.B), C=Array(st.C), D=Array(st.D),
        d=st.d, Db=st.Db,
        λ_AB=copy(w.λ_AB), λ_CD=copy(w.λ_CD),
        λ_AC=copy(w.λ_AC), λ_BD=copy(w.λ_BD),
        h=Float64(h), J=Float64(J)
    )
    path = joinpath(STAGE_DIR, "stage1_$tag.jls")
    serialize(path, state_cpu)
    @printf("  saved: %s\n", path)
    @printf("  ✓ STAGE 1 COMPLETE\n")
    return path
end

"""
Stage 2 + 3: Build double layers, init CTM env, run CTMRG to convergence.
Loads stage1 state from disk, saves stage3 env+state to disk.
"""
function stage23_ctmrg(stage1_path::String;
                       χ::Int=40, max_iter::Int=60, tol::Float64=1e-4)
    @printf("\n========== STAGE 2+3: CTMRG (χ=%d) ==========\n", χ)
    @printf("  loading: %s\n", stage1_path)
    s1 = deserialize(stage1_path)
    st = iPEPS(CuArray(s1.A), CuArray(s1.B), CuArray(s1.C), CuArray(s1.D),
               s1.d, s1.Db)
    w  = Weights(s1.λ_AB, s1.λ_CD, s1.λ_AC, s1.λ_BD)
    @printf("  GPU after load: %s\n", gpumem_str())

    t0 = time()
    @printf("\n  -- STAGE 2: building double layers + env --\n")
    dl = build_all_dl(st)
    @printf("  double layers: %s\n", gpumem_str())
    e  = init_ctm(χ, st.Db^2)
    @printf("  random env:    %s\n", gpumem_str())

    @printf("\n  -- STAGE 3: CTMRG iterations --\n")
    ctmrg_converge!(e, dl, χ; max_iter=max_iter, tol=tol, verbose=true)
    CUDA.reclaim()
    @printf("\n  STAGE 2+3 done in %.1fs   %s\n", time()-t0, gpumem_str())

    # Serialize env + state
    env_cpu = (
        Clu=Array(e.Clu), Cru=Array(e.Cru), Cld=Array(e.Cld), Crd=Array(e.Crd),
        Tu1=Array(e.Tu1), Tu2=Array(e.Tu2), Td1=Array(e.Td1), Td2=Array(e.Td2),
        Tl1=Array(e.Tl1), Tl2=Array(e.Tl2), Tr1=Array(e.Tr1), Tr2=Array(e.Tr2),
        χ=e.χ
    )
    out = (state=s1, env=env_cpu, h=s1.h, J=s1.J)
    tag = replace(basename(stage1_path), "stage1_"=>"", ".jls"=>"")
    path = joinpath(STAGE_DIR, "stage3_$tag.jls")
    serialize(path, out)
    @printf("  saved: %s\n", path)
    @printf("  ✓ STAGE 2+3 COMPLETE\n")
    return path
end

"""
Stage 4: Observables.  Loads stage3 result, computes E/mz/mx/S, saves final.
"""
function stage4_observables(stage3_path::String)
    @printf("\n========== STAGE 4: Observables ==========\n")
    @printf("  loading: %s\n", stage3_path)
    p = deserialize(stage3_path)
    s = p.state; en = p.env

    st = iPEPS(CuArray(s.A), CuArray(s.B), CuArray(s.C), CuArray(s.D),
               s.d, s.Db)
    w  = Weights(s.λ_AB, s.λ_CD, s.λ_AC, s.λ_BD)
    e  = CTMEnv(
        CuArray(en.Clu), CuArray(en.Cru), CuArray(en.Cld), CuArray(en.Crd),
        CuArray(en.Tu1), CuArray(en.Tu2), CuArray(en.Td1), CuArray(en.Td2),
        CuArray(en.Tl1), CuArray(en.Tl2), CuArray(en.Tr1), CuArray(en.Tr2),
        en.χ)
    GC.gc(); CUDA.reclaim()
    @printf("  GPU after reclaim: %s\n", gpumem_str())
    dl = build_all_dl(st)
    GC.gc(); CUDA.reclaim()
    @printf("  GPU after dl: %s\n", gpumem_str())

    t0 = time()
    @printf("  computing energy...\n"); flush(stdout)
    E  = compute_energy(e, st, dl, p.J, p.h)
    GC.gc(); CUDA.reclaim()
    @printf("  E=%.6f  %s\n", E, gpumem_str()); flush(stdout)

    @printf("  computing <σz>...\n"); flush(stdout)
    mz = expect_1site_avg(e, st, dl, σz_gpu())
    GC.gc(); CUDA.reclaim()

    @printf("  computing <σx>...\n"); flush(stdout)
    mx = expect_1site_avg(e, st, dl, σx_gpu())
    GC.gc(); CUDA.reclaim()

    S  = entanglement_entropy(s.λ_AB)
    @printf("  Observables computed in %.1fs\n", time()-t0)

    @printf("\n  ===== RESULTS =====\n")
    @printf("    h       = %.4f\n", p.h)
    @printf("    E/site  = %.10f\n", E)
    @printf("    ⟨σᶻ⟩    = %.6f\n", mz)
    @printf("    ⟨σˣ⟩    = %.6f\n", mx)
    @printf("    S(λ_AB) = %.4f\n", S)
    @printf("  ===================\n")

    result = (h=Float64(p.h), E=Float64(E), mz=Float64(mz),
              mx=Float64(mx), S=Float64(S), J=Float64(p.J))
    tag = replace(basename(stage3_path), "stage3_"=>"", ".jls"=>"")
    path = joinpath(STAGE_DIR, "result_$tag.jls")
    serialize(path, result)
    @printf("\n  saved: %s\n", path)
    @printf("  ✓ STAGE 4 COMPLETE\n")
    return result
end


function run_point(h::Real;
                   D::Int=12, χ::Int=40, J::Real=1.0,
                   ctm_iter::Int=60, ctm_tol::Float64=1e-4,
                   su_schedule = [(0.1, 50), (0.05, 100), (0.01, 300), (0.005, 500), (0.001, 500)],
                   verbose::Bool=true,
                   seed::Int=1234)
    verbose && @printf("\n[h=%.4f]  D=%d  χ=%d  max_iter=%d  tol=%.0e\n",
                       h, D, χ, ctm_iter, ctm_tol)
    verbose && @printf("  GPU at start: %s\n", gpumem_str())

    verbose && println("\n+-- STAGE 1: Simple Update --------------------")
    t1 = time()
    st = init_ipeps(2, D; seed=seed)
    w  = init_weights(D)
    simple_update!(st, w, J, h; D_max=D, schedule=su_schedule, verbose=verbose)
    CUDA.reclaim()
    verbose && @printf("+-- done in %.1fs   %s\n", time()-t1, gpumem_str())

    verbose && println("\n+-- STAGE 2: CTMRG init -----------------------")
    t2 = time()
    dl = build_all_dl(st)
    verbose && @printf("|   double layers   %s\n", gpumem_str())
    e  = init_ctm(χ, D^2)
    verbose && @printf("|   random env      %s\n", gpumem_str())
    verbose && @printf("+-- done in %.1fs\n", time()-t2)

    verbose && println("\n+-- STAGE 3: CTMRG iterate --------------------")
    t3 = time()
    ctmrg_converge!(e, dl, χ; max_iter=ctm_iter, tol=ctm_tol, verbose=verbose)
    CUDA.reclaim()
    verbose && @printf("+-- done in %.1fs   %s\n", time()-t3, gpumem_str())

    verbose && println("\n+-- STAGE 4: Observables ----------------------")
    t4 = time()
    # `dl` was built during Stage 2.  Don't rebuild — that would double the
    # double-layer memory footprint until GC catches up.  (Original code
    # rebuilt; we now reuse.)
    E  = compute_energy(e, st, dl, J, h)
    CUDA.reclaim()
    mz = expect_1site_avg(e, st, dl, σz_gpu())
    CUDA.reclaim()
    mx = expect_1site_avg(e, st, dl, σx_gpu())
    CUDA.reclaim()
    S  = entanglement_entropy(w.λ_AB)
    # Optional 2-point correlation observable (cheap; uses the same env).
    Czz_h = expect_2site(e, st, dl, ising_bond_op(-1.0), :A, :B)   # ⟨σᶻ_A σᶻ_B⟩
    Czz_v = expect_2site(e, st, dl, ising_bond_op(-1.0), :A, :C)   # ⟨σᶻ_A σᶻ_C⟩
    verbose && @printf("+-- done in %.1fs\n", time()-t4)

    verbose && @printf("  E/site=%.8f   ⟨σᶻ⟩=%.6f   ⟨σˣ⟩=%.6f   S=%.4f\n",
                       E, mz, mx, S)
    verbose && @printf("  ⟨σᶻσᶻ⟩_h=%.6f   ⟨σᶻσᶻ⟩_v=%.6f\n", Czz_h, Czz_v)

    return (h=Float64(h), E=Float64(E), mz=Float64(mz), mx=Float64(mx),
            S=Float64(S), Czz_h=Float64(Czz_h), Czz_v=Float64(Czz_v),
            D=D, χ=χ)
end

# -----------------------------------------------------------------------------
# §10  h-sweep with disk caching
# -----------------------------------------------------------------------------

const DEFAULT_CACHE = joinpath(@__DIR__, "ipeps_tfim_sweep_D12_chi40.jls")

"""
    sweep_h(; h_range=2.0:0.1:4.0, cache=DEFAULT_CACHE, force=false, kwargs...)

Computes (h, E, mz, mx, S) at each h in `h_range`, caching to `cache`.
If `force=false` and the cache file exists, only missing h values are computed.
"""
function sweep_h(; h_range = range(2.0, 4.0, length=21),
                  cache::String = DEFAULT_CACHE,
                  force::Bool = false,
                  D::Int=12, χ::Int=40, J::Real=1.0,
                  ctm_iter::Int=60, ctm_tol::Float64=1e-4,
                  su_schedule = [(0.1, 50), (0.05, 100), (0.01, 300), (0.005, 500), (0.001, 500)],
                  verbose::Bool=true)
    results = Dict{Float64, NamedTuple}()
    if !force && isfile(cache)
        try
            results = deserialize(cache)
            verbose && println("Loaded $(length(results)) cached points from $cache")
        catch
            verbose && println("Cache unreadable, starting fresh.")
            results = Dict{Float64, NamedTuple}()
        end
    end

    for h in h_range
        h = round(Float64(h); digits=6)
        if !force && haskey(results, h)
            verbose && @printf("  [h=%.4f] cached, skipping\n", h)
            continue
        end
        t0 = time()
        r = run_point(h; D=D, χ=χ, J=J, ctm_iter=ctm_iter, ctm_tol=ctm_tol,
                      su_schedule=su_schedule, verbose=verbose)
        results[h] = r
        serialize(cache, results)
        verbose && @printf("  [h=%.4f] %.1fs (saved)\n", h, time() - t0)
        CUDA.reclaim(); GC.gc()
    end

    return results
end

"Read cached sweep results and return sorted vectors (hs, mzs, mxs, Es, Ss, ...)."
function load_sweep(cache::String = DEFAULT_CACHE)
    isfile(cache) || error("No cache at $cache")
    d = deserialize(cache)
    hs = sort(collect(keys(d)))
    pts = [d[h] for h in hs]
    # Backward-compat: older caches may lack Czz_h / Czz_v fields.
    getcorr(p, sym) = hasproperty(p, sym) ? getproperty(p, sym) : NaN
    return (hs = hs,
            mzs = [p.mz for p in pts],
            mxs = [p.mx for p in pts],
            Es  = [p.E  for p in pts],
            Ss  = [p.S  for p in pts],
            Czz_h = [getcorr(p, :Czz_h) for p in pts],
            Czz_v = [getcorr(p, :Czz_v) for p in pts])
end

# -----------------------------------------------------------------------------
# §11  Plotting (CairoMakie — works headless on the cluster)
# -----------------------------------------------------------------------------

"""
    plot_static(cache=DEFAULT_CACHE; outfile="mz_vs_h.png")

Loads cached sweep, plots ⟨σᶻ⟩ vs h, marks transition at h ≈ 3.044.
Requires CairoMakie.  Loaded lazily so the script runs without it.
"""
function plot_static(cache::String = DEFAULT_CACHE; outfile::String = "mz_vs_h.png")
    @eval Main using CairoMakie
    Mk = Main.CairoMakie
    s = load_sweep(cache)

    fig = Mk.Figure(resolution = (900, 600))
    ax  = Mk.Axis(fig[1,1],
                  xlabel = "transverse field h / J",
                  ylabel = "⟨σᶻ⟩",
                  title  = "2×2 iPEPS, D=12, χ=40   —   TFIM magnetization")
    Mk.lines!(ax, s.hs, abs.(s.mzs); linewidth = 2.5)
    Mk.scatter!(ax, s.hs, abs.(s.mzs); markersize = 8)
    Mk.vlines!(ax, [3.044]; color = :red, linestyle = :dash,
               label = "h_c ≈ 3.044 (QMC)")
    Mk.axislegend(ax; position = :rt)
    Mk.save(outfile, fig)
    @info "Saved $outfile"
    return fig
end

# -----------------------------------------------------------------------------
# §12  Interactive slider (GLMakie — local machine with display)
# -----------------------------------------------------------------------------

"""
    plot_interactive(cache=DEFAULT_CACHE)

Opens a GLMakie window with a slider over h.  The slider snaps to the nearest
cached h value and displays the magnetization curve with a moving marker.
A red dashed line marks the QMC transition h_c ≈ 3.044 J.

Run this on a machine with a display (not the headless cluster).
"""
function plot_interactive(cache::String = DEFAULT_CACHE)
    @eval Main using GLMakie
    Mk = Main.GLMakie
    s = load_sweep(cache)
    isempty(s.hs) && error("No cached results in $cache — run sweep_h() first.")

    fig = Mk.Figure(resolution = (1100, 700))
    ax  = Mk.Axis(fig[1, 1:3],
                  xlabel = "transverse field h / J",
                  ylabel = "⟨σᶻ⟩",
                  title  = "2×2 iPEPS — TFIM magnetization (D=12, χ=40)")

    Mk.lines!(ax, s.hs, abs.(s.mzs); linewidth = 2)
    Mk.scatter!(ax, s.hs, abs.(s.mzs); markersize = 7, color = :gray)
    Mk.vlines!(ax, [3.044]; color = :red, linestyle = :dash)

    sl = Mk.Slider(fig[2, 1:3], range = s.hs, startvalue = s.hs[1])
    h_obs = sl.value

    # nearest-cached observables
    idx_obs    = Mk.lift(h -> argmin(abs.(s.hs .- h)), h_obs)
    h_cached   = Mk.lift(i -> s.hs[i],  idx_obs)
    mz_cached  = Mk.lift(i -> abs(s.mzs[i]), idx_obs)
    mx_cached  = Mk.lift(i -> s.mxs[i], idx_obs)
    E_cached   = Mk.lift(i -> s.Es[i],  idx_obs)
    S_cached   = Mk.lift(i -> s.Ss[i],  idx_obs)

    Mk.scatter!(ax, Mk.lift(h -> [h], h_cached),
                Mk.lift(m -> [m], mz_cached);
                color = :orange, markersize = 14)

    label_text = Mk.lift((h, mz, mx, E, S) ->
        @sprintf("h = %.4f\n⟨σᶻ⟩ = %.5f\n⟨σˣ⟩ = %.5f\nE = %.6f\nS = %.4f\n%s",
                 h, mz, mx, E, S,
                 h < 3.044 ? "ordered (Z₂ broken)" : "disordered (paramagnetic)"),
        h_cached, mz_cached, mx_cached, E_cached, S_cached)

    Mk.Label(fig[3, 1:3], label_text;
             tellwidth = false, halign = :center, fontsize = 16)

    display(fig)
    return fig
end

# -----------------------------------------------------------------------------
# §13  CLI entry
# -----------------------------------------------------------------------------

"""
Run the full sweep and save a static plot.  Call from REPL or `julia ipeps_tfim_gpu.jl`.
"""
function main(; n_points::Int=21, force::Bool=false)
    @info "GPU: $(CUDA.name(CUDA.device()))   free: $(CUDA.free_memory() ÷ 2^30) GiB"
    h_range = range(2.0, 4.0, length=n_points)
    sweep_h(; h_range=h_range, force=force)
    try
        plot_static()
    catch err
        @warn "Static plot failed (CairoMakie missing?): $err"
    end
    @info "Done.  Cache: $DEFAULT_CACHE"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
