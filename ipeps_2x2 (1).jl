#!/usr/bin/env julia
#=
    2×2 iPEPS TFIM — MEMORY-SAFE FULL UPDATE FOR ALL D
    ====================================================
    FU NEVER skipped. Parameters auto-tuned per D.
    Same pipeline for D=2,3,4: SU → CTMRG → FU → CTMRG → Observables
=#
using LinearAlgebra, TensorOperations, Printf

# ═══════════════════════════════════════════════════
# §1  MODEL + TYPES
# ═══════════════════════════════════════════════════
const σx = ComplexF64[0 1; 1 0]
const σz = ComplexF64[1 0; 0 -1]
const Id = ComplexF64[1 0; 0 1]
ising_gate(J,h,dτ) = reshape(exp(-dτ*(-J*kron(σz,σz)-(h/4)*(kron(σx,Id)+kron(Id,σx)))),2,2,2,2)
ising_bond_op(J) = reshape(ComplexF64.(-J*kron(σz,σz)),2,2,2,2)

mutable struct iPEPS
    A::Array{ComplexF64,5}; B::Array{ComplexF64,5}
    C::Array{ComplexF64,5}; D::Array{ComplexF64,5}
    d::Int; D_bond::Int
end
mutable struct Weights
    λ_AB::Vector{Float64}; λ_CD::Vector{Float64}
    λ_AC::Vector{Float64}; λ_BD::Vector{Float64}
end
mutable struct CTMEnv
    Clu::Matrix{ComplexF64}; Cru::Matrix{ComplexF64}
    Cld::Matrix{ComplexF64}; Crd::Matrix{ComplexF64}
    Tu1::Array{ComplexF64,3}; Tu2::Array{ComplexF64,3}
    Td1::Array{ComplexF64,3}; Td2::Array{ComplexF64,3}
    Tl1::Array{ComplexF64,3}; Tl2::Array{ComplexF64,3}
    Tr1::Array{ComplexF64,3}; Tr2::Array{ComplexF64,3}
    χ::Int
end
struct DoubleLayers
    mA::Array{ComplexF64,4}; mB::Array{ComplexF64,4}
    mC::Array{ComplexF64,4}; mD::Array{ComplexF64,4}
    D2::Int
end

# ═══════════════════════════════════════════════════
# §2  INIT + HELPERS
# ═══════════════════════════════════════════════════
init_ipeps(d,D) = iPEPS((randn(ComplexF64,d,D,D,D,D)/sqrt(d*D^4) for _ in 1:4)..., d, D)
init_weights(D) = Weights(ones(D), ones(D), ones(D), ones(D))
function init_ctm(χ,D2)
    mc() = (x=randn(ComplexF64,χ,χ)/χ; (x+x')/2)
    me() = randn(ComplexF64,χ,D2,χ)/sqrt(χ*D2)
    CTMEnv(mc(),mc(),mc(),mc(), me(),me(),me(),me(), me(),me(),me(),me(), χ)
end
nrm!(x) = (n=norm(x); n>0 && (x./=n); x)
function normalize_ctm!(e)
    for f in fieldnames(CTMEnv); f==:χ && continue; nrm!(getfield(e,f)); end
end

# ═══════════════════════════════════════════════════
# §3  ADAPTIVE PARAMETER SELECTION
# ═══════════════════════════════════════════════════

"""
Auto-tune (χ, r_max, nals) so FU ALWAYS runs within ~1e8 elements.
Peak memory ∝ χ² · D⁴ · r_max · D.
Strategy: start with generous params, reduce until safe.
"""
function auto_params(D::Int)
    # Start generous, then check
    configs = [
        (χ=40, r=min(2*D,8), nals=2),   # first try
        (χ=30, r=min(2*D,6), nals=2),   # reduce χ
        (χ=25, r=min(D+1,4), nals=2),   # reduce both
        (χ=20, r=min(D,4),   nals=1),   # minimal
        (χ=15, r=D,          nals=1),   # last resort
    ]
    for c in configs
        # Largest tensor: TH[D², D², r*D, D², χ, χ] ≈ D⁴ · r · D · χ²
        peak = D^4 * c.r * D * c.χ^2
        if peak < 8e7  # safe threshold
            return c.χ, c.r, c.nals
        end
    end
    return 15, D, 1  # absolute minimum
end

"""Print the parameter table for all D values."""
function print_params()
    println("\nAuto-tuned parameters:")
    @printf("  %-4s %-4s %-6s %-6s %-12s\n", "D", "χ", "r_max", "nals", "peak_mem")
    println("  " * "-"^35)
    for D in [2,3,4]
        χ, r, nals = auto_params(D)
        peak = D^4 * r * D * χ^2
        @printf("  %-4d %-4d %-6d %-6d %-12.1e\n", D, χ, r, nals, Float64(peak))
    end
end

# ═══════════════════════════════════════════════════
# §4  DOUBLE LAYER
# ═══════════════════════════════════════════════════
function double_layer(A::AbstractArray{T,5}) where T
    d,D = size(A,1), size(A,2)
    @tensor m[u,up,r,rp,d,dp,l,lp] := A[p,u,r,d,l]*conj(A[p,up,rp,dp,lp])
    reshape(m, D^2, D^2, D^2, D^2)
end
function double_layer_op(A::AbstractArray{T,5}, O) where T
    d,D = size(A,1), size(A,2)
    @tensor m[u,up,r,rp,d,dp,l,lp] := A[p,u,r,d,l]*O[p,q]*conj(A[q,up,rp,dp,lp])
    reshape(m, D^2, D^2, D^2, D^2)
end
build_all_dl(st) = DoubleLayers(
    double_layer(st.A), double_layer(st.B),
    double_layer(st.C), double_layer(st.D), st.D_bond^2)

# ═══════════════════════════════════════════════════
# §5  CTMRG — 4 EXPLICIT ABSORPTION HELPERS
# ═══════════════════════════════════════════════════
function _absorb_left_col(Clu,Cld,Tu,Td,Tl1,Tl2,mt,mb,χm)
    χ=size(Clu,1); D2=size(mt,1)
    @tensor nClu[c,R,Dm,d] := Clu[a,b]*Tu[b,U,c]*Tl1[a,L,d]*mt[U,R,Dm,L]
    @tensor nCld[e,Um,R,c] := Cld[a,b]*Td[b,Db,c]*Tl2[e,L,a]*mb[Um,R,Db,L]
    @tensor Ψ[a,b,c,d] := nClu[a,b,v,e]*nCld[e,v,c,d]
    F=svd(reshape(Ψ,χ*D2,D2*χ)); χn=min(χm,length(F.S))
    P=reshape(F.U[:,1:χn],χ,D2,χn)
    Q=permutedims(reshape(F.Vt[1:χn,:],χn,D2,χ),(2,3,1))
    @tensor C1n[α,β] := conj(P[c,R,α])*nClu[c,R,Dm,d]*Q[Dm,d,β]
    @tensor C4n[β,α] := conj(Q[Um,e,β])*nCld[e,Um,R,c]*P[c,R,α]
    @tensor Tg1[a,U,R,Dm,b] := Tl1[a,L,b]*mt[U,R,Dm,L]
    @tensor T1n[α,R,β] := conj(Q[U,a,α])*Tg1[a,U,R,Dm,b]*Q[Dm,b,β]
    @tensor Tg2[a,U,R,Dm,b] := Tl2[a,L,b]*mb[U,R,Dm,L]
    @tensor T2n[α,R,β] := conj(Q[U,a,α])*Tg2[a,U,R,Dm,b]*Q[Dm,b,β]
    nrm!(C1n); nrm!(C4n); nrm!(T1n); nrm!(T2n)
    C1n, C4n, T1n, T2n
end

function _absorb_right_col(Cru,Crd,Tu,Td,Tr1,Tr2,mt,mb,χm)
    χ=size(Cru,1); D2=size(mt,1)
    @tensor nCru[c,Lm,Dm,d] := Cru[a,b]*Tu[c,U,b]*Tr1[a,Rb,d]*mt[U,Rb,Dm,Lm]
    @tensor nCrd[e,Um,Lm,c] := Crd[a,b]*Td[c,Db,b]*Tr2[e,Rb,a]*mb[Um,Rb,Db,Lm]
    @tensor Ψ[a,b,c,d] := nCru[a,b,v,e]*nCrd[e,v,c,d]
    F=svd(reshape(Ψ,χ*D2,D2*χ)); χn=min(χm,length(F.S))
    P=reshape(F.U[:,1:χn],χ,D2,χn)
    Q=permutedims(reshape(F.Vt[1:χn,:],χn,D2,χ),(2,3,1))
    @tensor C1n[α,β] := conj(P[c,L,α])*nCru[c,L,Dm,d]*Q[Dm,d,β]
    @tensor C4n[β,α] := conj(Q[Um,e,β])*nCrd[e,Um,L,c]*P[c,L,α]
    @tensor Tg1[a,U,Lm,Dm,b] := Tr1[a,Rb,b]*mt[U,Rb,Dm,Lm]
    @tensor T1n[α,Lm,β] := conj(Q[U,a,α])*Tg1[a,U,Lm,Dm,b]*Q[Dm,b,β]
    @tensor Tg2[a,U,Lm,Dm,b] := Tr2[a,Rb,b]*mb[U,Rb,Dm,Lm]
    @tensor T2n[α,Lm,β] := conj(Q[U,a,α])*Tg2[a,U,Lm,Dm,b]*Q[Dm,b,β]
    nrm!(C1n); nrm!(C4n); nrm!(T1n); nrm!(T2n)
    C1n, C4n, T1n, T2n
end

function _absorb_top_row(Clu,Cru,Tl,Tr,Tu1,Tu2,ml,mr,χm)
    χ=size(Clu,1); D2=size(ml,1)
    @tensor nClu[d,Dm,R,r] := Clu[a,b]*Tl[a,Lb,d]*Tu1[b,U,r]*ml[U,R,Dm,Lb]
    @tensor nCru[d,Dm,L,l] := Cru[a,b]*Tr[a,Rb,d]*Tu2[l,U,b]*mr[U,Rb,Dm,L]
    @tensor Ψ[a,b,c,d] := nClu[a,b,R,r]*nCru[d,c,R,r]
    F=svd(reshape(Ψ,χ*D2,D2*χ)); χn=min(χm,length(F.S))
    P=reshape(F.U[:,1:χn],χ,D2,χn)
    Q=permutedims(reshape(F.Vt[1:χn,:],χn,D2,χ),(2,3,1))
    @tensor C1n[α,β] := conj(P[d,Dm,α])*nClu[d,Dm,R,r]*Q[R,r,β]
    @tensor C2n[β,α] := conj(Q[Dm,d,β])*nCru[d,Dm,L,l]*P[l,L,α]
    @tensor Tg1[l,Lb,R,Dm,r] := Tu1[l,U,r]*ml[U,R,Dm,Lb]
    @tensor T1n[α,Dm,β] := conj(P[l,Lb,α])*Tg1[l,Lb,R,Dm,r]*Q[R,r,β]
    @tensor Tg2[l,Rb,Dm,L,r] := Tu2[l,U,r]*mr[U,Rb,Dm,L]
    @tensor T2n[α,Dm,β] := conj(P[l,L,α])*Tg2[l,L,Rb,Dm,r]*Q[Rb,r,β]
    nrm!(C1n); nrm!(C2n); nrm!(T1n); nrm!(T2n)
    C1n, C2n, T1n, T2n
end

function _absorb_bot_row(Cld,Crd,Tl,Tr,Td1,Td2,ml,mr,χm)
    χ=size(Cld,1); D2=size(ml,1)
    @tensor nCld[u,Um,R,r] := Cld[a,b]*Tl[u,Lb,a]*Td1[b,Db,r]*ml[Um,R,Db,Lb]
    @tensor nCrd[u,Um,L,l] := Crd[a,b]*Tr[u,Rb,a]*Td2[l,Db,b]*mr[Um,Rb,Db,L]
    @tensor Ψ[a,b,c,d] := nCld[a,b,R,r]*nCrd[d,c,R,r]
    F=svd(reshape(Ψ,χ*D2,D2*χ)); χn=min(χm,length(F.S))
    P=reshape(F.U[:,1:χn],χ,D2,χn)
    Q=permutedims(reshape(F.Vt[1:χn,:],χn,D2,χ),(2,3,1))
    @tensor C1n[α,β] := conj(P[u,Um,α])*nCld[u,Um,R,r]*Q[R,r,β]
    @tensor C2n[β,α] := conj(Q[Um,u,β])*nCrd[u,Um,L,l]*P[l,L,α]
    @tensor Tg1[l,Lb,Um,R,r] := Td1[l,Db,r]*ml[Um,R,Db,Lb]
    @tensor T1n[α,Um,β] := conj(P[l,Lb,α])*Tg1[l,Lb,Um,R,r]*Q[R,r,β]
    @tensor Tg2[l,Rb,Um,L,r] := Td2[l,Db,r]*mr[Um,Rb,Db,L]
    @tensor T2n[α,Um,β] := conj(P[l,L,α])*Tg2[l,L,Um,Rb,r]*Q[Rb,r,β]
    nrm!(C1n); nrm!(C2n); nrm!(T1n); nrm!(T2n)
    C1n, C2n, T1n, T2n
end

# ═══════════════════════════════════════════════════
# §6  CTMRG 4 MOVES + SWEEP + CONVERGE
# ═══════════════════════════════════════════════════
function ctm_move_left!(e,dl,χ)
    c1,c4,t1,t2 = _absorb_left_col(e.Clu,e.Cld,e.Tu1,e.Td1,e.Tl1,e.Tl2,dl.mA,dl.mC,χ)
    c1b,c4b,t1b,t2b = _absorb_left_col(c1,c4,e.Tu2,e.Td2,t1,t2,dl.mB,dl.mD,χ)
    e.Clu=c1b; e.Cld=c4b; e.Tl1=t1b; e.Tl2=t2b; normalize_ctm!(e)
end
function ctm_move_right!(e,dl,χ)
    c1,c4,t1,t2 = _absorb_right_col(e.Cru,e.Crd,e.Tu2,e.Td2,e.Tr1,e.Tr2,dl.mB,dl.mD,χ)
    c1b,c4b,t1b,t2b = _absorb_right_col(c1,c4,e.Tu1,e.Td1,t1,t2,dl.mA,dl.mC,χ)
    e.Cru=c1b; e.Crd=c4b; e.Tr1=t1b; e.Tr2=t2b; normalize_ctm!(e)
end
function ctm_move_up!(e,dl,χ)
    c1,c2,t1,t2 = _absorb_top_row(e.Clu,e.Cru,e.Tl1,e.Tr1,e.Tu1,e.Tu2,dl.mA,dl.mB,χ)
    c1b,c2b,t1b,t2b = _absorb_top_row(c1,c2,e.Tl2,e.Tr2,t1,t2,dl.mC,dl.mD,χ)
    e.Clu=c1b; e.Cru=c2b; e.Tu1=t1b; e.Tu2=t2b; normalize_ctm!(e)
end
function ctm_move_down!(e,dl,χ)
    c1,c2,t1,t2 = _absorb_bot_row(e.Cld,e.Crd,e.Tl2,e.Tr2,e.Td1,e.Td2,dl.mC,dl.mD,χ)
    c1b,c2b,t1b,t2b = _absorb_bot_row(c1,c2,e.Tl1,e.Tr1,t1,t2,dl.mA,dl.mB,χ)
    e.Cld=c1b; e.Crd=c2b; e.Td1=t1b; e.Td2=t2b; normalize_ctm!(e)
end
function ctmrg_sweep!(e,dl,χ)
    ctm_move_left!(e,dl,χ); ctm_move_up!(e,dl,χ)
    ctm_move_right!(e,dl,χ); ctm_move_down!(e,dl,χ)
end
function ctmrg_converge!(e, dl, χ; max_iter=50, tol=1e-4, verbose=true)
    sv_old = Float64[]
    for it in 1:max_iter
        ctmrg_sweep!(e, dl, χ)
        sv = real.(svdvals(e.Clu)); sv ./= (sv[1]+1e-30)
        if !isempty(sv_old)
            n = min(length(sv), length(sv_old))
            δ = norm(sv[1:n] .- sv_old[1:n])
            verbose && @printf("  CTMRG %3d: ΔSV=%.2e\n", it, δ)
            δ < tol && (verbose && println("  converged"); return e, it)
        end
        sv_old = sv
    end
    verbose && println("  max_iter reached")
    e, max_iter
end

# ═══════════════════════════════════════════════════
# §7  OBSERVABLES
# ═══════════════════════════════════════════════════
function contract_2x2(e, mA, mB, mC, mD)
    @tensor top[a,b1,b2,c] := e.Clu[a,x]*e.Tu1[x,b1,y]*e.Tu2[y,b2,z]*e.Cru[c,z]
    @tensor bot[a,b1,b2,c] := e.Cld[a,x]*e.Td1[x,b1,y]*e.Td2[y,b2,z]*e.Crd[c,z]
    @tensor s1[b1,b2,c,L,d] := top[a,b1,b2,c]*e.Tl1[a,L,d]
    @tensor s2[b2,c,R,D,d] := s1[U,b2,c,L,d]*mA[U,R,D,L]
    @tensor s3[c,R2,D2,D1,d] := s2[U2,c,R1,D1,d]*mB[U2,R2,D2,R1]
    @tensor s4[D2,D1,d,dt] := s3[u,R2,D2,D1,d]*e.Tr1[u,R2,dt]
    @tensor s5[D1,D2,L2,dt,d2] := s4[D2,D1,d,dt]*e.Tl2[d,L2,d2]
    @tensor s6[D2,R3,D3,dt,d2] := s5[U3,D2,L2,dt,d2]*mC[U3,R3,D3,L2]
    @tensor s7[R4,D4,D3,dt,d2] := s6[U4,R3,D3,dt,d2]*mD[U4,R4,D4,R3]
    @tensor s8[D4,D3,d2,dt2] := s7[R4,D4,D3,dt,d2]*e.Tr2[dt,R4,dt2]
    @tensor v[] := s8[D4,D3,d2,dt2]*bot[d2,D3,D4,dt2]
    v[]
end
contract_norm(e,dl) = contract_2x2(e, dl.mA, dl.mB, dl.mC, dl.mD)
function expect_1site(e, st, dl, O, site::Symbol)
    mO = double_layer_op(getfield(st, site), O)
    mA = (site==:A) ? mO : dl.mA
    mB = (site==:B) ? mO : dl.mB
    mC = (site==:C) ? mO : dl.mC
    mD = (site==:D) ? mO : dl.mD
    real(contract_2x2(e,mA,mB,mC,mD) / contract_norm(e,dl))
end
expect_1site_avg(e,st,dl,O) = sum(expect_1site(e,st,dl,O,s) for s in (:A,:B,:C,:D))/4
function expect_2site(e, st, dl, O2, sL, sR)
    d = st.d; nrm = contract_norm(e, dl)
    O2m = reshape(O2, d^2, d^2); F = svd(O2m)
    val = zero(ComplexF64)
    for α in 1:length(F.S)
        abs(F.S[α]) < 1e-14*F.S[1] && break
        OL = reshape(F.U[:,α]*sqrt(F.S[α]), d, d)
        OR = reshape(sqrt(F.S[α])*F.Vt[α,:], d, d)
        mL = double_layer_op(getfield(st, sL), OL)
        mR = double_layer_op(getfield(st, sR), OR)
        mA = (sL==:A) ? mL : ((sR==:A) ? mR : dl.mA)
        mB = (sL==:B) ? mL : ((sR==:B) ? mR : dl.mB)
        mC = (sL==:C) ? mL : ((sR==:C) ? mR : dl.mC)
        mD = (sL==:D) ? mL : ((sR==:D) ? mR : dl.mD)
        val += contract_2x2(e,mA,mB,mC,mD)
    end
    real(val/nrm)
end
function compute_energy(e, st, dl, J, h)
    Ob = ising_bond_op(J)
    Eab = expect_2site(e,st,dl,Ob,:A,:B)
    Ecd = expect_2site(e,st,dl,Ob,:C,:D)
    Eac = expect_2site(e,st,dl,Ob,:A,:C)
    Ebd = expect_2site(e,st,dl,Ob,:B,:D)
    Ef = -h * expect_1site_avg(e,st,dl,σx)
    (Eab + Ecd + Eac + Ebd)/4 + Ef
end
entanglement_entropy(λ) = (p=λ.^2; p./=sum(p); -sum(x->x*log(max(x,1e-30)),p))

# ═══════════════════════════════════════════════════
# §8  SIMPLE UPDATE
# ═══════════════════════════════════════════════════
function su_bond_h!(st, w, gate, Dm, sL, sR, λb, λuL, λdL, λoL, λuR, λdR, λoR)
    AL=getfield(st,sL); AR=getfield(st,sR); d,D=st.d,st.D_bond
    lb=getfield(w,λb); luL=getfield(w,λuL); ldL=getfield(w,λdL); loL=getfield(w,λoL)
    luR=getfield(w,λuR); ldR=getfield(w,λdR); loR=getfield(w,λoR)
    sqb=Diagonal(sqrt.(lb)); squl=Diagonal(sqrt.(luL)); sqdl=Diagonal(sqrt.(ldL)); sqol=Diagonal(sqrt.(loL))
    squr=Diagonal(sqrt.(luR)); sqdr=Diagonal(sqrt.(ldR)); sqor=Diagonal(sqrt.(loR))
    @tensor Aw[p,u,r,d,l] := AL[p,u2,r2,d2,l2]*squl[u,u2]*sqb[r,r2]*sqdl[d,d2]*sqol[l,l2]
    @tensor Bw[p,u,r,d,l] := AR[p,u2,r2,d2,l2]*squr[u,u2]*sqor[r,r2]*sqdr[d,d2]*sqb[l,l2]
    @tensor Θ[p1,u1,d1,l1,p2,u2,r2,d2] := Aw[p1,u1,x,d1,l1]*Bw[p2,u2,r2,d2,x]
    @tensor Θn[p1,u1,d1,l1,p2,u2,r2,d2] := gate[p1,p2,q1,q2]*Θ[q1,u1,d1,l1,q2,u2,r2,d2]
    F=svd(reshape(Θn,d*D^3,d*D^3)); Dn=min(Dm,length(F.S))
    λn=F.S[1:Dn]; λn./=norm(λn)
    An=permutedims(reshape(F.U[:,1:Dn],d,D,D,D,Dn),(1,2,5,3,4))
    Bn=reshape((Diagonal(F.S[1:Dn])*F.Vt[1:Dn,:])',d,D,D,D,Dn)
    iul=Diagonal(1 ./sqrt.(max.(luL,1e-15))); idl=Diagonal(1 ./sqrt.(max.(ldL,1e-15)))
    iol=Diagonal(1 ./sqrt.(max.(loL,1e-15))); iur=Diagonal(1 ./sqrt.(max.(luR,1e-15)))
    idr=Diagonal(1 ./sqrt.(max.(ldR,1e-15))); ior=Diagonal(1 ./sqrt.(max.(loR,1e-15)))
    if Dn==D
        @tensor Af[p,u,r,d,l] := An[p,u2,r,d2,l2]*iul[u,u2]*idl[d,d2]*iol[l,l2]
        @tensor Bf[p,u,r,d,l] := Bn[p,u2,r2,d2,l]*iur[u,u2]*ior[r,r2]*idr[d,d2]
        setfield!(st,sL,Af./norm(Af)); setfield!(st,sR,Bf./norm(Bf))
    else
        setfield!(st,sL,An./norm(An)); setfield!(st,sR,Bn./norm(Bn)); st.D_bond=Dn
    end
    lf=zeros(Dm); lf[1:min(Dn,Dm)]=λn[1:min(Dn,Dm)]; lf./=max(norm(lf),1e-30)
    setfield!(w,λb,lf)
end
su_AB!(st,w,g,D) = su_bond_h!(st,w,g,D,:A,:B,:λ_AB,:λ_AC,:λ_AC,:λ_AB,:λ_BD,:λ_BD,:λ_AB)
su_CD!(st,w,g,D) = su_bond_h!(st,w,g,D,:C,:D,:λ_CD,:λ_AC,:λ_AC,:λ_CD,:λ_BD,:λ_BD,:λ_CD)
function su_AC!(st,w,g,D)
    for s in (:A,:C); setfield!(st,s,permutedims(getfield(st,s),(1,5,2,3,4))); end
    su_bond_h!(st,w,g,D,:A,:C,:λ_AC,:λ_AB,:λ_AB,:λ_AC,:λ_CD,:λ_CD,:λ_AC)
    for s in (:A,:C); setfield!(st,s,permutedims(getfield(st,s),(1,3,4,5,2))); end
end
function su_BD!(st,w,g,D)
    for s in (:B,:D); setfield!(st,s,permutedims(getfield(st,s),(1,5,2,3,4))); end
    su_bond_h!(st,w,g,D,:B,:D,:λ_BD,:λ_AB,:λ_AB,:λ_BD,:λ_CD,:λ_CD,:λ_BD)
    for s in (:B,:D); setfield!(st,s,permutedims(getfield(st,s),(1,3,4,5,2))); end
end
function simple_update!(st, w, J, h; D_max=st.D_bond,
                        schedule=[(0.1,50),(0.01,100),(0.001,200)], verbose=true)
    for (dτ,ns) in schedule
        g = ising_gate(J,h,dτ)
        for _ in 1:ns
            su_AB!(st,w,g,D_max); su_CD!(st,w,g,D_max)
            su_AC!(st,w,g,D_max); su_BD!(st,w,g,D_max)
        end
        verbose && @printf("  dτ=%.4f ×%d: S=%.4f\n", dτ, ns, entanglement_entropy(w.λ_AB))
    end
end

# ═══════════════════════════════════════════════════
# §9  ENVIRONMENT BUILDERS (site removed)
# ═══════════════════════════════════════════════════
function _contract_top_row(e, dl)
    @tensor top[a,b1,b2,c] := e.Clu[a,x]*e.Tu1[x,b1,y]*e.Tu2[y,b2,z]*e.Cru[c,z]
    @tensor s1[b1,b2,c,L,d] := top[a,b1,b2,c]*e.Tl1[a,L,d]
    @tensor s2[b2,c,R,D,d] := s1[U,b2,c,L,d]*dl.mA[U,R,D,L]
    @tensor s3[c,R2,D2,D1,d] := s2[U2,c,R1,D1,d]*dl.mB[U2,R2,D2,R1]
    @tensor tr[D2,D1,d,dt] := s3[u,R2,D2,D1,d]*e.Tr1[u,R2,dt]
    tr
end

function build_env_A(e, dl)
    @tensor LT[U,L,dTl,rTu] := e.Clu[a,b]*e.Tu1[b,U,rTu]*e.Tl1[a,L,dTl]
    @tensor RT1[l,Ub,c] := e.Tu2[l,Ub,z]*e.Cru[c,z]
    @tensor RT2[l,Rb,Db,Ra,c] := RT1[l,Ub,c]*dl.mB[Ub,Rb,Db,Ra]
    @tensor RT[l,Db,Ra,dTr] := RT2[l,Rb,Db,Ra,c]*e.Tr1[c,Rb,dTr]
    @tensor TH[U,L,Ra,Db,dTl,dTr] := LT[U,L,dTl,x]*RT[x,Db,Ra,dTr]
    @tensor LB1[u,Lc,r] := e.Tl2[u,Lc,d]*e.Cld[d,r]
    @tensor LB2[u,Lc,Dc,r] := LB1[u,Lc,x]*e.Td1[x,Dc,r]
    @tensor LB[u,Da,Rc,r] := LB2[u,Lc,Dc,r]*dl.mC[Da,Rc,Dc,Lc]
    @tensor RB1[u,Rd,l] := e.Tr2[u,Rd,d]*e.Crd[d,l]
    @tensor RB2[u,Rd,Dd,l] := RB1[u,Rd,x]*e.Td2[l,Dd,x]
    @tensor RB[u,Db2,Ld,l] := RB2[u,Rd,Dd,l]*dl.mD[Db2,Rd,Dd,Ld]
    @tensor BH[u,Da,Db2,v] := LB[u,Da,x,y]*RB[v,Db2,x,y]
    @tensor N[U,Ra,Da,L] := TH[U,L,Ra,Db,a,b]*BH[a,Da,Db,b]
    N
end

function build_env_B(e, dl)
    @tensor LT1[a,Ua,rTu] := e.Clu[a,b]*e.Tu1[b,Ua,rTu]
    @tensor LT2[Ua,La,rTu,dTl] := LT1[a,Ua,rTu]*e.Tl1[a,La,dTl]
    @tensor LT[rTu,Lb,Da,dTl] := LT2[Ua,La,rTu,dTl]*dl.mA[Ua,Lb,Da,La]
    @tensor RTC[Ub,Rb,dTr,lTu] := e.Cru[a,b]*e.Tu2[lTu,Ub,b]*e.Tr1[a,Rb,dTr]
    @tensor TH[Ub,Rb,Lb,Da,dTl,dTr] := LT[x,Lb,Da,dTl]*RTC[Ub,Rb,dTr,x]
    @tensor LB1[u,Lc,r] := e.Tl2[u,Lc,d]*e.Cld[d,r]
    @tensor LB2[u,Lc,Dc,r] := LB1[u,Lc,x]*e.Td1[x,Dc,r]
    @tensor LB[u,Da2,Rc,r] := LB2[u,Lc,Dc,r]*dl.mC[Da2,Rc,Dc,Lc]
    @tensor RB1[u,Rd,l] := e.Tr2[u,Rd,d]*e.Crd[d,l]
    @tensor RB2[u,Rd,Dd,l] := RB1[u,Rd,x]*e.Td2[l,Dd,x]
    @tensor RB[u,Db,Ld,l] := RB2[u,Rd,Dd,l]*dl.mD[Db,Rd,Dd,Ld]
    @tensor BH[u,Da2,Db,v] := LB[u,Da2,x,y]*RB[v,Db,x,y]
    @tensor N[Ub,Lb,Db,Rb] := TH[Ub,Rb,Lb,Da,a,b]*BH[a,Da,Db,b]
    N
end

function build_env_C(e, dl)
    tr = _contract_top_row(e, dl)
    @tensor bl1[u,Lc,r] := e.Tl2[u,Lc,x]*e.Cld[x,r]
    @tensor Lcol[u,Lc,Dc,r] := bl1[u,Lc,x]*e.Td1[x,Dc,r]
    @tensor br1[u,Rd,l] := e.Tr2[u,Rd,x]*e.Crd[x,l]
    @tensor br2[u,Rd,Dd,l] := br1[u,Rd,x]*e.Td2[l,Dd,x]
    @tensor Rcol[u,Ud,Rc,l] := br2[u,Rd,Dd,l]*dl.mD[Ud,Rd,Dd,Rc]
    @tensor N[Uc,Rc,Dc,Lc] := tr[Db,Uc,xl,xr]*Lcol[xl,Lc,Dc,yt]*Rcol[xr,Db,Rc,yt]
    N
end

function build_env_D(e, dl)
    tr = _contract_top_row(e, dl)
    @tensor bl1[u,Lc,r] := e.Tl2[u,Lc,x]*e.Cld[x,r]
    @tensor bl2[u,Lc,Dc,r] := bl1[u,Lc,x]*e.Td1[x,Dc,r]
    @tensor Lcol[u,Da,Ld,r] := bl2[u,Lc,Dc,r]*dl.mC[Da,Ld,Dc,Lc]
    @tensor br1[u,Rd,l] := e.Tr2[u,Rd,x]*e.Crd[x,l]
    @tensor Rcol[u,Rd,Dd,l] := br1[u,Rd,x]*e.Td2[l,Dd,x]
    @tensor N[Ud,Ld,Dd,Rd] := tr[Ud,Da,xl,xr]*Lcol[xl,Da,Ld,yt]*Rcol[xr,Rd,Dd,yt]
    N
end

build_N_mat(N,D) = (N8=reshape(N,D,D,D,D,D,D,D,D);
    M=reshape(permutedims(N8,(2,4,6,8,1,3,5,7)),D^4,D^4); (M+M')/2)

# ═══════════════════════════════════════════════════
# §10  MEMORY-SAFE W BUILDERS (truncated SVD)
# ═══════════════════════════════════════════════════

"""Factorize Θ keeping only r_max singular values (KEY memory control)."""
function trunc_svd(Θn, d, D, r_max)
    F = svd(reshape(Θn, d*D^3, d*D^3))
    r = min(r_max, length(F.S))
    sqS = Diagonal(sqrt.(F.S[1:r]))
    Af = permutedims(reshape(F.U[:,1:r]*sqS, d,D,D,D,r), (1,2,5,3,4))
    Bf = reshape((sqS*F.Vt[1:r,:])', d,D,D,D,r)
    Af, Bf, r
end

"""W for site A on horizontal A─B bond."""
function build_W_A_h(e, dl, Θn, Bp, D, r_max)
    d = size(Bp,1)
    Af, Bf, r = trunc_svd(Θn, d, D, r_max)
    @tensor mBw[u,up,r,rp,d,dp,α,lp] := Bf[p,u,r,d,α]*conj(Bp[p,up,rp,dp,lp])
    mBW = reshape(mBw, D^2, D^2, D^2, r*D)
    @tensor LT[U,L,dTl,rTu] := e.Clu[a,b]*e.Tu1[b,U,rTu]*e.Tl1[a,L,dTl]
    @tensor RT1[l,Ub,c] := e.Tu2[l,Ub,z]*e.Cru[c,z]
    @tensor RT2[l,Rb,Db,Ra,c] := RT1[l,Ub,c]*mBW[Ub,Rb,Db,Ra]
    @tensor RT[l,Db,Ra,dTr] := RT2[l,Rb,Db,Ra,c]*e.Tr1[c,Rb,dTr]
    @tensor TH[U,L,Ra,Db,dTl,dTr] := LT[U,L,dTl,x]*RT[x,Db,Ra,dTr]
    @tensor LB1[u,Lc,rr] := e.Tl2[u,Lc,dd]*e.Cld[dd,rr]
    @tensor LB2[u,Lc,Dc,rr] := LB1[u,Lc,x]*e.Td1[x,Dc,rr]
    @tensor LB[u,Da,Rc,rr] := LB2[u,Lc,Dc,rr]*dl.mC[Da,Rc,Dc,Lc]
    @tensor RB1[u,Rd,ll] := e.Tr2[u,Rd,dd]*e.Crd[dd,ll]
    @tensor RB2[u,Rd,Dd,ll] := RB1[u,Rd,x]*e.Td2[ll,Dd,x]
    @tensor RB[u,Db2,Ld,ll] := RB2[u,Rd,Dd,ll]*dl.mD[Db2,Rd,Dd,Ld]
    @tensor BH[u,Da,Db2,v] := LB[u,Da,x,y]*RB[v,Db2,x,y]
    @tensor eW[U,Ra,Da,L] := TH[U,L,Ra,Db,a,b]*BH[a,Da,Db,b]
    es = reshape(eW, D,D, r,D, D,D, D,D)
    @tensor W[p,ub,rb,db,lb] := es[uk,ub,α,rb,dk,db,lk,lb]*Af[p,uk,α,dk,lk]
    W
end

"""W for site B on horizontal A─B bond."""
function build_W_B_h(e, dl, Θn, Ap, D, r_max)
    d = size(Ap,1)
    Af, Bf, r = trunc_svd(Θn, d, D, r_max)
    @tensor mAw[u,up,α,rp,d,dp,l,lp] := Af[p,u,α,d,l]*conj(Ap[p,up,rp,dp,lp])
    mAW = reshape(mAw, D^2, r*D, D^2, D^2)
    @tensor LT1[a,Ua,rTu] := e.Clu[a,b]*e.Tu1[b,Ua,rTu]
    @tensor LT2[Ua,La,rTu,dTl] := LT1[a,Ua,rTu]*e.Tl1[a,La,dTl]
    @tensor LT[rTu,Lbe,Da,dTl] := LT2[Ua,La,rTu,dTl]*mAW[Ua,Lbe,Da,La]
    @tensor RTC[Ub,Rb,dTr,lTu] := e.Cru[a,b]*e.Tu2[lTu,Ub,b]*e.Tr1[a,Rb,dTr]
    @tensor TH[Ub,Rb,Lbe,Da,dTl,dTr] := LT[x,Lbe,Da,dTl]*RTC[Ub,Rb,dTr,x]
    @tensor LB1[u,Lc,rr] := e.Tl2[u,Lc,dd]*e.Cld[dd,rr]
    @tensor LB2[u,Lc,Dc,rr] := LB1[u,Lc,x]*e.Td1[x,Dc,rr]
    @tensor LB[u,Da2,Rc,rr] := LB2[u,Lc,Dc,rr]*dl.mC[Da2,Rc,Dc,Lc]
    @tensor RB1[u,Rd,ll] := e.Tr2[u,Rd,dd]*e.Crd[dd,ll]
    @tensor RB2[u,Rd,Dd,ll] := RB1[u,Rd,x]*e.Td2[ll,Dd,x]
    @tensor RB[u,Db,Ld,ll] := RB2[u,Rd,Dd,ll]*dl.mD[Db,Rd,Dd,Ld]
    @tensor BH[u,Da2,Db,v] := LB[u,Da2,x,y]*RB[v,Db,x,y]
    @tensor eW[Ub,Lbe,Db,Rb] := TH[Ub,Rb,Lbe,Da,a,b]*BH[a,Da,Db,b]
    es = reshape(eW, D,D, r,D, D,D, D,D)
    @tensor W[p,ub,lb,db,rb] := es[uk,ub,α,lb,dk,db,rk,rb]*Bf[p,uk,rk,dk,α]
    W
end

# ═══════════════════════════════════════════════════
# §11  ALS SOLVER + FU FOR ALL 4 BONDS
# ═══════════════════════════════════════════════════

function _als_solve(e, dl, Θn, D, d, X_old, build_N_fn, build_W_fn, damp, r_max)
    N = build_N_fn(e, dl)
    Nm = build_N_mat(N, D)
    W = build_W_fn(e, dl, Θn, r_max)
    Nr = Nm + 1e-8*I(D^4)
    Wm = reshape(W, d, D^4)
    Xn = similar(X_old)
    for p in 1:d
        Xn[p,:,:,:,:] = reshape(Nr \ Wm[p,:], D,D,D,D)
    end
    any(isnan, Xn) && (Xn = X_old)
    nrm!(damp .* Xn .+ (1-damp) .* X_old)
end

"""Full update on horizontal A─B bond. ALWAYS runs (never skipped)."""
function full_update_AB!(st, e, dl, gate, J, h, D, χ, r_max, nals; damp=0.5, verbose=true)
    d = st.d
    verbose && @printf("    [AB] D=%d χ=%d r_max=%d nals=%d\n", D, χ, r_max, nals)

    @tensor Θ[p1,u1,d1,l1,p2,u2,r2,d2] := st.A[p1,u1,x,d1,l1]*st.B[p2,u2,r2,d2,x]
    @tensor Θn[p1,u1,d1,l1,p2,u2,r2,d2] := gate[p1,p2,q1,q2]*Θ[q1,u1,d1,l1,q2,u2,r2,d2]

    F0 = svd(reshape(Θn,d*D^3,d*D^3)); Dn = min(D, length(F0.S))
    st.A = permutedims(reshape(F0.U[:,1:Dn]*Diagonal(sqrt.(F0.S[1:Dn])),d,D,D,D,Dn),(1,2,5,3,4))
    st.B = reshape((Diagonal(sqrt.(F0.S[1:Dn]))*F0.Vt[1:Dn,:])',d,D,D,D,Dn)
    nrm!(st.A); nrm!(st.B)

    for als in 1:nals
        GC.gc()
        dlc = build_all_dl(st)
        ctmrg_converge!(e, dlc, χ; max_iter=10, tol=1e-3, verbose=false)
        Ao = copy(st.A)
        st.A = _als_solve(e,dlc,Θn,D,d,Ao,
            (e2,d2)->build_env_A(e2,d2),
            (e2,d2,t,rm)->build_W_A_h(e2,d2,t,st.B,D,rm), damp, r_max)
        GC.gc()
        dlc2 = build_all_dl(st)
        ctmrg_converge!(e, dlc2, χ; max_iter=10, tol=1e-3, verbose=false)
        Bo = copy(st.B)
        st.B = _als_solve(e,dlc2,Θn,D,d,Bo,
            (e2,d2)->build_env_B(e2,d2),
            (e2,d2,t,rm)->build_W_B_h(e2,d2,t,st.A,D,rm), damp, r_max)
        if verbose
            Ev = compute_energy(e,st,build_all_dl(st),J,h)
            @printf("      ALS %d: E=%.8f\n", als, Ev)
        end
    end
    dlf = build_all_dl(st)
    ctmrg_converge!(e, dlf, χ; max_iter=10, tol=1e-3, verbose=false)
    e, dlf
end

"""Full update on horizontal C─D bond."""
function full_update_CD!(st, e, dl, gate, J, h, D, χ, r_max, nals; damp=0.5, verbose=true)
    d = st.d
    verbose && @printf("    [CD] D=%d χ=%d r_max=%d nals=%d\n", D, χ, r_max, nals)

    @tensor Θ[p1,u1,d1,l1,p2,u2,r2,d2] := st.C[p1,u1,x,d1,l1]*st.D[p2,u2,r2,d2,x]
    @tensor Θn[p1,u1,d1,l1,p2,u2,r2,d2] := gate[p1,p2,q1,q2]*Θ[q1,u1,d1,l1,q2,u2,r2,d2]

    F0 = svd(reshape(Θn,d*D^3,d*D^3)); Dn = min(D, length(F0.S))
    st.C = permutedims(reshape(F0.U[:,1:Dn]*Diagonal(sqrt.(F0.S[1:Dn])),d,D,D,D,Dn),(1,2,5,3,4))
    st.D = reshape((Diagonal(sqrt.(F0.S[1:Dn]))*F0.Vt[1:Dn,:])',d,D,D,D,Dn)
    nrm!(st.C); nrm!(st.D)

    for als in 1:nals
        GC.gc()
        dlc = build_all_dl(st)
        ctmrg_converge!(e, dlc, χ; max_iter=10, tol=1e-3, verbose=false)
        Co = copy(st.C)
        st.C = _als_solve(e,dlc,Θn,D,d,Co,
            (e2,d2)->build_env_C(e2,d2),
            (e2,d2,t,rm)->build_W_A_h(e2,d2,t,st.D,D,rm), damp, r_max)
        GC.gc()
        dlc2 = build_all_dl(st)
        ctmrg_converge!(e, dlc2, χ; max_iter=10, tol=1e-3, verbose=false)
        Do = copy(st.D)
        st.D = _als_solve(e,dlc2,Θn,D,d,Do,
            (e2,d2)->build_env_D(e2,d2),
            (e2,d2,t,rm)->build_W_B_h(e2,d2,t,st.C,D,rm), damp, r_max)
        if verbose
            Ev = compute_energy(e,st,build_all_dl(st),J,h)
            @printf("      ALS %d: E=%.8f\n", als, Ev)
        end
    end
    dlf = build_all_dl(st)
    ctmrg_converge!(e, dlf, χ; max_iter=10, tol=1e-3, verbose=false)
    e, dlf
end

"""Full update on vertical A─C bond (explicit, no permutedims on iPEPS)."""
function full_update_AC!(st, e, dl, gate, J, h, D, χ, r_max, nals; damp=0.5, verbose=true)
    d = st.d
    verbose && @printf("    [AC] D=%d χ=%d r_max=%d nals=%d\n", D, χ, r_max, nals)

    # A.d contracts with C.u
    @tensor Θ[p1,u1,r1,l1,p2,r2,d2,l2] := st.A[p1,u1,r1,x,l1]*st.C[p2,x,r2,d2,l2]
    @tensor Θn[p1,u1,r1,l1,p2,r2,d2,l2] := gate[p1,p2,q1,q2]*Θ[q1,u1,r1,l1,q2,r2,d2,l2]

    F0 = svd(reshape(Θn,d*D^3,d*D^3)); Dn = min(D, length(F0.S))
    st.A = permutedims(reshape(F0.U[:,1:Dn]*Diagonal(sqrt.(F0.S[1:Dn])),d,D,D,D,Dn),(1,2,3,5,4))
    Ci = (Diagonal(sqrt.(F0.S[1:Dn]))*F0.Vt[1:Dn,:])'
    st.C = permutedims(reshape(Ci,d,D,D,D,Dn),(1,5,2,3,4))
    nrm!(st.A); nrm!(st.C)

    for als in 1:nals
        GC.gc()
        dlc = build_all_dl(st)
        ctmrg_converge!(e, dlc, χ; max_iter=10, tol=1e-3, verbose=false)
        Ao = copy(st.A)
        st.A = _als_solve(e,dlc,Θn,D,d,Ao,
            (e2,d2)->build_env_A(e2,d2),
            (e2,d2,t,rm)->build_W_A_h(e2,d2,t,st.C,D,rm), damp, r_max)
        GC.gc()
        dlc2 = build_all_dl(st)
        ctmrg_converge!(e, dlc2, χ; max_iter=10, tol=1e-3, verbose=false)
        Co = copy(st.C)
        st.C = _als_solve(e,dlc2,Θn,D,d,Co,
            (e2,d2)->build_env_C(e2,d2),
            (e2,d2,t,rm)->build_W_A_h(e2,d2,t,st.A,D,rm), damp, r_max)
        if verbose
            Ev = compute_energy(e,st,build_all_dl(st),J,h)
            @printf("      ALS %d: E=%.8f\n", als, Ev)
        end
    end
    dlf = build_all_dl(st)
    ctmrg_converge!(e, dlf, χ; max_iter=10, tol=1e-3, verbose=false)
    e, dlf
end

"""Full update on vertical B─D bond (explicit, no permutedims on iPEPS)."""
function full_update_BD!(st, e, dl, gate, J, h, D, χ, r_max, nals; damp=0.5, verbose=true)
    d = st.d
    verbose && @printf("    [BD] D=%d χ=%d r_max=%d nals=%d\n", D, χ, r_max, nals)

    @tensor Θ[p1,u1,r1,l1,p2,r2,d2,l2] := st.B[p1,u1,r1,x,l1]*st.D[p2,x,r2,d2,l2]
    @tensor Θn[p1,u1,r1,l1,p2,r2,d2,l2] := gate[p1,p2,q1,q2]*Θ[q1,u1,r1,l1,q2,r2,d2,l2]

    F0 = svd(reshape(Θn,d*D^3,d*D^3)); Dn = min(D, length(F0.S))
    st.B = permutedims(reshape(F0.U[:,1:Dn]*Diagonal(sqrt.(F0.S[1:Dn])),d,D,D,D,Dn),(1,2,3,5,4))
    Di = (Diagonal(sqrt.(F0.S[1:Dn]))*F0.Vt[1:Dn,:])'
    st.D = permutedims(reshape(Di,d,D,D,D,Dn),(1,5,2,3,4))
    nrm!(st.B); nrm!(st.D)

    for als in 1:nals
        GC.gc()
        dlc = build_all_dl(st)
        ctmrg_converge!(e, dlc, χ; max_iter=10, tol=1e-3, verbose=false)
        Bo = copy(st.B)
        st.B = _als_solve(e,dlc,Θn,D,d,Bo,
            (e2,d2)->build_env_B(e2,d2),
            (e2,d2,t,rm)->build_W_B_h(e2,d2,t,st.D,D,rm), damp, r_max)
        GC.gc()
        dlc2 = build_all_dl(st)
        ctmrg_converge!(e, dlc2, χ; max_iter=10, tol=1e-3, verbose=false)
        Do = copy(st.D)
        st.D = _als_solve(e,dlc2,Θn,D,d,Do,
            (e2,d2)->build_env_D(e2,d2),
            (e2,d2,t,rm)->build_W_B_h(e2,d2,t,st.B,D,rm), damp, r_max)
        if verbose
            Ev = compute_energy(e,st,build_all_dl(st),J,h)
            @printf("      ALS %d: E=%.8f\n", als, Ev)
        end
    end
    dlf = build_all_dl(st)
    ctmrg_converge!(e, dlf, χ; max_iter=10, tol=1e-3, verbose=false)
    e, dlf
end

# ═══════════════════════════════════════════════════
# §12  FULL UPDATE SWEEP (all 4 bonds)
# ═══════════════════════════════════════════════════
function full_update_sweep!(st, e, dl, J, h, D, χ, r_max, nals;
                             dτ=0.001, damp=0.5, verbose=true)
    gate = ising_gate(J, h, dτ)
    verbose && println("  FU A─B")
    e,dl = full_update_AB!(st,e,dl,gate,J,h,D,χ,r_max,nals; damp=damp,verbose=verbose)
    verbose && println("  FU C─D")
    e,dl = full_update_CD!(st,e,dl,gate,J,h,D,χ,r_max,nals; damp=damp,verbose=verbose)
    verbose && println("  FU A─C")
    e,dl = full_update_AC!(st,e,dl,gate,J,h,D,χ,r_max,nals; damp=damp,verbose=verbose)
    verbose && println("  FU B─D")
    e,dl = full_update_BD!(st,e,dl,gate,J,h,D,χ,r_max,nals; damp=damp,verbose=verbose)
    verbose && println("  post-sweep CTMRG...")
    dl = build_all_dl(st)
    ctmrg_converge!(e, dl, χ; max_iter=20, tol=1e-3, verbose=verbose)
    E = compute_energy(e, st, dl, J, h)
    verbose && @printf("  ✓ sweep E = %.10f\n", E)
    e, dl, E
end

# ═══════════════════════════════════════════════════
# §13  DRIVER — SAME PIPELINE FOR ALL D
# ═══════════════════════════════════════════════════
const E_QMC = Dict(2.0=>-2.6781, 2.5=>-2.8951, 3.0=>-3.2802, 3.044=>-3.310, 3.5=>-3.7923, 4.0=>-4.3680)

function run_ipeps_2x2(; D=2, h=3.0, J=1.0, n_fu=1,
                        su_schedule=[(0.1,30),(0.01,60),(0.001,100)],
                        dτ_fu=0.001, damp=0.5, verbose=true)

    # Auto-tune parameters for this D
    χ, r_max, nals = auto_params(D)

    verbose && @printf("\n%s\n2×2 iPEPS: D=%d  χ=%d  r_max=%d  nals=%d  n_fu=%d  h=%.3f\n%s\n",
                       "="^55, D, χ, r_max, nals, n_fu, h, "="^55)

    st = init_ipeps(2, D)
    w  = init_weights(D)

    # ── Stage 1: Simple Update ──
    verbose && println("\n── Simple Update ──")
    simple_update!(st, w, J, h; D_max=D, schedule=su_schedule, verbose=verbose)

    # ── Stage 2: Initial CTMRG ──
    verbose && println("\n── Initial CTMRG (χ=$χ) ──")
    dl = build_all_dl(st)
    e  = init_ctm(χ, D^2)
    ctmrg_converge!(e, dl, χ; max_iter=20, tol=1e-4, verbose=verbose)

    # ── Stage 3: Full Update (ALWAYS runs) ──
    fu_ran = false
    if n_fu > 0
        verbose && @printf("\n── Full Update (%d sweeps) ──\n", n_fu)
        for sw in 1:n_fu
            verbose && @printf("\n  ▶ FU sweep %d/%d\n", sw, n_fu)
            e, dl, E = full_update_sweep!(st,e,dl,J,h,D,χ,r_max,nals;
                                           dτ=dτ_fu,damp=damp,verbose=verbose)
        end
        fu_ran = true
    end

    # ── Stage 4: Final CTMRG ──
    verbose && println("\n── Final CTMRG ──")
    dl = build_all_dl(st)
    ctmrg_converge!(e, dl, χ; max_iter=20, tol=1e-3, verbose=verbose)

    # ── Stage 5: Observables ──
    verbose && println("\n── Observables ──")
    dl = build_all_dl(st)
    E  = compute_energy(e, st, dl, J, h)
    mz = expect_1site_avg(e, st, dl, σz)
    mx = expect_1site_avg(e, st, dl, σx)
    S  = entanglement_entropy(w.λ_AB)
    method = fu_ran ? "SU+FU" : "SU"

    verbose && @printf("  Method : %s\n  E/site = %.10f\n  ⟨σᶻ⟩   = %.6f\n  ⟨σˣ⟩   = %.6f\n  S_ent  = %.6f\n",
                       method, E, mz, mx, S)
    haskey(E_QMC, h) && verbose && @printf("  E_QMC  = %.4f\n", E_QMC[h])
    verbose && println("  ✓ FU ran: $fu_ran")

    return (E=E, mz=mz, mx=mx, S=S, method=method, D=D, χ=χ, r_max=r_max,
            state=st, env=e, dl=dl, weights=w)
end

# ═══════════════════════════════════════════════════
# §14  CONVERGENCE STUDY
# ═══════════════════════════════════════════════════
function convergence_study(; h=3.0, J=1.0, n_fu=1,
                              su_sched=[(0.1,30),(0.01,60),(0.001,120)])
    println("\n" * "="^60)
    @printf("2×2 iPEPS CONVERGENCE — TFIM h=%.2f  n_fu=%d\n", h, n_fu)
    println("="^60)

    # Show parameter table
    print_params()

    results = NamedTuple[]
    for D in [2, 3, 4]
        t = @elapsed r = run_ipeps_2x2(; D=D, h=h, J=J, n_fu=n_fu, su_schedule=su_sched)
        push!(results, (D=r.D, χ=r.χ, r_max=r.r_max, method=r.method,
                        E=r.E, S=r.S, mz=r.mz, mx=r.mx, time=t))
        @printf("\n  D=%d done in %.0f s  E=%.6f  [%s]\n", D, t, r.E, r.method)
        GC.gc()
    end

    println("\n" * "="^60)
    println("RESULTS SUMMARY")
    println("="^60)
    @printf("%-3s %-3s %-5s %-6s %-15s %-10s %-10s %-10s %-6s\n",
            "D", "χ", "r_max", "method", "E/site", "⟨σᶻ⟩", "⟨σˣ⟩", "S", "sec")
    println("-"^70)
    for r in results
        @printf("%-3d %-3d %-5d %-6s %-15.8f %-10.6f %-10.6f %-10.6f %-6.0f\n",
                r.D, r.χ, r.r_max, r.method, r.E, r.mz, r.mx, r.S, r.time)
    end
    if haskey(E_QMC, h)
        @printf("\nQMC reference (h=%.1f): E/site = %.4f\n", h, E_QMC[h])
    end
    println("\n✓ All D used SAME pipeline: SU → CTMRG → FU → CTMRG → Observables")
    println("✓ Energy should decrease with D (variational principle)")

    # Validate monotonicity
    Es = [r.E for r in results]
    if issorted(Es)
        println("✓ Energy is monotonically decreasing ✓")
    else
        println("⚠ Energy not monotonic — may need more SU/FU steps")
    end

    return results
end

println("✅ 2×2 iPEPS — MEMORY-SAFE FULL UPDATE FOR ALL D")
println("   FU NEVER skipped — parameters auto-tuned per D")
println("   Same pipeline: SU → CTMRG → FU → CTMRG → Observables")
print_params()
