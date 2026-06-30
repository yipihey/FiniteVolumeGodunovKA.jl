# EulerDEColors{NC} — DUAL-ENERGY Euler (EulerDE) + NC PASSIVE COLOUR scalars (EulerColors), merged so a
# COLD gas can carry advected species in the ALL-f16 path WITHOUT the pressure underflowing to NaN.
#
# This is the union of the two existing extensions:
#   * dual energy   (src/euler_de.jl):    a second internal-energy-density field Ge = ρ·eint is evolved
#                                          alongside E; pressure p = (γ-1)·Ge — never the cancelling
#                                          E−½ρv² — so cold gas survives f16 (verified for EulerDE).
#   * passive colours (src/euler_colors.jl): NC extra conserved variables ρXᵢ advected by the SAME mass
#                                          flux as the hydro (colour flux = ρu·Xᵢ) — so ΣXᵢ=1 / uniform-X
#                                          are preserved BY CONSTRUCTION.
#
# Conserved layout  U = (ρ, ρu, ρv, ρw, E, Ge, ρX₁ … ρX_NC)      nconserved = 6 + NC
#                       slot:  0   1   2   3  4   5    6 … 5+NC   (0-based, matching the CUDA tile)
# Primitive layout  W = (ρ, u, v, w, P, e,  X₁ … X_NC)           P = (γ-1)·Ge,  e = Ge/ρ = eint,  Xᵢ = ρXᵢ/ρ
#
# The (ρ,ρv,E,Ge) part is NUMERICALLY IDENTICAL to EulerDE: pressure from the dual energy, the same
# physflux for E (= u(E+P)) and Ge (= Ge·u), the same PdV source −P(∇·v) and the Enzo dual-energy switch
# (both added per-cell by the kernel, NOT in the metadata below).  The colours ride on top exactly as in
# EulerColors: flux = (ρu)·Xᵢ.
#
# NV = 6+NC; momentum (2,3,4) rotates under dim-permutation; E (5), Ge (6) and the colours are scalars.
#
# ALL-f16 PATH (de_prec=:f16): the WHOLE primitive vector — including the colour slots — is stored in a
# single __half tile (`k_ctus_de16` extended with NCOL colour slots, see transpile_cuda.jl).  The energy
# slots P,e are GE_SCALE-lifted into f16's normal range; the colours are NOT scaled because the CICASS
# colour (HII, X≈0.05) is already normal-valued in f16.  A *trace* colour (X≲1e-4, near the f16 subnormal
# floor) would underflow here and would need the f32 colour side-channel (k_ctus_col, used by EulerColors'
# run_ctus!) — NOT implemented for this all-f16 path; document/assert normal-valued colours instead.

struct EulerDEColors{NC} <: FVSystem
    γ::Float32
    η::Float32                                 # dual-energy switch threshold (Enzo eta1 ≈ 1e-3)
end
EulerDEColors{NC}(; γ = 5f0/3f0, η = 1f-3) where {NC} = EulerDEColors{NC}(Float32(γ), Float32(η))

"Number of passive colour/species scalars carried by the system."
ncolors_sys(::EulerDEColors{NC}) where {NC} = NC

@inline nconserved(::EulerDEColors{NC}) where {NC} = 6 + NC
@inline vidx(::EulerDEColors{NC}) where {NC} = ((2, 3, 4),)

# Generic Julia contract methods — pressure from the DUAL energy Ge (slot 6), never E−½ρv² (identical to
# EulerDE for the hydro part); colours advected as Xᵢ = ρXᵢ/ρ (identical to EulerColors).  Mirror the
# unrolled Exprs in `_euler_de_colors_meta` exactly and feed the host self-check.
@inline function cons2prim(s::EulerDEColors{NC}, U) where {NC}
    ρ = U[1]; iρ = inv(ρ)
    u = U[2]*iρ; v = U[3]*iρ; w = U[4]*iρ
    Ge = U[6]
    P = (s.γ - 1f0) * Ge
    (ρ, u, v, w, P, Ge*iρ, ntuple(q -> U[6+q]*iρ, Val(NC))...)
end
@inline function prim2cons(s::EulerDEColors{NC}, W) where {NC}
    ρ = W[1]; u = W[2]; v = W[3]; w = W[4]; P = W[5]
    Ge = P/(s.γ - 1f0)
    E = Ge + 0.5f0*ρ*(u*u + v*v + w*w)
    (ρ, ρ*u, ρ*v, ρ*w, E, Ge, ntuple(q -> ρ*W[6+q], Val(NC))...)
end
@inline function physflux_x(s::EulerDEColors{NC}, W) where {NC}
    ρ = W[1]; u = W[2]; v = W[3]; w = W[4]; P = W[5]
    Ge = P/(s.γ - 1f0)
    E = Ge + 0.5f0*ρ*(u*u + v*v + w*w)
    (ρ*u, ρ*u*u + P, ρ*u*v, ρ*u*w, u*(E + P), Ge*u, ntuple(q -> ρ*u*W[6+q], Val(NC))...)
end
@inline function maxspeed_x(s::EulerDEColors{NC}, W) where {NC}
    ρ = W[1]; u = W[2]; P = W[5]
    abs(u) + sqrt(s.γ * P / ρ)
end

# Transpiler metadata — unrolled Exprs matching the methods above.  Slot 6 = Ge (advected by Ge·u); slots
# 7..6+NC (0-based 6..5+NC) = colours (advected by (ρu)·Xᵢ).  The PdV source and dual-energy switch are
# NOT here (non-conservative / per-cell): the kernels `k_ctus_de` / `k_ctus_de16` add them.
function _euler_de_colors_meta(NC::Int)
    cs = [Symbol(:c, q) for q in 1:NC]   # conserved colour slots ρXᵢ
    xs = [Symbol(:x, q) for q in 1:NC]   # primitive colour slots Xᵢ
    consdestr = Expr(:tuple, :ρ, :mx, :my, :mz, :E, :Ge, cs...)
    primdestr = Expr(:tuple, :ρ, :u, :v, :w, :P, :e, xs...)

    c2p = Expr(:block,
        Expr(:(=), consdestr, :U),
        :(iρ = inv(ρ)),
        Expr(:(=), Expr(:tuple, :u, :v, :w), Expr(:tuple, :(mx*iρ), :(my*iρ), :(mz*iρ))),
        :(P = (p.γ - 1) * Ge),
        Expr(:tuple, :ρ, :u, :v, :w, :P, :(Ge*iρ), (:( $(cs[q]) * iρ ) for q in 1:NC)...))

    p2c = Expr(:block,
        Expr(:(=), primdestr, :W),
        :(Ge = P/(p.γ - 1)),
        :(E = Ge + 0.5f0*ρ*(u*u + v*v + w*w)),
        Expr(:tuple, :ρ, :(ρ*u), :(ρ*v), :(ρ*w), :E, :Ge, (:( ρ * $(xs[q]) ) for q in 1:NC)...))

    pfx = Expr(:block,
        Expr(:(=), primdestr, :W),
        :(Ge = P/(p.γ - 1)),
        :(E = Ge + 0.5f0*ρ*(u*u + v*v + w*w)),
        Expr(:tuple, :(ρ*u), :(ρ*u*u + P), :(ρ*u*v), :(ρ*u*w), :(u*(E + P)), :(Ge*u),
             (:( (ρ*u) * $(xs[q]) ) for q in 1:NC)...))

    msx = Expr(:block,
        Expr(:(=), Expr(:tuple, :ρ, :u, :v, :w, :P, :e, (xs[q] for q in 1:NC)...), :W),
        :(abs(u) + sqrt(p.γ * P / ρ)))

    (nvars = 6 + NC, vidx = ((2, 3, 4),), params = (:γ, :η),
     phys = (cons2prim = (:U, c2p), prim2cons = (:W, p2c),
             physflux_x = (:W, pfx), maxspeed_x = (:W, msx)))
end
_fvmeta(::EulerDEColors{NC}) where {NC} = _euler_de_colors_meta(NC)

export EulerDEColors

@doc """    EulerDEColors{NC}(; γ = 5/3, η = 1e-3)

Compressible Euler with **both** the dual-energy formalism (a second internal-energy-density field
`Ge = ρ·eint`, conserved slot 6, pressure `p = (γ-1)·Ge` — never the cancelling `E − ½ρv²`) **and** `NC`
passive colour/species scalars `ρXᵢ` (slots 6..5+NC) advected by the same mass flux. This lets a COLD gas
carry advected species in the **all-f16** path (`de_prec=:f16`) without the pressure underflowing to NaN.
Advance with [`run_ctus_de16!`](@ref) (all-f16) or [`run_ctus_de!`](@ref) (mixed precision). Assumes the
colours are normal-valued in f16 (e.g. HII X≈0.05); a *trace* colour would need the f32 colour
side-channel, which the all-f16 path does not provide.""" EulerDEColors
