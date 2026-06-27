# EulerColors{NC} — compressible Euler + NC passive color/species scalars carried as EXTRA conserved
# variables ρXᵢ and advected by the SAME Godunov flux as the hydro (conserved slots 6..5+NC).
#
# This is the transpile/GPU realization of CMA passive advection: each color rides the identical mass
# flux the kernel already computes (color flux = ρu·Xᵢ), so ΣXᵢ=1 and uniform-X are preserved BY
# CONSTRUCTION — no separate color solve, no flux-matching subtlety (unlike porting colors.jl's explicit
# packed update onto the unsplit CTU). The packed UInt16 log2 sidecars (colors.jl) remain the STORAGE /
# ChemistryKernels boundary; inside a hydro step the colors are linear Float32 ρXᵢ.
#
# PRECISION: advect colors with the f32 kernels (`run_ctu!` / `run_rk2!`), NOT the f16-tiled
# `run_ctus!` / `run_ctumh!` — they store primitives in __half and a trace species X~1e-30 underflows.
#
# NV = 5+NC; momentum (2,3,4) rotates under dim-permutation, the colors are scalars (NOT in vidx).

struct EulerColors{NC} <: FVSystem
    γ::Float32
end
EulerColors{NC}(; γ = 5f0/3f0) where {NC} = EulerColors{NC}(Float32(γ))

"Number of passive color/species scalars carried by the system."
ncolors_sys(::EulerColors{NC}) where {NC} = NC

@inline nconserved(::EulerColors{NC}) where {NC} = 5 + NC
@inline vidx(::EulerColors{NC}) where {NC} = ((2, 3, 4),)

# Generic Julia contract methods (CPU backends + transpile_selfcheck + primitives); numerically
# identical to the unrolled Exprs handed to the transpiler in `_fvmeta` below.
@inline function cons2prim(s::EulerColors{NC}, U) where {NC}
    ρ = U[1]; iρ = inv(ρ)
    u = U[2]*iρ; v = U[3]*iρ; w = U[4]*iρ
    P = (s.γ - 1f0) * (U[5] - 0.5f0*ρ*(u*u + v*v + w*w))
    (ρ, u, v, w, P, ntuple(q -> U[5+q]*iρ, Val(NC))...)
end
@inline function prim2cons(s::EulerColors{NC}, W) where {NC}
    ρ = W[1]; u = W[2]; v = W[3]; w = W[4]; P = W[5]
    E = P/(s.γ - 1f0) + 0.5f0*ρ*(u*u + v*v + w*w)
    (ρ, ρ*u, ρ*v, ρ*w, E, ntuple(q -> ρ*W[5+q], Val(NC))...)
end
@inline function physflux_x(s::EulerColors{NC}, W) where {NC}
    ρ = W[1]; u = W[2]; v = W[3]; w = W[4]; P = W[5]
    E = P/(s.γ - 1f0) + 0.5f0*ρ*(u*u + v*v + w*w)
    (ρ*u, ρ*u*u + P, ρ*u*v, ρ*u*w, u*(E + P), ntuple(q -> ρ*u*W[5+q], Val(NC))...)
end
@inline function maxspeed_x(s::EulerColors{NC}, W) where {NC}
    ρ = W[1]; u = W[2]; P = W[5]
    abs(u) + sqrt(s.γ * P / ρ)
end

# The transpiler consumes `_fvmeta(sys).phys` = (argname, body_Expr) pairs and emits straight-line C
# (no loops), so the color components must be UNROLLED for each NC. Built here to mirror the Julia
# methods above exactly (`p.γ` → PRM[0]; the returned tuple's slots 6..5+NC are the colors).
function _euler_colors_meta(NC::Int)
    cs = [Symbol(:c, q) for q in 1:NC]   # conserved color slots ρXᵢ
    xs = [Symbol(:x, q) for q in 1:NC]   # primitive color slots Xᵢ
    consdestr = Expr(:tuple, :ρ, :mx, :my, :mz, :E, cs...)
    primdestr = Expr(:tuple, :ρ, :u, :v, :w, :P, xs...)

    c2p = Expr(:block,
        Expr(:(=), consdestr, :U),
        :(iρ = inv(ρ)),
        Expr(:(=), Expr(:tuple, :u, :v, :w), Expr(:tuple, :(mx*iρ), :(my*iρ), :(mz*iρ))),
        :(P = (p.γ - 1) * (E - 0.5f0*ρ*(u*u + v*v + w*w))),
        Expr(:tuple, :ρ, :u, :v, :w, :P, (:( $(cs[q]) * iρ ) for q in 1:NC)...))

    p2c = Expr(:block,
        Expr(:(=), primdestr, :W),
        :(E = P/(p.γ - 1) + 0.5f0*ρ*(u*u + v*v + w*w)),
        Expr(:tuple, :ρ, :(ρ*u), :(ρ*v), :(ρ*w), :E, (:( ρ * $(xs[q]) ) for q in 1:NC)...))

    pfx = Expr(:block,
        Expr(:(=), primdestr, :W),
        :(E = P/(p.γ - 1) + 0.5f0*ρ*(u*u + v*v + w*w)),
        Expr(:tuple, :(ρ*u), :(ρ*u*u + P), :(ρ*u*v), :(ρ*u*w), :(u*(E + P)),
             (:( (ρ*u) * $(xs[q]) ) for q in 1:NC)...))

    msx = Expr(:block,
        Expr(:(=), Expr(:tuple, :ρ, :u, :v, :w, :P), :W),
        :(abs(u) + sqrt(p.γ * P / ρ)))

    (nvars = 5 + NC, vidx = ((2, 3, 4),), params = (:γ,),
     phys = (cons2prim = (:U, c2p), prim2cons = (:W, p2c),
             physflux_x = (:W, pfx), maxspeed_x = (:W, msx)))
end
_fvmeta(::EulerColors{NC}) where {NC} = _euler_colors_meta(NC)

@doc """    EulerColors{NC}(; γ = 5/3)

Compressible Euler hydrodynamics with `NC` passive color/species scalars carried as extra conserved
variables `ρXᵢ` (slots 6..5+NC) and advected by the same Godunov flux as the fluid — so `ΣXᵢ=1` and
uniform colors are preserved exactly. Advect with the f32 kernels (`run_ctu!`/`run_rk2!`); the f16-tiled
path underflows trace species.""" EulerColors
