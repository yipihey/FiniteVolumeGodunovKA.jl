# EulerDE — compressible Euler with the DUAL-ENERGY formalism (Enzo/cosmology standard), so a COLD gas
# (internal energy ≪ kinetic energy) can be advanced with ρ and the momenta stored in f16 WITHOUT the
# pressure underflowing to NaN.
#
# THE PROBLEM. For cold gas p = (γ-1)(E − ½ρv²) is a tiny difference of two large numbers; in f16 it
# cancels/underflows to NaN (verified on the real cosmology state ρ≈0.17, E≈2e-7, eint≈7e-8, (E−KE)/E≈0.06).
#
# THE FIX. Carry a SECOND energy field, the internal-energy density Ge = ρ·eint, evolved by its own
# equation alongside the total energy E:
#       ∂t Ge + ∇·(Ge·v) = −p (∇·v)            (advection  +  PdV work)
# The advective part ∇·(Ge·v) is an ordinary conservative flux (slot 6, flux = Ge·u), handled by the
# SAME Godunov machinery as every other variable; the PdV source −p(∇·v) is non-conservative and is added
# by the kernel (see `k_ctus_de` in transpile_cuda.jl).  Pressure ALWAYS comes from the dual energy,
#       p = (γ-1)·Ge,
# so the Riemann flux never forms the cancelling E−½ρv².
#
# DUAL-ENERGY SWITCH (applied per cell AFTER the conservative update, Enzo/Bryan et al.): with KE=½ρv²,
#   if (E−KE)/E > η   (η≈1e-3; pressure-reliable / shocked gas):  eint = (E−KE)/ρ,  Ge ← ρ·eint   (sync)
#   else              (cold / kinetic-dominated):                 keep the EVOLVED Ge,  E ← KE + Ge  (sync)
# so the warm/shocked gas tracks the (accurate there) total energy while the cold gas trusts the
# separately-evolved internal energy — and the two stay mutually consistent.
#
# PRECISION (the win): in the tiled kernel slots 0–3 (ρ,ρu,ρv,ρw) live in an f16 tile; E and Ge live in
# an f32 tile.  This is the mixed-precision dual-energy path: f16 memory/bandwidth for the bulk fields,
# f32 only where the cold-gas pressure demands it.
#
# Primitive  W = (ρ, u, v, w, P, e)        P = (γ-1)·Ge,  e = Ge/ρ = eint
# Conserved  U = (ρ, ρu, ρv, ρw, E, Ge)    E = P/(γ-1) + ½ρ|v|²,  Ge = P/(γ-1) = ρ·eint
#
# NV = 6; momentum (2,3,4) rotates under dim-permutation; E (slot 5) and Ge (slot 6) are scalars.

struct EulerDE <: FVSystem
    γ::Float32
    η::Float32                                 # dual-energy switch threshold (Enzo eta1 ≈ 1e-3)
end
EulerDE(; γ = 5f0/3f0, η = 1f-3) = EulerDE(Float32(γ), Float32(η))

@inline nconserved(::EulerDE) = 6
@inline vidx(::EulerDE) = ((2, 3, 4),)

# Generic Julia contract methods — pressure from the DUAL energy Ge (slot 6), never E−½ρv².  These mirror
# the unrolled Exprs the transpiler consumes (`_euler_de_meta`) exactly and feed the host self-check.
@inline function cons2prim(s::EulerDE, U)
    ρ = U[1]; iρ = inv(ρ)
    u = U[2]*iρ; v = U[3]*iρ; w = U[4]*iρ
    Ge = U[6]
    P = (s.γ - 1f0) * Ge
    (ρ, u, v, w, P, Ge*iρ)
end
@inline function prim2cons(s::EulerDE, W)
    ρ = W[1]; u = W[2]; v = W[3]; w = W[4]; P = W[5]
    Ge = P/(s.γ - 1f0)
    E = Ge + 0.5f0*ρ*(u*u + v*v + w*w)
    (ρ, ρ*u, ρ*v, ρ*w, E, Ge)
end
@inline function physflux_x(s::EulerDE, W)
    ρ = W[1]; u = W[2]; v = W[3]; w = W[4]; P = W[5]
    Ge = P/(s.γ - 1f0)
    E = Ge + 0.5f0*ρ*(u*u + v*v + w*w)
    (ρ*u, ρ*u*u + P, ρ*u*v, ρ*u*w, u*(E + P), Ge*u)   # slot 6: advective internal-energy flux Ge·u
end
@inline function maxspeed_x(s::EulerDE, W)
    ρ = W[1]; u = W[2]; P = W[5]
    abs(u) + sqrt(s.γ * P / ρ)
end

# Transpiler metadata — unrolled Exprs matching the methods above (slot 6 = Ge, advected by Ge·u). The
# PdV source and the dual-energy switch are NOT here (they are non-conservative / per-cell): the kernel
# `k_ctus_de` adds them.
function _euler_de_meta()
    consdestr = Expr(:tuple, :ρ, :mx, :my, :mz, :E, :Ge)
    primdestr = Expr(:tuple, :ρ, :u, :v, :w, :P, :e)

    c2p = Expr(:block,
        Expr(:(=), consdestr, :U),
        :(iρ = inv(ρ)),
        Expr(:(=), Expr(:tuple, :u, :v, :w), Expr(:tuple, :(mx*iρ), :(my*iρ), :(mz*iρ))),
        :(P = (p.γ - 1) * Ge),
        Expr(:tuple, :ρ, :u, :v, :w, :P, :(Ge*iρ)))

    p2c = Expr(:block,
        Expr(:(=), primdestr, :W),
        :(Ge = P/(p.γ - 1)),
        :(E = Ge + 0.5f0*ρ*(u*u + v*v + w*w)),
        Expr(:tuple, :ρ, :(ρ*u), :(ρ*v), :(ρ*w), :E, :Ge))

    pfx = Expr(:block,
        Expr(:(=), primdestr, :W),
        :(Ge = P/(p.γ - 1)),
        :(E = Ge + 0.5f0*ρ*(u*u + v*v + w*w)),
        Expr(:tuple, :(ρ*u), :(ρ*u*u + P), :(ρ*u*v), :(ρ*u*w), :(u*(E + P)), :(Ge*u)))

    msx = Expr(:block,
        Expr(:(=), Expr(:tuple, :ρ, :u, :v, :w, :P, :e), :W),
        :(abs(u) + sqrt(p.γ * P / ρ)))

    (nvars = 6, vidx = ((2, 3, 4),), params = (:γ, :η),
     phys = (cons2prim = (:U, c2p), prim2cons = (:W, p2c),
             physflux_x = (:W, pfx), maxspeed_x = (:W, msx)))
end
_fvmeta(::EulerDE) = _euler_de_meta()

export EulerDE

@doc """    EulerDE(; γ = 5/3, η = 1e-3)

Compressible Euler with the **dual-energy formalism**: a second internal-energy-density field `Ge = ρ·eint`
(conserved slot 6) is evolved alongside the total energy `E` (slot 5), and the pressure is taken from the
dual energy `p = (γ-1)·Ge` — never the cancelling `E − ½ρv²`.  This lets a COLD gas (eint ≪ KE) run with
density and momenta stored in **f16** without the pressure underflowing to NaN.  Advance it with the
mixed-precision [`run_ctus_de!`](@ref) kernel (ρ,ρu,ρv,ρw in f16; E,Ge in f32).  `η` is the Enzo
dual-energy switch threshold (use the evolved `Ge` where `(E−KE)/E ≤ η`).""" EulerDE
