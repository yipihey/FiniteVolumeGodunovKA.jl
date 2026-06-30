# CPU-only physics validation for the Metal f16 dual-energy + GE_SCALE port.
# Validates (1) the Enzo dual-energy switch + PdV applied per-cell after a Strang step (the SPLIT-sweep
# analog of the CUDA CTU per-substep switch), and (2) the GE_SCALE f16 lift/unlift round-trip.
using FiniteVolumeGodunovKA
const FV = FiniteVolumeGodunovKA

# ---- mirror of the CUDA de16 per-cell switch + PdV, in host Julia, on the conserved tuple ----
# divv from central differences of the post-step velocity field (the split-sweep analog of the
# tile-local divv in k_ctus_de16). λ = dt/dx folds 1/dx; central diff uses 0.5*λ → dt*(div v).
@inline function de_pdv_switch(s, Un::NTuple{6,T}, divv_dx::T, λ::T) where {T}
    ρ, mx, my, mz, E, Ge = Un
    P  = (s.γ - one(T))*Ge
    Ge = Ge - λ*P*divv_dx                      # PdV: -dt*P*(div v)   (divv_dx = dx*divv)
    irho = inv(ρ); ke = T(0.5)*irho*(mx*mx+my*my+mz*mz)
    ratio = E > zero(T) ? (E-ke)/E : -one(T)
    if ratio > s.η
        Ged = E - ke; Ge = Ged > zero(T) ? Ged : Ge; E = ke + Ge   # warm/shocked: trust E
    else
        E = ke + Ge                                                 # cold: trust evolved Ge
    end
    (ρ, mx, my, mz, E, Ge)
end

# host post-step pass over a Grid3D{EulerDE}: central-diff divv + switch, periodic BC.
function de_postswitch!(g::FV.Grid3D, dt)
    s = g.sys; nx,ny,nz = g.nx,g.ny,g.nz; U = g.U
    λx = Float32(dt)/g.dx
    gi(i,n) = mod1(i,n)
    Unew = similar(U)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        uxp = U[gi(i+1,nx),j,k][2]/U[gi(i+1,nx),j,k][1]; uxm = U[gi(i-1,nx),j,k][2]/U[gi(i-1,nx),j,k][1]
        vyp = U[i,gi(j+1,ny),k][3]/U[i,gi(j+1,ny),k][1]; vym = U[i,gi(j-1,ny),k][3]/U[i,gi(j-1,ny),k][1]
        wzp = U[i,j,gi(k+1,nz)][4]/U[i,j,gi(k+1,nz)][1]; wzm = U[i,j,gi(k-1,nz)][4]/U[i,j,gi(k-1,nz)][1]
        divv_dx = 0.5f0*((uxp-uxm)+(vyp-vym)+(wzp-wzm))
        Unew[i,j,k] = de_pdv_switch(s, U[i,j,k], divv_dx, λx)
    end
    g.U = Unew
    return g
end

println("="^70)
println("TEST 1: GE_SCALE f16 lift/unlift round-trip on a cold cosmology state")
println("="^70)
ρ = 0.17f0; eint = 6.96f-8; Ge = ρ*eint   # ≈1.18e-8, FAR below the f16 subnormal floor ~6e-8
GS = 1f7
# with GE_SCALE: store Ge*GS as f16, read back /GS
Ge_lift = Float32(Float16(Ge*GS))/GS
# without GE_SCALE: raw f16
Ge_raw  = Float32(Float16(Ge))
println("  Ge (cold)            = ", Ge)
println("  Ge via f16 + GE_SCALE= ", Ge_lift, "   relerr=", abs(Ge_lift-Ge)/Ge)
println("  Ge via RAW f16       = ", Ge_raw,  "   (flushes to 0 → loses the cold internal energy)")
@assert abs(Ge_lift-Ge)/Ge < 1f-2  "GE_SCALE lift must preserve cold Ge to <1%"
@assert Ge_raw == 0f0               "raw f16 must flush the subnormal cold Ge to 0"
# E ≈ 2e-7 likewise
E = Ge + 0.5f0*ρ*(4f-4)^2
E_lift = Float32(Float16(E*GS))/GS; E_raw = Float32(Float16(E))
println("  E (cold)             = ", E, "   E via f16+GE_SCALE=", E_lift, "  raw f16=", E_raw)
@assert abs(E_lift-E)/E < 1f-2
println("  PASS: GE_SCALE=1e7 survives f16; raw f16 flushes cold Ge to 0.\n")

println("="^70)
println("TEST 2: Enzo dual-energy switch — COLD gas keeps evolved Ge (eint preserved)")
println("="^70)
γd = 5f0/3f0; nd=(32,16,16); dxd=Float32(2π/nd[1]); ρ0=0.17f0; e0=6.96f-8; v0=4f-4
mkd(i,j,k) = begin
    x=2f0π*(i-1)/nd[1]; y=2f0π*(j-1)/nd[2]; z=2f0π*(k-1)/nd[3]
    ρ=ρ0*(1f0+0.03f0*sin(x)); u=v0*(1f0+0.1f0*sin(y)); v=v0*0.1f0*cos(z); w=v0*0.1f0*sin(x+y)
    eint=e0*(1f0+0.02f0*cos(z)); Ge=ρ*eint; E=Ge+0.5f0*ρ*(u*u+v*v+w*w)
    (ρ,ρ*u,ρ*v,ρ*w,E,Ge)
end
Ud=[mkd(i,j,k) for i in 1:nd[1], j in 1:nd[2], k in 1:nd[3]]
sde = EulerDE(γ=γd, η=1f-3)
g = FV.Grid3D(sde, copy(Ud); dx=dxd, dy=dxd, dz=dxd, bc=:periodic, rsol=FV.LLF())
# CFL
function dtcfl(g; cfl=0.3f0)
    c = FV.max_wavespeed(g); cfl*min(g.dx,g.dy,g.dz)/c
end
dtd = dtcfl(g)
for nstep in 0:19
    FV.step!(g, dtd; rev=isodd(nstep))   # Strang sweeps advect Ge via physflux slot 6
    de_postswitch!(g, dtd)               # per-cell PdV + Enzo switch (the split analog of de16)
end
W = FV.primitives(g)
emin,emax = extrema(w[6] for w in W)
ρmin = minimum(w[1] for w in W)
allfin = all(w->all(isfinite,w), W)
println("  steps=20  finite=", allfin, "  ρmin=", ρmin)
println("  eint range = [", emin, ", ", emax, "]   target e0=", e0)
@assert allfin "no NaN/Inf"
@assert ρmin > 0 "rho>0"
@assert all(0.9f0*e0 < w[6] < 1.1f0*e0 for w in W) "cold eint preserved within 10%"
@assert emin > 1f-9 "eint NOT flushed to subnormal 0"
println("  PASS: cold gas keeps evolved Ge; eint preserved ≈", round(e0,sigdigits=3), "; no NaN.\n")

println("="^70)
println("TEST 3: Enzo switch — WARM/SHOCK (Sod) uses E−KE, NOT a stale Ge")
println("="^70)
# Uniform warm gas at rest: (E-KE)/E = 1 > η, so switch takes the E branch. Perturb Ge to a WRONG value
# and confirm the switch resyncs eint = (E-KE)/ρ from E, discarding the corrupted Ge.
ρw=1f0; Pw=1f0; Ge_true=Pw/(γd-1f0); E=Ge_true   # at rest E=Ge_true
Ge_corrupt = 0.5f0*Ge_true                        # pretend the evolved Ge drifted low
Un = (ρw, 0f0, 0f0, 0f0, E, Ge_corrupt)
out = de_pdv_switch(sde, Un, 0f0, 0.1f0)          # divv=0 (uniform), warm branch should fix Ge
eint_out = out[6]/out[1]
println("  corrupt Ge=", Ge_corrupt, " → switch Ge=", out[6], "  eint=", eint_out, " (true ", Ge_true/ρw, ")")
@assert isapprox(out[6], Ge_true; rtol=1f-5) "warm branch must resync Ge = E-KE"
println("  PASS: warm gas trusts E−KE and discards the corrupted Ge.\n")

println("="^70)
println("ALL CPU PHYSICS VALIDATIONS PASSED")
println("="^70)
