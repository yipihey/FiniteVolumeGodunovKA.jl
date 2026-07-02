# Smoke test for MOM_SCALE — in a bulk-subtracted (baryon-rest) frame the gas momentum S=ρ·δv is a small
# no-pedestal number; with ρ≈f_b~0.05 it lands in the f16 SUBNORMAL range and the velocity perturbation
# underflows.  MOM_SCALE lifts S into f16's normal range (like GE_SCALE for the cold energy).  Check: the
# velocity perturbation is floored at mom_scale=1 but preserved at mom_scale=1e4.  (Density f16 in BOTH.)
using CUDA, FiniteVolumeGodunovKA
const FV = FiniteVolumeGodunovKA

γ = 5f0/3f0; nd = (64,64,64); dx = Float32(2π/nd[1])
ρ0 = 5f-2                     # ~ baryon fraction: makes S = ρ·δv small
e0 = 1f-6; v0 = 1f-5          # DEEP subnormal: S ~ ρ0·v0 ~ 5e-7 (f16 subnormal ULP ~6e-8 → only ~3 bits)
GE = 1f7                      # energy lift (cold Ge underflows without it)
mk(i,j,k) = begin
    x=2f0π*(i-1)/nd[1]
    ρ=ρ0; u=v0*sin(x); v=0f0; w=0f0     # <-- the sub-normal momentum ripple to preserve
    Ge=ρ*e0; E=Ge+0.5f0*ρ*(u*u+v*v+w*w)
    (ρ, ρ*u, ρ*v, ρ*w, E, Ge)
end
U0 = [mk(i,j,k) for i in 1:nd[1], j in 1:nd[2], k in 1:nd[3]]
sde = FV.EulerDE(γ=γ, η=1f-3)

vin = [v0*sin(2f0π*(i-1)/nd[1]) for i in 1:nd[1], j in 1:nd[2], k in 1:nd[3]]   # the true vx field
# per-cell reconstruction error of vx after the f16 round-trip (the quantization-noise metric):
relerr(g,ms) = (W=FV.primitives(g; ge_scale=GE, mom_scale=ms); vx=[w[2] for w in W];
                Float64(maximum(abs.(vx .- vin)))/Float64(v0))
mass(g,ms) = Float64(FV.conserved_total(g; ge_scale=GE, mom_scale=ms)[1])

gno = FV.Grid3DCuMarch(sde, copy(U0); dx=dx, de_prec=:f16, store=:f16, ge_scale=GE, mom_scale=1f0,  scratch=:minimal)
gms = FV.Grid3DCuMarch(sde, copy(U0); dx=dx, de_prec=:f16, store=:f16, ge_scale=GE, mom_scale=1f4,  scratch=:minimal)
en, em = relerr(gno,1f0), relerr(gms,1f4)
println("init  vx max-rel-recon-error (0=perfect):  mom_scale=1 -> ", en, "    mom_scale=1e4 -> ", em)
finite = all(w->all(isfinite,w), FV.primitives(gms; ge_scale=GE, mom_scale=1f4))
ok = finite && (em < 0.1) && (en > 3*em)
println(ok ? "SMOKE PASS ✅  (MOM_SCALE preserves the deep-subnormal velocity; mom_scale=1 quantizes it badly)" :
             "SMOKE FAIL ❌  (mom_scale=1 relerr=$en, mom_scale=1e4 relerr=$em)")
