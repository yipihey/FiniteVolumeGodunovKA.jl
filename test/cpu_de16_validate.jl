# CPU emulation of the Metal f16-STORAGE dual-energy split path: U/Unew live in Float16 (GE_SCALE-lifted
# energies), each step reads f16->f32 (un-lift), runs the f32 split sweeps + per-cell PdV+switch, writes
# back f16 (lift). This is exactly the ld_de16s_U/st_de16s_U convention, on the CPU split scheme.
using FiniteVolumeGodunovKA
const FV = FiniteVolumeGodunovKA

const GS = 1f7
# ld/st conventions (mirror ld_de16s_U / st_de16s_U): energies (slots 5,6 1-based) ×GS in store, /GS on load.
@inline ld_de16s(Uh::NTuple{N,Float16}) where {N} =
    ntuple(c -> (c==5||c==6) ? Float32(Uh[c])/GS : Float32(Uh[c]), Val(N))
@inline st_de16s(U::NTuple{N,Float32}) where {N} =
    ntuple(c -> (c==5||c==6) ? Float16(U[c]*GS) : Float16(U[c]), Val(N))

@inline function de_pdv_switch(s, Un::NTuple{NV,Float32}, divv_dx, λ) where {NV}
    ρ, mx, my, mz, E, Ge = Un[1],Un[2],Un[3],Un[4],Un[5],Un[6]
    P  = (s.γ - 1f0)*Ge
    Ge = Ge - λ*P*divv_dx
    irho = inv(ρ); ke = 0.5f0*irho*(mx*mx+my*my+mz*mz)
    ratio = E > 0f0 ? (E-ke)/E : -1f0
    if ratio > s.η
        Ged = E - ke; Ge = Ged > 0f0 ? Ged : Ge; E = ke + Ge
    else
        E = ke + Ge
    end
    ntuple(c -> c==5 ? E : c==6 ? Ge : Un[c], Val(NV))
end

# one full f16-storage step: unlift -> Grid3D f32 sweeps -> PdV+switch -> lift back
function de16s_step!(Uf16::Array{<:NTuple,3}, s, dx, dt, rev, NV)
    nx,ny,nz = size(Uf16)
    U32 = [ld_de16s(Uf16[i,j,k]) for i in 1:nx,j in 1:ny,k in 1:nz]
    g = FV.Grid3D(s, U32; dx=dx,dy=dx,dz=dx, bc=:periodic, rsol=FV.LLF())
    FV.step!(g, dt; rev=rev)
    U = g.U
    λx = Float32(dt)/dx; gi(i,n)=mod1(i,n)
    @inbounds for k in 1:nz,j in 1:ny,i in 1:nx
        uxp=U[gi(i+1,nx),j,k][2]/U[gi(i+1,nx),j,k][1]; uxm=U[gi(i-1,nx),j,k][2]/U[gi(i-1,nx),j,k][1]
        vyp=U[i,gi(j+1,ny),k][3]/U[i,gi(j+1,ny),k][1]; vym=U[i,gi(j-1,ny),k][3]/U[i,gi(j-1,ny),k][1]
        wzp=U[i,j,gi(k+1,nz)][4]/U[i,j,gi(k+1,nz)][1]; wzm=U[i,j,gi(k-1,nz)][4]/U[i,j,gi(k-1,nz)][1]
        divv_dx=0.5f0*((uxp-uxm)+(vyp-vym)+(wzp-wzm))
        Uf16[i,j,k] = st_de16s(de_pdv_switch(s, U[i,j,k], divv_dx, λx))
    end
end

println("="^70)
println("TEST 4: f16-STORAGE split dual-energy on cold gas + advected colour (Metal-emulation)")
println("="^70)
γd=5f0/3f0; nd=(32,32,32); dxd=Float32(2π/nd[1]); ρ0=0.17f0; e0=6.96f-8; v0=4f-4
sph(i,j,k)=((i-16f0)^2+(j-16f0)^2+(k-16f0)^2)<6f0^2
mkc(i,j,k)=begin
    x=2f0π*(i-1)/nd[1]; y=2f0π*(j-1)/nd[2]; z=2f0π*(k-1)/nd[3]
    ρ=ρ0*(1f0+0.03f0*sin(x)); u=v0*(1f0+0.1f0*sin(y)); v=v0*0.1f0*cos(z); w=v0*0.1f0*sin(x+y)
    eint=e0*(1f0+0.02f0*cos(z)); Ge=ρ*eint; E=Ge+0.5f0*ρ*(u*u+v*v+w*w)
    (ρ,ρ*u,ρ*v,ρ*w,E,Ge, ρ*(sph(i,j,k) ? 0.09f0 : 0.05f0))
end
sdc=EulerDEColors{1}(γ=γd, η=1f-3)
U0=[Float32.(mkc(i,j,k)) for i in 1:nd[1],j in 1:nd[2],k in 1:nd[3]]
# f16-storage buffer
Uf16=[st_de16s(U0[i,j,k]) for i in 1:nd[1],j in 1:nd[2],k in 1:nd[3]]
# memory check: f16 buffer is half the bytes of f32
b16 = sizeof(eltype(Uf16))*length(Uf16)
b32 = sizeof(NTuple{7,Float32})*length(U0)
println("  buffer bytes: f16=", b16, "  f32=", b32, "  ratio=", b16/b32)
@assert b16 == b32 ÷ 2 "f16 storage must halve the buffer"
X0=[U0[i,j,k][7]/U0[i,j,k][1] for i in 1:nd[1],j in 1:nd[2],k in 1:nd[3]]
# CFL from initial state
g0=FV.Grid3D(sdc, [ld_de16s(Uf16[i,j,k]) for i in 1:nd[1],j in 1:nd[2],k in 1:nd[3]]; dx=dxd,dy=dxd,dz=dxd, bc=:periodic, rsol=FV.LLF())
dtd = 0.3f0*dxd/FV.max_wavespeed(g0)
for n in 0:19
    de16s_step!(Uf16, sdc, dxd, dtd, isodd(n), 7)
end
Wf = [FV.cons2prim(sdc, ld_de16s(Uf16[i,j,k])) for i in 1:nd[1],j in 1:nd[2],k in 1:nd[3]]
allfin = all(w->all(isfinite,w), Wf)
e16 = [w[6] for w in Wf]; X16=[w[7] for w in Wf]
println("  finite=", allfin, "  ρmin=", minimum(w[1] for w in Wf))
println("  eint range=[", minimum(e16), ", ", maximum(e16), "]  target=", e0)
println("  X range=[", minimum(X16), ", ", maximum(X16), "]  max|X-X0|=", maximum(abs.(X16.-X0)))
@assert allfin
@assert all(w->w[1]>0, Wf)
@assert all(0.5f0*e0 < e < 1.5f0*e0 for e in e16) "cold eint preserved in f16 storage"
@assert minimum(e16) > 1f-9 "eint not flushed to subnormal 0"
@assert all(0f0 <= x <= 1f0 for x in X16) "colour bounded 0<=X<=1"
@assert maximum(abs.(X16.-X0)) > 1f-2 "colour ADVECTED"
println("  PASS: f16-storage split dual-energy — no NaN, eint preserved, colour advects, buffer halved.")
println("="^70)
println("ALL f16-STORAGE CPU EMULATION VALIDATIONS PASSED")
println("="^70)
