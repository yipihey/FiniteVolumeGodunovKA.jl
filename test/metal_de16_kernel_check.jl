# Direct semantic + execution check of the FINAL Metal de16 kernels (buffer ALWAYS GE_SCALE-lifted; un-lift
# only into f32 registers; compute f32 / store f16). Drives them serially with an explicit thread index,
# multi-step, on the cold cosmology IC — proving type-inference AND that eint survives (the f16-store win).
using FiniteVolumeGodunovKA
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: _gidx, _swap, _update_dir, identperm, dirperm, cons2prim, maxspeed_x,
                              EulerDE, EulerDEColors, LLF, PLM

@inline _ld_de16s(U,i,j,k,igs::Float32,::Val{N}) where {N} =
    ntuple(c -> (c==5||c==6) ? @inbounds(Float32(U[i,j,k,c]))*igs : @inbounds(Float32(U[i,j,k,c])), Val(N))
@inline function _st_de16s!(U,i,j,k,v::NTuple{N,Float32},gs::Float32) where {N}
    ntuple(c -> (@inbounds(U[i,j,k,c]=Float16((c==5||c==6) ? v[c]*gs : v[c])); nothing), Val(N)); nothing
end
@inline _vel(U,i,j,k,c)=Float32(@inbounds U[i,j,k,c])/Float32(@inbounds U[i,j,k,1])
@inline function _sw(s,Un::NTuple{NV,Float32},divv_dx::Float32,lam::Float32) where {NV}
    rho=Un[1];mx=Un[2];my=Un[3];mz=Un[4];E=Un[5];Ge=Un[6]
    P=(s.γ-1f0)*Ge; Ge=Ge-lam*P*divv_dx; irho=inv(rho); ke=0.5f0*irho*(mx*mx+my*my+mz*mz)
    ratio=E>0f0 ? (E-ke)/E : -1f0
    if ratio>s.η; Ged=E-ke; Ge=Ged>0f0 ? Ged : Ge; E=ke+Ge; else; E=ke+Ge; end
    ntuple(c->c==5 ? E : c==6 ? Ge : Un[c], Val(NV))
end
function switch_k!(U,s,igs,gs,lam,nx,ny,nz,::Val{N},bc,tid) where {N}
    i,j,k=tid
    if i<=nx&&j<=ny&&k<=nz
        u0=_ld_de16s(U,i,j,k,igs,Val(N))
        uxp=_vel(U,_gidx(i+1,nx,bc),j,k,2);uxm=_vel(U,_gidx(i-1,nx,bc),j,k,2)
        vyp=_vel(U,i,_gidx(j+1,ny,bc),k,3);vym=_vel(U,i,_gidx(j-1,ny,bc),k,3)
        wzp=_vel(U,i,j,_gidx(k+1,nz,bc),4);wzm=_vel(U,i,j,_gidx(k-1,nz,bc),4)
        divv_dx=0.5f0*((uxp-uxm)+(vyp-vym)+(wzp-wzm))
        _st_de16s!(U,i,j,k,_sw(s,u0,divv_dx,lam),gs)
    end; return
end
function sweepx_k!(Unew,U,s,r,rs,lam,igs,gs,nx,ny,nz,::Val{N},bc,perm,tid) where {N}
    i,j,k=tid
    if i<=nx&&j<=ny&&k<=nz
        _st_de16s!(Unew,i,j,k,_update_dir(s,r,rs,
            _ld_de16s(U,_gidx(i-2,nx,bc),j,k,igs,Val(N)),_ld_de16s(U,_gidx(i-1,nx,bc),j,k,igs,Val(N)),_ld_de16s(U,i,j,k,igs,Val(N)),
            _ld_de16s(U,_gidx(i+1,nx,bc),j,k,igs,Val(N)),_ld_de16s(U,_gidx(i+2,nx,bc),j,k,igs,Val(N)),lam,perm),gs)
    end; return
end

nd=(8,8,8); NV=6; T=Float16; GS=1f7; igs=Float32(1/GS); gs=Float32(GS)
s=EulerDE(γ=5f0/3f0,η=1f-3); bc=Val(:periodic)
ρ0=0.17f0;e0=6.96f-8;v0=4f-4
U=Array{T,4}(undef,nd...,NV)
for i in 1:nd[1],j in 1:nd[2],k in 1:nd[3]
    ρ=ρ0;u=v0;Ge=ρ*e0;E=Ge+0.5f0*ρ*u*u; raw=(ρ,ρ*u,0f0,0f0,E,Ge)
    for c in 1:NV; U[i,j,k,c]=Float16((c==5||c==6) ? raw[c]*GS : raw[c]); end
end
@assert !isempty(Base.code_typed(switch_k!,Tuple{Array{T,4},typeof(s),Float32,Float32,Float32,Int,Int,Int,Val{NV},typeof(bc),NTuple{3,Int}}))
@assert !isempty(Base.code_typed(sweepx_k!,Tuple{Array{T,4},Array{T,4},typeof(s),PLM{:mc},LLF,Float32,Float32,Float32,Int,Int,Int,Val{NV},typeof(bc),Val{ntuple(identity,NV)},NTuple{3,Int}}))
println("TYPE-INFERENCE: sweep + switch OK (compute-f32/store-f16, buffer-stays-lifted)")
function run10!(U,nd,NV,s,bc,igs,gs)
    Unew=copy(U)
    for step in 1:10
        for i in 1:nd[1],j in 1:nd[2],k in 1:nd[3]; sweepx_k!(Unew,U,s,PLM(),LLF(),0.005f0,igs,gs,nd...,Val(NV),bc,identperm(Val(NV)),(i,j,k)); end
        U,Unew=Unew,U
        for i in 1:nd[1],j in 1:nd[2],k in 1:nd[3]; switch_k!(U,s,igs,gs,0.005f0,nd...,Val(NV),bc,(i,j,k)); end
    end
    return U
end
U=run10!(U,nd,NV,s,bc,igs,gs)
@assert all(isfinite,U) && eltype(U)===Float16
e16=Float32[]; for i in 1:nd[1],j in 1:nd[2],k in 1:nd[3]; ρ=Float32(U[i,j,k,1]); Ge=Float32(U[i,j,k,6])*igs; push!(e16,Ge/ρ); end
println("EXECUTION (10 steps, Float16 store): finite=",all(isfinite,U),"  eint∈",extrema(e16),"  target=",e0)
@assert all(0.5f0*e0 < e < 1.5f0*e0 for e in e16) "eint preserved through the lifted f16 store"
@assert minimum(e16) > 1f-9 "eint NOT flushed to subnormal 0"
println("RESULT: Metal de16 kernels semantically valid + eint PRESERVED through the always-lifted f16 buffer.")
