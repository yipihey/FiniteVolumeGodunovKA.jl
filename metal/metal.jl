# ============================================================================================
# Metal (Apple GPU) backend for FiniteVolumeGodunovKA — 1D / 2D / 3D, at parity with the CUDA backends.
#
# Metal.jl is macOS/Apple-Silicon only. Each kernel BODY here is identical to the corresponding
# CUDA backend (src/backend_cuda{,_2d,_3d}.jl): the same branch-free `_update_dir` physics compiles
# for Metal via GPUCompiler in Float32. Only three things differ from CUDA: the launch macro
# (`Metal.@metal threads= groups=`), the thread-index intrinsics (`thread_position_in_grid_*d`),
# and the array type (`MtlArray`). The dimensional-split scheme, the alternating-Strang `rev`, the
# `has_source` guard, and the dynamic-`ch` `prestep` are all mirrored exactly.
#
#     The transpile-to-nvcc backend (`Grid3DCuMarch`) has NO Metal analog — it shells out to `nvcc` to
#     build a CUDA-C `.so`. The Metal equivalent would transpile to MSL and build via `metallib`; that is
#     left as future work (and is the "does Metal.jl have a CUDA-sized codegen gap?" question below).
#
# Validate on a Mac:
#         ] add Metal
#         using FiniteVolumeGodunovKA, Metal
#         include("metal/metal.jl")
#         metal_selfcheck_1d(); metal_selfcheck_2d(); metal_selfcheck_3d()   # all must be max|Δ| = 0
#         metal_selfcheck_3d_colors()                                        # packed UInt16 colors
#
# This is intentionally NOT a package dependency (it would break the Linux package's resolution).
# Productionization on the Mac: move it to a weakdep + package extension `ext/MetalExt.jl`.
#
# OPEN DESIGN QUESTION (from DESIGN_fvkernel.md): compare this Metal.jl-native backend to the Metal
# bandwidth roofline (and a hand-MSL kernel if you write one). If Metal.jl is close to native → keep it;
# if it has a CUDA-sized codegen gap → the transpile escape hatch must target MSL too, not just CUDA-C.
# ============================================================================================

using FiniteVolumeGodunovKA, Metal
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: _update_dir, _gidx, _swap, identperm, dirperm, has_source,
                              cons2prim, source, fastspeed_x, maxspeed_x, _halfstep, riemann,
                              _pack_colors, _update_packed_color, unpack_color_fraction,
                              FVSystem, PLM, HLLC

# Metal thread-index intrinsics return 1-based global grid positions.
@inline _mtid1() = thread_position_in_grid_1d()
@inline _mtid2() = thread_position_in_grid_2d()
@inline _mtid3() = thread_position_in_grid_3d()

# =================================== 1D ===================================
# Mirrors src/backend_cuda.jl: one x-sweep per step (1D has no dimensional splitting), maxspeed CFL.

@inline _mread1(U, i, ::Val{N}) where {N} = ntuple(k -> @inbounds(U[i, k]), Val(N))
@inline _mwrite1!(U, i, v::NTuple{N}) where {N} = (ntuple(k -> (@inbounds(U[i, k] = v[k]); nothing), Val(N)); nothing)

function _mstep1_kernel!(Unew, U, s, r, rs, λ, nx, ::Val{N}, bc, perm) where {N}
    i = _mtid1()
    if i <= nx
        _mwrite1!(Unew, i, _update_dir(s, r, rs,
            _mread1(U,_gidx(i-2,nx,bc),Val(N)), _mread1(U,_gidx(i-1,nx,bc),Val(N)), _mread1(U,i,Val(N)),
            _mread1(U,_gidx(i+1,nx,bc),Val(N)), _mread1(U,_gidx(i+2,nx,bc),Val(N)), λ, perm))
    end
    return
end
function _mspeed1_kernel!(spd, U, s, nx, ::Val{N}) where {N}
    i = _mtid1()
    (i <= nx) && (@inbounds spd[i] = maxspeed_x(s, cons2prim(s, _mread1(U, i, Val(N)))))
    return
end
function _msource1_kernel!(U, s, dt, nx, ::Val{N}) where {N}
    i = _mtid1()
    (i <= nx) && _mwrite1!(U, i, source(s, _mread1(U, i, Val(N)), dt))
    return
end

mutable struct Grid1DMtl{N,S<:FVSystem,R,RS}
    sys::S; recon::R; rsol::RS
    U::MtlArray{Float32,2}; Unew::MtlArray{Float32,2}; spd::MtlArray{Float32,1}
    nx::Int; dx::Float32; bc::Symbol; cfl::Float32
end
function Grid1DMtl(sys::FVSystem, U0::Vector{NTuple{N,T}};
                   dx, bc::Symbol = :outflow, recon = PLM(), rsol = HLLC(), cfl = 0.4f0) where {N,T}
    nx = length(U0); Uh = Matrix{Float32}(undef, nx, N)
    @inbounds for i in 1:nx, k in 1:N; Uh[i, k] = Float32(U0[i][k]); end
    U = MtlArray(Uh)
    Grid1DMtl{N,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, U, similar(U), Metal.zeros(Float32, nx), nx, Float32(dx), bc, Float32(cfl))
end
@inline _mcfg1(nx) = (256, cld(nx, 256))

function mstep1d!(g::Grid1DMtl{N}, dt) where {N}
    thr, grp = _mcfg1(g.nx)
    Metal.@metal threads=thr groups=grp _mstep1_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt)/g.dx, g.nx, Val(N), Val(g.bc), identperm(Val(N)))
    g.U, g.Unew = g.Unew, g.U
    has_source(g.sys) && Metal.@metal threads=thr groups=grp _msource1_kernel!(g.U, g.sys, Float32(dt), g.nx, Val(N))
    return g
end
function mmax_wavespeed_1d(g::Grid1DMtl{N}) where {N}
    thr, grp = _mcfg1(g.nx)
    Metal.@metal threads=thr groups=grp _mspeed1_kernel!(g.spd, g.U, g.sys, g.nx, Val(N))
    return maximum(g.spd)
end
function mevolve1d!(g::Grid1DMtl, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        c = mmax_wavespeed_1d(g); g.sys = FV.prestep(g.sys, c)
        dt = min(g.cfl * g.dx / c, tend - t); mstep1d!(g, dt); t += dt; n += 1
    end
    return g
end
mprimitives_1d(g::Grid1DMtl{N}) where {N} =
    (Uh = Array(g.U); [cons2prim(g.sys, ntuple(k -> Uh[i, k], Val(N))) for i in 1:g.nx])

# =================================== 2D ===================================
# Mirrors src/backend_cuda_2d.jl: 2 full-dt sweeps, alternating order (rev), source skipped when none.

@inline _mread2(U, i, j, ::Val{N}) where {N} = ntuple(c -> @inbounds(U[i, j, c]), Val(N))
@inline _mwrite2!(U, i, j, v::NTuple{N}) where {N} = (ntuple(c -> (@inbounds(U[i, j, c] = v[c]); nothing), Val(N)); nothing)

function _msweepx2_kernel!(Unew, U, s, r, rs, λ, nx, ny, ::Val{N}, bc, perm) where {N}
    i, j = _mtid2()
    if i <= nx && j <= ny
        _mwrite2!(Unew, i, j, _update_dir(s, r, rs,
            _mread2(U,_gidx(i-2,nx,bc),j,Val(N)), _mread2(U,_gidx(i-1,nx,bc),j,Val(N)), _mread2(U,i,j,Val(N)),
            _mread2(U,_gidx(i+1,nx,bc),j,Val(N)), _mread2(U,_gidx(i+2,nx,bc),j,Val(N)), λ, perm))
    end
    return
end
function _msweepy2_kernel!(Unew, U, s, r, rs, λ, nx, ny, ::Val{N}, bc, perm) where {N}
    i, j = _mtid2()
    if i <= nx && j <= ny
        _mwrite2!(Unew, i, j, _update_dir(s, r, rs,
            _mread2(U,i,_gidx(j-2,ny,bc),Val(N)), _mread2(U,i,_gidx(j-1,ny,bc),Val(N)), _mread2(U,i,j,Val(N)),
            _mread2(U,i,_gidx(j+1,ny,bc),Val(N)), _mread2(U,i,_gidx(j+2,ny,bc),Val(N)), λ, perm))
    end
    return
end
function _msource2_kernel!(U, s, dt, nx, ny, ::Val{N}) where {N}
    i, j = _mtid2()
    (i <= nx && j <= ny) && _mwrite2!(U, i, j, source(s, _mread2(U,i,j,Val(N)), dt))
    return
end
function _mspeed2_kernel!(spd, U, s, nx, ny, ::Val{N}, py) where {N}
    i, j = _mtid2()
    if i <= nx && j <= ny
        W = cons2prim(s, _mread2(U, i, j, Val(N)))
        @inbounds spd[i, j] = max(fastspeed_x(s, W), fastspeed_x(s, _swap(W, py)))
    end
    return
end

mutable struct Grid2DMtl{N,S<:FVSystem,R,RS}
    sys::S; recon::R; rsol::RS
    U::MtlArray{Float32,3}; Unew::MtlArray{Float32,3}; spd::MtlArray{Float32,2}
    nx::Int; ny::Int; dx::Float32; dy::Float32; bc::Symbol; cfl::Float32
end
function Grid2DMtl(sys::FVSystem, U0::Matrix{NTuple{N,T}};
                   dx, dy, bc::Symbol = :outflow, recon = PLM(), rsol = HLLC(), cfl = 0.4f0) where {N,T}
    nx, ny = size(U0); Uh = Array{Float32,3}(undef, nx, ny, N)
    @inbounds for j in 1:ny, i in 1:nx, c in 1:N; Uh[i,j,c] = Float32(U0[i,j][c]); end
    U = MtlArray(Uh)
    Grid2DMtl{N,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, U, similar(U), Metal.zeros(Float32, nx, ny), nx, ny, Float32(dx), Float32(dy), bc, Float32(cfl))
end
@inline _mcfg2(nx, ny) = ((16, 16), (cld(nx, 16), cld(ny, 16)))

function mstep2d!(g::Grid2DMtl{N}, dt; rev::Bool = false) where {N}
    thr, grp = _mcfg2(g.nx, g.ny); bc = Val(g.bc); px = identperm(Val(N)); py = dirperm(g.sys, N, 2)
    swx() = (Metal.@metal threads=thr groups=grp _msweepx2_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt)/g.dx, g.nx, g.ny, Val(N), bc, px); (g.U, g.Unew) = (g.Unew, g.U))
    swy() = (Metal.@metal threads=thr groups=grp _msweepy2_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt)/g.dy, g.nx, g.ny, Val(N), bc, py); (g.U, g.Unew) = (g.Unew, g.U))
    rev ? (swy(); swx()) : (swx(); swy())
    has_source(g.sys) && Metal.@metal threads=thr groups=grp _msource2_kernel!(g.U, g.sys, Float32(dt), g.nx, g.ny, Val(N))
    return g
end
function mmax_wavespeed_2d(g::Grid2DMtl{N}) where {N}
    thr, grp = _mcfg2(g.nx, g.ny)
    Metal.@metal threads=thr groups=grp _mspeed2_kernel!(g.spd, g.U, g.sys, g.nx, g.ny, Val(N), dirperm(g.sys, N, 2))
    return maximum(g.spd)
end
function mevolve2d!(g::Grid2DMtl, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        c = mmax_wavespeed_2d(g); g.sys = FV.prestep(g.sys, c)
        dt = min(g.cfl * min(g.dx, g.dy) / c, tend - t); mstep2d!(g, dt; rev = isodd(n)); t += dt; n += 1
    end
    return g
end
mprimitives_2d(g::Grid2DMtl{N}) where {N} =
    (Uh = Array(g.U); [cons2prim(g.sys, ntuple(c -> Uh[i,j,c], Val(N))) for i in 1:g.nx, j in 1:g.ny])

# =================================== 3D ===================================
# Mirrors src/backend_cuda_3d.jl: symmetric Strang x·y·z (rev → z·y·x) + source skipped when none.

@inline _mread3(U, i, j, k, ::Val{N}) where {N} = ntuple(c -> @inbounds(U[i, j, k, c]), Val(N))
@inline _mwrite3!(U, i, j, k, v::NTuple{N}) where {N} = (ntuple(c -> (@inbounds(U[i, j, k, c] = v[c]); nothing), Val(N)); nothing)

function _msweepx3_kernel!(Unew, U, s, r, rs, λ, nx, ny, nz, ::Val{N}, bc, perm) where {N}
    i, j, k = _mtid3()
    if i <= nx && j <= ny && k <= nz
        _mwrite3!(Unew, i, j, k, _update_dir(s, r, rs,
            _mread3(U,_gidx(i-2,nx,bc),j,k,Val(N)), _mread3(U,_gidx(i-1,nx,bc),j,k,Val(N)), _mread3(U,i,j,k,Val(N)),
            _mread3(U,_gidx(i+1,nx,bc),j,k,Val(N)), _mread3(U,_gidx(i+2,nx,bc),j,k,Val(N)), λ, perm))
    end
    return
end
function _msweepy3_kernel!(Unew, U, s, r, rs, λ, nx, ny, nz, ::Val{N}, bc, perm) where {N}
    i, j, k = _mtid3()
    if i <= nx && j <= ny && k <= nz
        _mwrite3!(Unew, i, j, k, _update_dir(s, r, rs,
            _mread3(U,i,_gidx(j-2,ny,bc),k,Val(N)), _mread3(U,i,_gidx(j-1,ny,bc),k,Val(N)), _mread3(U,i,j,k,Val(N)),
            _mread3(U,i,_gidx(j+1,ny,bc),k,Val(N)), _mread3(U,i,_gidx(j+2,ny,bc),k,Val(N)), λ, perm))
    end
    return
end
function _msweepz3_kernel!(Unew, U, s, r, rs, λ, nx, ny, nz, ::Val{N}, bc, perm) where {N}
    i, j, k = _mtid3()
    if i <= nx && j <= ny && k <= nz
        _mwrite3!(Unew, i, j, k, _update_dir(s, r, rs,
            _mread3(U,i,j,_gidx(k-2,nz,bc),Val(N)), _mread3(U,i,j,_gidx(k-1,nz,bc),Val(N)), _mread3(U,i,j,k,Val(N)),
            _mread3(U,i,j,_gidx(k+1,nz,bc),Val(N)), _mread3(U,i,j,_gidx(k+2,nz,bc),Val(N)), λ, perm))
    end
    return
end
function _msource3_kernel!(U, s, dt, nx, ny, nz, ::Val{N}) where {N}
    i, j, k = _mtid3()
    (i <= nx && j <= ny && k <= nz) && _mwrite3!(U, i, j, k, source(s, _mread3(U,i,j,k,Val(N)), dt))
    return
end
function _mspeed3_kernel!(spd, U, s, nx, ny, nz, ::Val{N}, py, pz) where {N}
    i, j, k = _mtid3()
    if i <= nx && j <= ny && k <= nz
        W = cons2prim(s, _mread3(U, i, j, k, Val(N)))
        @inbounds spd[i, j, k] = max(fastspeed_x(s, W), fastspeed_x(s, _swap(W, py)), fastspeed_x(s, _swap(W, pz)))
    end
    return
end

@inline _mread3_axis(U, i, j, k, o, nx, ny, nz, bc, ::Val{1}, ::Val{N}) where {N} =
    _mread3(U, _gidx(i + o, nx, bc), j, k, Val(N))
@inline _mread3_axis(U, i, j, k, o, nx, ny, nz, bc, ::Val{2}, ::Val{N}) where {N} =
    _mread3(U, i, _gidx(j + o, ny, bc), k, Val(N))
@inline _mread3_axis(U, i, j, k, o, nx, ny, nz, bc, ::Val{3}, ::Val{N}) where {N} =
    _mread3(U, i, j, _gidx(k + o, nz, bc), Val(N))

@inline _mwrite3_axis!(U, i, j, k, o, v, ::Val{1}) = _mwrite3!(U, i + o, j, k, v)
@inline _mwrite3_axis!(U, i, j, k, o, v, ::Val{2}) = _mwrite3!(U, i, j + o, k, v)
@inline _mwrite3_axis!(U, i, j, k, o, v, ::Val{3}) = _mwrite3!(U, i, j, k + o, v)

@inline _mcread3_axis(C, i, j, k, o, q, nx, ny, nz, bc, ::Val{1}) =
    @inbounds C[_gidx(i + o, nx, bc), j, k, q]
@inline _mcread3_axis(C, i, j, k, o, q, nx, ny, nz, bc, ::Val{2}) =
    @inbounds C[i, _gidx(j + o, ny, bc), k, q]
@inline _mcread3_axis(C, i, j, k, o, q, nx, ny, nz, bc, ::Val{3}) =
    @inbounds C[i, j, _gidx(k + o, nz, bc), q]

@inline _maxis_coord(i, j, k, ::Val{1}) = (2 * (i - 1) + 1, j, k)
@inline _maxis_coord(i, j, k, ::Val{2}) = (i, 2 * (j - 1) + 1, k)
@inline _maxis_coord(i, j, k, ::Val{3}) = (i, j, 2 * (k - 1) + 1)

@inline _maxis_valid(i, j, k, nx, ny, nz, ::Val{1}) = i <= nx && j <= ny && k <= nz
@inline _maxis_valid(i, j, k, nx, ny, nz, ::Val{2}) = i <= nx && j <= ny && k <= nz
@inline _maxis_valid(i, j, k, nx, ny, nz, ::Val{3}) = i <= nx && j <= ny && k <= nz

@inline function _mhalf_axis(U, s, r, i, j, k, o, λ, nx, ny, nz, bc, axis, perm, ::Val{N}) where {N}
    um = _mread3_axis(U, i, j, k, o - 1, nx, ny, nz, bc, axis, Val(N))
    u0 = _mread3_axis(U, i, j, k, o,     nx, ny, nz, bc, axis, Val(N))
    up = _mread3_axis(U, i, j, k, o + 1, nx, ny, nz, bc, axis, Val(N))
    return _halfstep(s, r, _swap(um, perm), _swap(u0, perm), _swap(up, perm), λ)
end

function _mstep3_pair_kernel!(Unew, U, s, r, rs, λ, nx, ny, nz, ::Val{N}, bc, axis::Val{A}, perm) where {N,A}
    ti, tj, tk = _mtid3()
    i, j, k = _maxis_coord(ti, tj, tk, axis)
    if _maxis_valid(i, j, k, nx, ny, nz, axis)
        wlm, wrm = _mhalf_axis(U, s, r, i, j, k, -1, λ, nx, ny, nz, bc, axis, perm, Val(N))
        wl0, wr0 = _mhalf_axis(U, s, r, i, j, k,  0, λ, nx, ny, nz, bc, axis, perm, Val(N))
        wl1, wr1 = _mhalf_axis(U, s, r, i, j, k,  1, λ, nx, ny, nz, bc, axis, perm, Val(N))
        wl2, wr2 = _mhalf_axis(U, s, r, i, j, k,  2, λ, nx, ny, nz, bc, axis, perm, Val(N))
        f0 = riemann(rs, s, wrm, wl0)
        f1 = riemann(rs, s, wr0, wl1)
        f2 = riemann(rs, s, wr1, wl2)
        u0 = _mread3_axis(U, i, j, k, 0, nx, ny, nz, bc, axis, Val(N))
        _mwrite3_axis!(Unew, i, j, k, 0, u0 .- λ .* _swap(f1 .- f0, perm), axis)
        if A == 1
            ok2 = i + 1 <= nx
        elseif A == 2
            ok2 = j + 1 <= ny
        else
            ok2 = k + 1 <= nz
        end
        if ok2
            u1 = _mread3_axis(U, i, j, k, 1, nx, ny, nz, bc, axis, Val(N))
            _mwrite3_axis!(Unew, i, j, k, 1, u1 .- λ .* _swap(f2 .- f1, perm), axis)
        end
    end
    return
end

function _mcolor3_kernel!(Cnew, C, Unew, U, s, r, rs, λ, nx, ny, nz, ::Val{N}, bc, axis::Val{A}, perm, q) where {N,A}
    i, j, k = _mtid3()
    if i <= nx && j <= ny && k <= nz
        u0 = _mread3_axis(U, i, j, k, 0, nx, ny, nz, bc, axis, Val(N))
        @inbounds Cnew[i, j, k, q] = _update_packed_color(s, r, rs,
            _mread3_axis(U, i, j, k, -2, nx, ny, nz, bc, axis, Val(N)),
            _mread3_axis(U, i, j, k, -1, nx, ny, nz, bc, axis, Val(N)),
            u0,
            _mread3_axis(U, i, j, k,  1, nx, ny, nz, bc, axis, Val(N)),
            _mread3_axis(U, i, j, k,  2, nx, ny, nz, bc, axis, Val(N)),
            _mcread3_axis(C, i, j, k, -2, q, nx, ny, nz, bc, axis),
            _mcread3_axis(C, i, j, k, -1, q, nx, ny, nz, bc, axis),
            (@inbounds C[i, j, k, q]),
            _mcread3_axis(C, i, j, k,  1, q, nx, ny, nz, bc, axis),
            _mcread3_axis(C, i, j, k,  2, q, nx, ny, nz, bc, axis),
            λ, perm, _mread3(Unew, i, j, k, Val(N))[1])
    end
    return
end

mutable struct Grid3DMtl{N,S<:FVSystem,R,RS}
    sys::S; recon::R; rsol::RS
    U::MtlArray{Float32,4}; Unew::MtlArray{Float32,4}; spd::MtlArray{Float32,3}
    colors::Union{Nothing,MtlArray{UInt16,4}}; colorst::Union{Nothing,MtlArray{UInt16,4}}
    nx::Int; ny::Int; nz::Int; dx::Float32; dy::Float32; dz::Float32; bc::Symbol; cfl::Float32
end
function Grid3DMtl(sys::FVSystem, U0::Array{NTuple{N,T},3};
                   dx, dy, dz, bc::Symbol = :outflow, recon = PLM(), rsol = HLLC(), cfl = 0.4f0,
                   colors = nothing) where {N,T}
    nx, ny, nz = size(U0); Uh = Array{Float32,4}(undef, nx, ny, nz, N)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx, c in 1:N; Uh[i,j,k,c] = Float32(U0[i,j,k][c]); end
    U = MtlArray(Uh)
    C, Ct = _pack_colors(colors, (nx, ny, nz))
    Grid3DMtl{N,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, U, similar(U), Metal.zeros(Float32, nx, ny, nz),
        C === nothing ? nothing : MtlArray(C), Ct === nothing ? nothing : MtlArray(Ct),
        nx, ny, nz, Float32(dx), Float32(dy), Float32(dz), bc, Float32(cfl))
end
@inline _mcfg3(nx, ny, nz) = ((16, 8, 2), (cld(nx, 16), cld(ny, 8), cld(nz, 2)))
@inline _mpaircfg3(nx, ny, nz, ::Val{1}) = ((16, 8, 2), (cld(cld(nx, 2), 16), cld(ny, 8), cld(nz, 2)))
@inline _mpaircfg3(nx, ny, nz, ::Val{2}) = ((16, 8, 2), (cld(nx, 16), cld(cld(ny, 2), 8), cld(nz, 2)))
@inline _mpaircfg3(nx, ny, nz, ::Val{3}) = ((16, 8, 2), (cld(nx, 16), cld(ny, 8), cld(cld(nz, 2), 2)))

function _mpair_sweep3!(g::Grid3DMtl{N}, dt, axis::Val{A}, perm) where {N,A}
    thr, grp = _mpaircfg3(g.nx, g.ny, g.nz, axis)
    λ = A == 1 ? Float32(dt)/g.dx : A == 2 ? Float32(dt)/g.dy : Float32(dt)/g.dz
    Metal.@metal threads=thr groups=grp _mstep3_pair_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        λ, g.nx, g.ny, g.nz, Val(N), Val(g.bc), axis, perm)
    if g.colors !== nothing
        cthr, cgrp = _mcfg3(g.nx, g.ny, g.nz)
        for q in 1:size(g.colors, 4)
            Metal.@metal threads=cthr groups=cgrp _mcolor3_kernel!(g.colorst, g.colors, g.Unew, g.U,
                g.sys, g.recon, g.rsol, λ, g.nx, g.ny, g.nz, Val(N), Val(g.bc), axis, perm, q)
        end
        g.colors, g.colorst = g.colorst, g.colors
    end
    g.U, g.Unew = g.Unew, g.U
    return g
end

@inline _muse_pair3(g::Grid3DMtl) = min(g.nx, g.ny, g.nz) >= 96

function _mfused_sweep3!(g::Grid3DMtl{N}, dt, axis::Val{A}, perm) where {N,A}
    thr, grp = _mcfg3(g.nx, g.ny, g.nz)
    λ = A == 1 ? Float32(dt)/g.dx : A == 2 ? Float32(dt)/g.dy : Float32(dt)/g.dz
    kern = A == 1 ? _msweepx3_kernel! : A == 2 ? _msweepy3_kernel! : _msweepz3_kernel!
    Metal.@metal threads=thr groups=grp kern(g.Unew, g.U, g.sys, g.recon, g.rsol,
        λ, g.nx, g.ny, g.nz, Val(N), Val(g.bc), perm)
    if g.colors !== nothing
        for q in 1:size(g.colors, 4)
            Metal.@metal threads=thr groups=grp _mcolor3_kernel!(g.colorst, g.colors, g.Unew, g.U,
                g.sys, g.recon, g.rsol, λ, g.nx, g.ny, g.nz, Val(N), Val(g.bc), axis, perm, q)
        end
        g.colors, g.colorst = g.colorst, g.colors
    end
    g.U, g.Unew = g.Unew, g.U
    return g
end

function mstep3d!(g::Grid3DMtl{N}, dt; rev::Bool = false) where {N}
    px = identperm(Val(N)); py = dirperm(g.sys, N, 2); pz = dirperm(g.sys, N, 3)
    if _muse_pair3(g)
        if rev
            _mpair_sweep3!(g, dt, Val(3), pz); _mpair_sweep3!(g, dt, Val(2), py); _mpair_sweep3!(g, dt, Val(1), px)
        else
            _mpair_sweep3!(g, dt, Val(1), px); _mpair_sweep3!(g, dt, Val(2), py); _mpair_sweep3!(g, dt, Val(3), pz)
        end
    else
        if rev
            _mfused_sweep3!(g, dt, Val(3), pz); _mfused_sweep3!(g, dt, Val(2), py); _mfused_sweep3!(g, dt, Val(1), px)
        else
            _mfused_sweep3!(g, dt, Val(1), px); _mfused_sweep3!(g, dt, Val(2), py); _mfused_sweep3!(g, dt, Val(3), pz)
        end
    end
    thr, grp = _mcfg3(g.nx, g.ny, g.nz)
    has_source(g.sys) && Metal.@metal threads=thr groups=grp _msource3_kernel!(g.U, g.sys, Float32(dt), g.nx, g.ny, g.nz, Val(N))
    return g
end
function mmax_wavespeed_3d(g::Grid3DMtl{N}) where {N}
    thr, grp = _mcfg3(g.nx, g.ny, g.nz)
    Metal.@metal threads=thr groups=grp _mspeed3_kernel!(g.spd, g.U, g.sys, g.nx, g.ny, g.nz, Val(N),
                                                         dirperm(g.sys, N, 2), dirperm(g.sys, N, 3))
    return maximum(g.spd)
end
function mevolve3d!(g::Grid3DMtl, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        c = mmax_wavespeed_3d(g); g.sys = FV.prestep(g.sys, c)
        dt = min(g.cfl * min(g.dx, g.dy, g.dz) / c, tend - t); mstep3d!(g, dt; rev = isodd(n)); t += dt; n += 1
    end
    return g
end
mprimitives_3d(g::Grid3DMtl{N}) where {N} =
    (Uh = Array(g.U); [cons2prim(g.sys, ntuple(c -> Uh[i,j,k,c], Val(N))) for i in 1:g.nx, j in 1:g.ny, k in 1:g.nz])

function mcolor_fractions_3d(g::Grid3DMtl)
    g.colors === nothing && return nothing
    C = Array(g.colors)
    F = Array{Float32}(undef, size(C))
    @inbounds for I in eachindex(C)
        F[I] = unpack_color_fraction(C[I])
    end
    return F
end

# =============================== validation (run on a Mac) ===============================
# Each Metal backend must be bit-identical to its scalar reference (same branch-free physics, Float32).

function metal_selfcheck_1d()
    s = FV.Euler(γ = 1.4f0); n = 256; d = 1f0/n
    U0 = [FV.prim2cons(s, (1f0+0.3f0*sinpi(2f0*(i-1f0)/n), 0.5f0,0f0,0f0, 1f0)) for i in 1:n]
    gsc = FV.Grid1D(s, copy(U0); dx=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    gm  = Grid1DMtl(s, copy(U0); dx=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    for _ in 1:20; FV.step!(gsc, 0.1f0*d); mstep1d!(gm, 0.1f0*d); end
    Wc = FV.primitives(gsc); Wm = mprimitives_1d(gm)
    md = maximum(maximum(abs.(Wc[i] .- Wm[i])) for i in 1:n)
    println("Metal ≡ scalar Grid1D max|Δ| = ", md, md == 0f0 ? "  ✓ bit-identical" : "  (investigate)"); md
end

function metal_selfcheck_2d()
    s = FV.Euler(γ = 1.4f0); n = 64; d = 1f0/n
    U0 = [FV.prim2cons(s, (1f0+0.3f0*sinpi(2f0*(i+j-1f0)/n), 0.5f0,0.3f0,0f0, 1f0)) for i in 1:n, j in 1:n]
    gsc = FV.Grid2D(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    gm  = Grid2DMtl(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    for nn in 0:14; FV.step!(gsc, 0.1f0*d; rev=isodd(nn)); mstep2d!(gm, 0.1f0*d; rev=isodd(nn)); end
    Wc = FV.primitives(gsc); Wm = mprimitives_2d(gm)
    md = maximum(maximum(abs.(Wc[i,j] .- Wm[i,j])) for i in 1:n, j in 1:n)
    println("Metal ≡ scalar Grid2D max|Δ| = ", md, md == 0f0 ? "  ✓ bit-identical" : "  (investigate)"); md
end

function metal_selfcheck_3d()
    s = FV.Euler(γ = 1.4f0); n = 32; d = 1f0/n
    U0 = [FV.prim2cons(s, (1f0+0.3f0*sinpi(2f0*(i+j+k-1f0)/n), 0.5f0,0.3f0,0.2f0, 1f0)) for i in 1:n, j in 1:n, k in 1:n]
    gsc = FV.Grid3D(s, copy(U0); dx=d, dy=d, dz=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    gm  = Grid3DMtl(s, copy(U0); dx=d, dy=d, dz=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    for nn in 0:9; FV.step!(gsc, 0.1f0*d; rev=isodd(nn)); mstep3d!(gm, 0.1f0*d; rev=isodd(nn)); end
    Wc = FV.primitives(gsc); Wm = mprimitives_3d(gm)
    md = maximum(maximum(abs.(Wc[i,j,k] .- Wm[i,j,k])) for i in 1:n, j in 1:n, k in 1:n)
    println("Metal ≡ scalar Grid3D max|Δ| = ", md, md == 0f0 ? "  ✓ bit-identical" : "  (investigate)"); md
end

# ============================================================================================
# DUAL-ENERGY + f16-STORAGE Metal path (cold-gas cosmology, halved grid buffer) — M5-VALIDATION-READY.
#
# STATUS: written to MIRROR the CUDA de16/de16s logic (src/transpile_cuda.jl: k_ctus_de16 / k_ctus_de16s,
# ld_de16s_U / st_de16s_U, the Enzo dual-energy switch + PdV, GE_SCALE) AND the Metal split-sweep scheme
# above; the PHYSICS (dual-energy switch, GE_SCALE lift/unlift, f16 cold-gas survival) is validated on the
# CPU scalar reference (see test/cpu_de_validate.jl + metal_selfcheck_de16 below). The @metal kernels here
# CANNOT be GPU-compiled on the Linux/NVIDIA dev host — they are for the user to run on an M5 (Apple Silicon).
# NOT claimed validated on Metal hardware: run `metal_selfcheck_de16()` on the M5 to validate.
#
# THE PORT (faithful, NOT a CTU port): the CUDA backend uses a fused tiled CTU (k_ctus_de16); the Metal
# backend uses the Metal-native dimensionally-split Strang sweeps above. The split sweeps ALREADY advect the
# dual energy Ge (slot 6) and the colours through the generic FVSystem contract (physflux_x slot 6 = Ge*u),
# because EulerDE.cons2prim gives p = (gamma-1)*Ge so the Riemann fluxes use the dual-energy pressure (NEVER
# the cancelling E-1/2 rho v^2). What the split path lacks vs the CTU — and what these kernels ADD per cell
# AFTER the full Strang step — is the two NON-conservative pieces the CUDA kernel applies per substep:
#     (1) PdV work on Ge:   Ge -= dt*P*(div v)        (P = (gamma-1)*Ge, div v from central differences)
#     (2) the Enzo dual-energy switch:  if (E-KE)/E > eta -> eint = (E-KE)/rho, Ge <- rho*eint (warm, trust E)
#                                       else              -> keep evolved Ge, E <- KE + Ge      (cold, trust Ge)
# This is the SAME physics (cold-gas-safe, dual energy) with the split integrator — the faithful Metal
# equivalent of the CUDA de16 path, applied once per Strang step (the split analog of the CTU per-substep).
#
# f16 COMPUTE + f16 STORAGE (Grid3DMtlDE16, MtlArray{Float16}): mirrors k_ctus_de16s + ld_de16s_U/st_de16s_U.
# The global U/Unew buffers are MtlArray{Float16} (HALF the bytes of the f32 Grid3DMtl). The GE_SCALE
# convention (the SAME as the CUDA storage layout) — energies lifted, everything else raw:
#     slot 1   rho          stored RAW       (rho~0.17, normal in f16)
#     slots2-4 ru,rv,rw     stored RAW       (|ru|~2.5e-4 > f16 subnormal floor ~6e-5)
#     slot 5   E            stored E*GE_SCALE (cold E~2e-7 is f16-subnormal; lift to normal)
#     slot 6   Ge           stored Ge*GE_SCALE (cold Ge=rho*eint~1.2e-8 << subnormal floor; lift ESSENTIAL)
#     slots7.. rXi          stored RAW       (colour rho*X~0.05*rho, normal-valued; trace colours unsupported)
# Each step: ld (f16->f32, energies /GE_SCALE) -> f32 split sweeps -> PdV+switch -> st (f32->f16, energies
# *GE_SCALE). The sweep kernels run with the contract physics (element-type-generic); the energy slots only
# ever leave f16's subnormal danger zone via GE_SCALE. `read_conserved_de16` is the read-back analog of CUDA
# `read_conserved_f32` (un-lifts the energies).
#
# Build/validate on a Mac (M5):
#     using FiniteVolumeGodunovKA, Metal; include("metal/metal.jl")
#     metal_selfcheck_de16()            # cold IC, f16-storage dual energy: no NaN + eint preserved + halved mem
#     metal_selfcheck_de16_colors()     # + an advected colour sphere (EulerDEColors{1})
# ============================================================================================

import FiniteVolumeGodunovKA: EulerDE, EulerDEColors, LLF
const MTL_GE_SCALE = 1f7   # default GE_SCALE (mirrors the CUDA ge_scale=1e7 used in the f16-storage tests)

# COMPUTE-IN-f32, STORE-IN-f16 with the buffer ALWAYS GE_SCALE-LIFTED (the EXACT CUDA de16s convention).
# CRITICAL: the cold energies are un-lifted ONLY into f32 registers, NEVER back into the Float16 buffer —
# a cold Ge~1.2e-8 written to Float16 flushes to 0 (subnormal). So `ld_de16s_U`/`st_de16s_U` keep the f16
# global buffer permanently lifted (energies ×GE_SCALE, normal-valued in f16) and do the /GE_SCALE only into
# `float` registers inside each kernel. These read/write helpers mirror that: read f16(lifted) → f32(true),
# write f32(true) → f16(lifted). Energies = slots 5,6; ρ/momenta/colours are raw (already f16-normal).
@inline _ld_de16s(U, i, j, k, igs::Float32, ::Val{N}) where {N} =
    ntuple(c -> (c == 5 || c == 6) ? @inbounds(Float32(U[i, j, k, c])) * igs : @inbounds(Float32(U[i, j, k, c])), Val(N))
@inline function _st_de16s!(U, i, j, k, v::NTuple{N,Float32}, gs::Float32) where {N}
    ntuple(c -> (@inbounds(U[i, j, k, c] = Float16((c == 5 || c == 6) ? v[c] * gs : v[c])); nothing), Val(N))
    nothing
end
# velocity component from a lifted-buffer cell (slots 2,3,4 / ρ are raw, so no un-lift needed for u=ρu/ρ).
@inline _vel_de16s(U, i, j, k, comp) = Float32(@inbounds U[i, j, k, comp]) / Float32(@inbounds U[i, j, k, 1])

# --- the Enzo dual-energy switch + PdV (Float32), applied per cell AFTER the full Strang step. Mirrors the
#     tail of k_ctus_de16 (transpile_cuda.jl): PdV on Ge, then the eta switch syncing Ge<->E. --------------
# `divv_dx` = dx*(div v) from central differences; lam = dt/dx folds the 1/dx (0.5*lam central diff -> dt).
@inline function _mde_pdv_switch(s, Un::NTuple{NV,Float32}, divv_dx::Float32, lam::Float32) where {NV}
    rho = Un[1]; mx = Un[2]; my = Un[3]; mz = Un[4]; E0 = Un[5]; Ge0 = Un[6]
    P  = (s.γ - 1f0) * Ge0
    Ge1 = Ge0 - lam * P * divv_dx                                 # PdV: -dt*P*(div v)
    irho = inv(rho); ke = 0.5f0 * irho * (mx*mx + my*my + mz*mz)
    ratio = ifelse(E0 > 0f0, (E0 - ke) / E0, -1f0)
    Ged = E0 - ke
    Gew = ifelse(Ged > 0f0, Ged, Ge1)
    Ge2 = ifelse(ratio > s.η, Gew, Ge1)
    E2 = ke + Ge2                                                 # warm trusts E; cold trusts evolved Ge
    ntuple(c -> c == 5 ? E2 : c == 6 ? Ge2 : Un[c], Val(NV))
end

# Metal kernel: PdV + Enzo switch over the whole grid (compute f32, store f16, buffer stays lifted). Reads
# the lifted f16 buffer, un-lifts into f32, applies PdV+switch, re-lifts the (E,Ge) on store. bc via _gidx.
function _mde_switch3_kernel!(U, s, igs, gs, lam, nx, ny, nz, ::Val{N}, bc) where {N}
    i, j, k = _mtid3()
    if i <= nx && j <= ny && k <= nz
        u0 = _ld_de16s(U, i, j, k, igs, Val(N))
        uxp = _vel_de16s(U, _gidx(i+1,nx,bc), j, k, 2); uxm = _vel_de16s(U, _gidx(i-1,nx,bc), j, k, 2)
        vyp = _vel_de16s(U, i, _gidx(j+1,ny,bc), k, 3); vym = _vel_de16s(U, i, _gidx(j-1,ny,bc), k, 3)
        wzp = _vel_de16s(U, i, j, _gidx(k+1,nz,bc), 4); wzm = _vel_de16s(U, i, j, _gidx(k-1,nz,bc), 4)
        divv_dx = 0.5f0 * ((uxp - uxm) + (vyp - vym) + (wzp - wzm))
        _st_de16s!(U, i, j, k, _mde_pdv_switch(s, u0, divv_dx, lam), gs)
    end
    return
end

# f32-compute / f16-store dual-energy sweep kernels (one per axis), buffer stays lifted. Identical structure
# to _msweepx3/y3/z3 but reading f16(lifted)->f32(true), computing _update_dir in Float32, writing f16(lifted).
# The contract advects Ge (slot 6) and the colours (slots 7..NV) via physflux_x; p=(gamma-1)Ge from cons2prim.
function _mde16_sweepx3_kernel!(Unew, U, s, r, rs, lam, igs, gs, nx, ny, nz, ::Val{N}, bc, perm) where {N}
    i, j, k = _mtid3()
    if i <= nx && j <= ny && k <= nz
        _st_de16s!(Unew, i, j, k, _update_dir(s, r, rs,
            _ld_de16s(U,_gidx(i-2,nx,bc),j,k,igs,Val(N)), _ld_de16s(U,_gidx(i-1,nx,bc),j,k,igs,Val(N)), _ld_de16s(U,i,j,k,igs,Val(N)),
            _ld_de16s(U,_gidx(i+1,nx,bc),j,k,igs,Val(N)), _ld_de16s(U,_gidx(i+2,nx,bc),j,k,igs,Val(N)), lam, perm), gs)
    end
    return
end
function _mde16_sweepy3_kernel!(Unew, U, s, r, rs, lam, igs, gs, nx, ny, nz, ::Val{N}, bc, perm) where {N}
    i, j, k = _mtid3()
    if i <= nx && j <= ny && k <= nz
        _st_de16s!(Unew, i, j, k, _update_dir(s, r, rs,
            _ld_de16s(U,i,_gidx(j-2,ny,bc),k,igs,Val(N)), _ld_de16s(U,i,_gidx(j-1,ny,bc),k,igs,Val(N)), _ld_de16s(U,i,j,k,igs,Val(N)),
            _ld_de16s(U,i,_gidx(j+1,ny,bc),k,igs,Val(N)), _ld_de16s(U,i,_gidx(j+2,ny,bc),k,igs,Val(N)), lam, perm), gs)
    end
    return
end
function _mde16_sweepz3_kernel!(Unew, U, s, r, rs, lam, igs, gs, nx, ny, nz, ::Val{N}, bc, perm) where {N}
    i, j, k = _mtid3()
    if i <= nx && j <= ny && k <= nz
        _st_de16s!(Unew, i, j, k, _update_dir(s, r, rs,
            _ld_de16s(U,i,j,_gidx(k-2,nz,bc),igs,Val(N)), _ld_de16s(U,i,j,_gidx(k-1,nz,bc),igs,Val(N)), _ld_de16s(U,i,j,k,igs,Val(N)),
            _ld_de16s(U,i,j,_gidx(k+1,nz,bc),igs,Val(N)), _ld_de16s(U,i,j,_gidx(k+2,nz,bc),igs,Val(N)), lam, perm), gs)
    end
    return
end

# Wavespeed kernel for the dual-energy grid (compute f32; maxspeed_x via cons2prim -> p=(gamma-1)Ge; LLF-only).
# Reads the lifted f16 buffer, un-lifts into f32. Summed over the three axes like the CUDA k_speed_de16s.
function _mde_speed3_kernel!(spd, U, s, igs, nx, ny, nz, ::Val{N}, py, pz) where {N}
    i, j, k = _mtid3()
    if i <= nx && j <= ny && k <= nz
        W = cons2prim(s, _ld_de16s(U, i, j, k, igs, Val(N)))
        @inbounds spd[i, j, k] = maxspeed_x(s, W) + maxspeed_x(s, _swap(W, py)) + maxspeed_x(s, _swap(W, pz))
    end
    return
end

# --- f16-STORAGE dual-energy 3D grid: MtlArray{Float16} U/Unew (HALF the f32 bytes), GE_SCALE-lifted energies.
mutable struct Grid3DMtlDE16{N,T,S<:FVSystem,R,RS}
    sys::S; recon::R; rsol::RS
    U::MtlArray{T,4}; Unew::MtlArray{T,4}; spd::MtlArray{Float32,3}
    nx::Int; ny::Int; nz::Int; dx::Float32; dy::Float32; dz::Float32; bc::Symbol; cfl::Float32
    gs::Float32; store::Symbol; de_prec::Symbol   # GE_SCALE is Float32 (1e7 overflows Float16->Inf)
end

"""    Grid3DMtlDE16(sys, U0; dx,dy,dz, ge_scale=MTL_GE_SCALE, store=:f16, de_prec=:f16, ...)

f16-storage dual-energy Metal grid for `EulerDE` / `EulerDEColors{NC}`: the conserved buffer lives in
`MtlArray{Float16}` (HALF the bytes of the f32 `Grid3DMtl`), with the energy slots (5=E, 6=Ge) GE_SCALE-lifted
into f16's normal range on store and un-lifted on load (mirrors CUDA `ld_de16s_U`/`st_de16s_U`). Advance with
`mde16_step!`; read back with `read_conserved_de16`. RUN ON AN M5 — not GPU-compilable on the dev host."""
function Grid3DMtlDE16(sys::Union{EulerDE,EulerDEColors}, U0::Array{NTuple{N,TI},3};
                       dx, dy, dz, bc::Symbol = :periodic, recon = PLM(), rsol = LLF(),
                       cfl = 0.3f0, ge_scale::Real = MTL_GE_SCALE,
                       store::Symbol = :f16, de_prec::Symbol = :f16) where {N,TI}
    store === :f16 || error("Grid3DMtlDE16 is the f16-storage path (store=:f16).")
    de_prec === :f16 || error("Grid3DMtlDE16 is the all-f16 dual-energy path (de_prec=:f16).")
    T = Float16; gs = Float32(ge_scale); nx, ny, nz = size(U0)  # GE_SCALE stays f32 (1e7 > f16 max)
    Uh = Array{T,4}(undef, nx, ny, nz, N)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx, c in 1:N
        v = Float32(U0[i,j,k][c]); Uh[i,j,k,c] = T((c == 5 || c == 6) ? v * gs : v)   # lift energies (f32 GE_SCALE)
    end
    U = MtlArray(Uh)
    Grid3DMtlDE16{N,T,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, U, similar(U), Metal.zeros(Float32, nx, ny, nz),
        nx, ny, nz, Float32(dx), Float32(dy), Float32(dz), bc, Float32(cfl), gs, store, de_prec)
end

# one f16-storage dual-energy sweep along `axis` (fused; compute f32, store f16). Uses the de16 sweep kernels
# (_mde16_sweepx3/y3/z3) — the SAME contract physics as the f32 split sweeps, advecting Ge via slot 6 and the
# colours via slots 7..NV, but reading/writing the Float16 buffer (lam is Float32, the compute precision).
function _mde16_sweep!(g::Grid3DMtlDE16{N,T}, dt, axis::Val{A}, perm) where {N,T,A}
    thr, grp = _mcfg3(g.nx, g.ny, g.nz)
    lam = A == 1 ? Float32(dt)/g.dx : A == 2 ? Float32(dt)/g.dy : Float32(dt)/g.dz
    igs = Float32(inv(g.gs)); gs = Float32(g.gs)
    kern = A == 1 ? _mde16_sweepx3_kernel! : A == 2 ? _mde16_sweepy3_kernel! : _mde16_sweepz3_kernel!
    Metal.@metal threads=thr groups=grp kern(g.Unew, g.U, g.sys, g.recon, g.rsol,
        lam, igs, gs, g.nx, g.ny, g.nz, Val(N), Val(g.bc), perm)
    g.U, g.Unew = g.Unew, g.U
    return g
end

# one full f16-storage dual-energy Strang step. The buffer stays GE_SCALE-LIFTED at ALL times (cold energies
# would flush to 0 if ever written un-lifted to Float16); each kernel un-lifts only into f32 registers (the
# CUDA ld_de16s_U/st_de16s_U convention) — so NO standalone scale passes. The split sweeps advect Ge + the
# colours; the per-cell PdV + Enzo switch then syncs Ge<->E. lam uses dx for the divv central diff (cubic dx).
function mde16_step!(g::Grid3DMtlDE16{N,T}, dt; rev::Bool = false) where {N,T}
    px = identperm(Val(N)); py = dirperm(g.sys, N, 2); pz = dirperm(g.sys, N, 3)
    thr, grp = _mcfg3(g.nx, g.ny, g.nz); igs = Float32(inv(g.gs)); gs = Float32(g.gs)
    if rev
        _mde16_sweep!(g, dt, Val(3), pz); _mde16_sweep!(g, dt, Val(2), py); _mde16_sweep!(g, dt, Val(1), px)
    else
        _mde16_sweep!(g, dt, Val(1), px); _mde16_sweep!(g, dt, Val(2), py); _mde16_sweep!(g, dt, Val(3), pz)
    end
    lam = Float32(dt) / g.dx
    Metal.@metal threads=thr groups=grp _mde_switch3_kernel!(g.U, g.sys, igs, gs, lam, g.nx, g.ny, g.nz, Val(N), Val(g.bc))
    return g
end

function mde16_max_wavespeed(g::Grid3DMtlDE16{N,T}) where {N,T}
    thr, grp = _mcfg3(g.nx, g.ny, g.nz); igs = Float32(inv(g.gs))
    Metal.@metal threads=thr groups=grp _mde_speed3_kernel!(g.spd, g.U, g.sys, igs, g.nx, g.ny, g.nz, Val(N),
        dirperm(g.sys, N, 2), dirperm(g.sys, N, 3))
    return maximum(g.spd)
end

function mde16_evolve!(g::Grid3DMtlDE16, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        c = mde16_max_wavespeed(g); g.sys = FV.prestep(g.sys, c)
        dt = min(g.cfl * min(g.dx, g.dy, g.dz) / c, tend - t)
        mde16_step!(g, dt; rev = isodd(n)); t += dt; n += 1
    end
    return g
end

"""    read_conserved_de16(g) -> Array{NTuple{N,Float32},3}

Read the f16-storage dual-energy grid back to host f32 conserved tuples, un-lifting the GE_SCALE-lifted energy
slots (5=E, 6=Ge). The read-back analog of CUDA `read_conserved_f32` for the f16-storage de16 path."""
function read_conserved_de16(g::Grid3DMtlDE16{N}) where {N}
    Uh = Array(g.U); igs = Float32(inv(g.gs))
    [ntuple(c -> (c == 5 || c == 6) ? Float32(Uh[i,j,k,c]) * igs : Float32(Uh[i,j,k,c]), Val(N))
     for i in 1:g.nx, j in 1:g.ny, k in 1:g.nz]
end

mde16_primitives(g::Grid3DMtlDE16{N}) where {N} =
    (U = read_conserved_de16(g); [cons2prim(g.sys, U[i,j,k]) for i in 1:g.nx, j in 1:g.ny, k in 1:g.nz])

# =============================== M5 validation scripts (run on Apple Silicon) ===============================
# Mirror the CUDA `EulerDE`/`EulerDEColors` f16-storage tests in test/runtests.jl: build the f16-storage
# dual-energy grid on the COLD cosmology IC, step it, assert no NaN, rho>0, eint preserved (~6.96e-8, NOT
# flushed to subnormal 0), colour advects+bounded, and the buffer is HALF the f32 size.

function metal_selfcheck_de16(; nd = (32, 16, 16), nsteps = 20)
    γd = 5f0/3f0; dxd = Float32(2π/nd[1]); ρ0 = 0.17f0; e0 = 6.96f-8; v0 = 4f-4
    mkd(i,j,k) = begin
        x = 2f0π*(i-1)/nd[1]; y = 2f0π*(j-1)/nd[2]; z = 2f0π*(k-1)/nd[3]
        ρ = ρ0*(1f0+0.03f0*sin(x)); u = v0*(1f0+0.1f0*sin(y)); v = v0*0.1f0*cos(z); w = v0*0.1f0*sin(x+y)
        eint = e0*(1f0+0.02f0*cos(z)); Ge = ρ*eint; E = Ge + 0.5f0*ρ*(u*u+v*v+w*w)
        (ρ, ρ*u, ρ*v, ρ*w, E, Ge)
    end
    Ud = [mkd(i,j,k) for i in 1:nd[1], j in 1:nd[2], k in 1:nd[3]]
    sde = EulerDE(γ = γd, η = 1f-3)
    g = Grid3DMtlDE16(sde, copy(Ud); dx = dxd, dy = dxd, dz = dxd, ge_scale = MTL_GE_SCALE)
    bytes16 = sizeof(eltype(g.U)) * length(g.U); bytes32 = sizeof(Float32) * length(g.U)
    c = mde16_max_wavespeed(g); dtd = g.cfl * dxd / c
    for n in 0:(nsteps-1); mde16_step!(g, dtd; rev = isodd(n)); end
    Metal.synchronize()
    W = mde16_primitives(g)
    fin = all(w -> all(isfinite, w), W); ρmin = minimum(w[1] for w in W)
    e16 = [w[6] for w in W]; emin, emax = extrema(e16)
    ok = fin && ρmin > 0 && all(0.5f0*e0 < e < 1.5f0*e0 for e in e16) && emin > 1f-9 && bytes16 == bytes32 ÷ 2
    println("Metal f16-storage dual-energy (cold gas):")
    println("  finite=", fin, "  rhomin=", ρmin, "  eint in [", emin, ", ", emax, "]  target e0=", e0)
    println("  buffer bytes f16=", bytes16, " vs f32=", bytes32, " (HALVED=", bytes16 == bytes32 ÷ 2, ")")
    println(ok ? "  OK: no NaN, eint preserved, memory halved" : "  (investigate)")
    return ok
end

function metal_selfcheck_de16_colors(; nd = (32, 32, 32), nsteps = 20)
    γd = 5f0/3f0; dxd = Float32(2π/nd[1]); ρ0 = 0.17f0; e0 = 6.96f-8; v0 = 4f-4
    sph(i,j,k) = ((i-nd[1]/2f0)^2 + (j-nd[2]/2f0)^2 + (k-nd[3]/2f0)^2) < (nd[1]/5f0)^2
    mkc(i,j,k) = begin
        x = 2f0π*(i-1)/nd[1]; y = 2f0π*(j-1)/nd[2]; z = 2f0π*(k-1)/nd[3]
        ρ = ρ0*(1f0+0.03f0*sin(x)); u = v0*(1f0+0.1f0*sin(y)); v = v0*0.1f0*cos(z); w = v0*0.1f0*sin(x+y)
        eint = e0*(1f0+0.02f0*cos(z)); Ge = ρ*eint; E = Ge + 0.5f0*ρ*(u*u+v*v+w*w)
        (ρ, ρ*u, ρ*v, ρ*w, E, Ge, ρ*(sph(i,j,k) ? 0.09f0 : 0.05f0))
    end
    Uc = [mkc(i,j,k) for i in 1:nd[1], j in 1:nd[2], k in 1:nd[3]]
    sdc = EulerDEColors{1}(γ = γd, η = 1f-3)
    X0 = [Uc[i,j,k][7]/Uc[i,j,k][1] for i in 1:nd[1], j in 1:nd[2], k in 1:nd[3]]
    g = Grid3DMtlDE16(sdc, copy(Uc); dx = dxd, dy = dxd, dz = dxd, ge_scale = MTL_GE_SCALE)
    c = mde16_max_wavespeed(g); dtd = g.cfl * dxd / c
    for n in 0:(nsteps-1); mde16_step!(g, dtd; rev = isodd(n)); end
    Metal.synchronize()
    W = mde16_primitives(g)
    fin = all(w -> all(isfinite, w), W); e16 = [w[6] for w in W]; X16 = [w[7] for w in W]
    advected = maximum(abs.(X16 .- X0)) > 1f-2; bounded = all(0f0 <= x <= 1f0 for x in X16)
    ok = fin && all(0.5f0*e0 < e < 1.5f0*e0 for e in e16) && advected && bounded
    println("Metal f16-storage dual-energy + colour:")
    println("  finite=", fin, "  eint in ", extrema(e16), "  X in ", extrema(X16), "  max|dX|=", maximum(abs.(X16 .- X0)))
    println(ok ? "  OK: no NaN, eint preserved, colour advects + bounded" : "  (investigate)")
    return ok
end

function metal_selfcheck_3d_colors(; n = 16)
    s = FV.Euler(γ = 1.4f0); d = 1f0 / n
    U0 = [FV.prim2cons(s, (1f0 + 0.1f0 * sinpi(2f0 * Float32(i + j + k) / n),
                            0.3f0, 0.2f0, 0.1f0, 1f0)) for i in 1:n, j in 1:n, k in 1:n]
    C = Array{Float32,4}(undef, n, n, n, 2)
    C[:, :, :, 1] .= 0.4f0
    C[:, :, :, 2] .= 1f-18
    gsc = FV.Grid3D(s, copy(U0); dx=d, dy=d, dz=d, bc=:periodic, recon=PLM(), rsol=HLLC(), colors=C)
    gm = Grid3DMtl(s, copy(U0); dx=d, dy=d, dz=d, bc=:periodic, recon=PLM(), rsol=HLLC(), colors=C)
    for step in 1:3
        FV.step!(gsc, 0.03f0 * d; rev=isodd(step))
        mstep3d!(gm, 0.03f0 * d; rev=isodd(step))
    end
    Metal.synchronize()
    Xc = FV.color_fractions(gsc)
    Xm = mcolor_fractions_3d(gm)
    md = maximum(abs, Xc .- Xm)
    pd = maximum(abs, Int.(gsc.colors) .- Int.(Array(gm.colors)))
    println("Metal ≡ scalar Grid3D packed colors max|ΔX| = ", md,
            " packed Δ = ", pd, (md == 0f0 && pd == 0) ? "  ✓ bit-identical" : "  (investigate)")
    return md
end
