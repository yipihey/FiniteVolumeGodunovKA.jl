using Printf
using Statistics

include(joinpath(@__DIR__, "..", "metal", "metal.jl"))

const SAMPLES = parse(Int, get(ENV, "METAL_HALO_SAMPLES", "7"))
const SIZES = Tuple(parse.(Int, split(get(ENV, "METAL_HALO_SIZES", "64,96,128"), ",")))

mutable struct Grid3DMtlHalo{N,S<:FVSystem,R,RS}
    sys::S; recon::R; rsol::RS
    U::MtlArray{Float32,4}; Unew::MtlArray{Float32,4}
    nx::Int; ny::Int; nz::Int; dx::Float32; dy::Float32; dz::Float32
end

function Grid3DMtlHalo(sys::FVSystem, U0::Array{NTuple{N,T},3};
                       dx, dy, dz, bc::Symbol = :periodic, recon = PLM(), rsol = HLLC()) where {N,T}
    bc == :periodic || error("halo prototype only implements periodic boundaries")
    nx, ny, nz = size(U0)
    Uh = zeros(Float32, nx + 4, ny + 4, nz + 4, N)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx, c in 1:N
        Uh[i + 2, j + 2, k + 2, c] = Float32(U0[i, j, k][c])
    end
    U = MtlArray(Uh)
    Grid3DMtlHalo{N,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, U, similar(U), nx, ny, nz, Float32(dx), Float32(dy), Float32(dz))
end

@inline _hidx(p, n) = p < 3 ? p + n : p > n + 2 ? p - n : p

function _mfill_periodic_halo3_kernel!(U, nx, ny, nz, ::Val{N}) where {N}
    i, j, k = _mtid3()
    if i <= nx + 4 && j <= ny + 4 && k <= nz + 4
        if i < 3 || i > nx + 2 || j < 3 || j > ny + 2 || k < 3 || k > nz + 2
            ii = _hidx(i, nx)
            jj = _hidx(j, ny)
            kk = _hidx(k, nz)
            sx = nx + 4
            sy = ny + 4
            sz = nz + 4
            stride = sx * sy * sz
            dst = i + (j - 1) * sx + (k - 1) * sx * sy
            src = ii + (jj - 1) * sx + (kk - 1) * sx * sy
            @inbounds U[dst] = U[src]
            @inbounds U[dst + stride] = U[src + stride]
            @inbounds U[dst + 2 * stride] = U[src + 2 * stride]
            @inbounds U[dst + 3 * stride] = U[src + 3 * stride]
            @inbounds U[dst + 4 * stride] = U[src + 4 * stride]
            if N == 9
                @inbounds U[dst + 5 * stride] = U[src + 5 * stride]
                @inbounds U[dst + 6 * stride] = U[src + 6 * stride]
                @inbounds U[dst + 7 * stride] = U[src + 7 * stride]
                @inbounds U[dst + 8 * stride] = U[src + 8 * stride]
            end
        end
    end
    return
end

function _mfill_periodic_halo3!(g::Grid3DMtlHalo{N}) where {N}
    thr, grp = _mcfg3(g.nx + 4, g.ny + 4, g.nz + 4)
    Metal.@metal threads=thr groups=grp _mfill_periodic_halo3_kernel!(g.U, g.nx, g.ny, g.nz, Val(N))
    return g
end

@inline _hread_axis(U, i, j, k, o, ::Val{1}, ::Val{N}) where {N} =
    _mread3(U, i + 2 + o, j + 2, k + 2, Val(N))
@inline _hread_axis(U, i, j, k, o, ::Val{2}, ::Val{N}) where {N} =
    _mread3(U, i + 2, j + 2 + o, k + 2, Val(N))
@inline _hread_axis(U, i, j, k, o, ::Val{3}, ::Val{N}) where {N} =
    _mread3(U, i + 2, j + 2, k + 2 + o, Val(N))

@inline _hwrite_axis!(U, i, j, k, o, v, ::Val{1}) = _mwrite3!(U, i + 2 + o, j + 2, k + 2, v)
@inline _hwrite_axis!(U, i, j, k, o, v, ::Val{2}) = _mwrite3!(U, i + 2, j + 2 + o, k + 2, v)
@inline _hwrite_axis!(U, i, j, k, o, v, ::Val{3}) = _mwrite3!(U, i + 2, j + 2, k + 2 + o, v)

@inline function _hhalf_axis(U, s, r, i, j, k, o, λ, axis, perm, ::Val{N}) where {N}
    um = _hread_axis(U, i, j, k, o - 1, axis, Val(N))
    u0 = _hread_axis(U, i, j, k, o,     axis, Val(N))
    up = _hread_axis(U, i, j, k, o + 1, axis, Val(N))
    return _halfstep(s, r, _swap(um, perm), _swap(u0, perm), _swap(up, perm), λ)
end

function _mstep3_pair_halo_kernel!(Unew, U, s, r, rs, λ, nx, ny, nz, ::Val{N}, axis::Val{A}, perm) where {N,A}
    ti, tj, tk = _mtid3()
    i, j, k = _maxis_coord(ti, tj, tk, axis)
    if _maxis_valid(i, j, k, nx, ny, nz, axis)
        wlm, wrm = _hhalf_axis(U, s, r, i, j, k, -1, λ, axis, perm, Val(N))
        wl0, wr0 = _hhalf_axis(U, s, r, i, j, k,  0, λ, axis, perm, Val(N))
        wl1, wr1 = _hhalf_axis(U, s, r, i, j, k,  1, λ, axis, perm, Val(N))
        f0 = riemann(rs, s, wrm, wl0)
        f1 = riemann(rs, s, wr0, wl1)
        u0 = _hread_axis(U, i, j, k, 0, axis, Val(N))
        _hwrite_axis!(Unew, i, j, k, 0, u0 .- λ .* _swap(f1 .- f0, perm), axis)
        if A == 1
            ok2 = i + 1 <= nx
        elseif A == 2
            ok2 = j + 1 <= ny
        else
            ok2 = k + 1 <= nz
        end
        if ok2
            wl2, wr2 = _hhalf_axis(U, s, r, i, j, k, 2, λ, axis, perm, Val(N))
            f2 = riemann(rs, s, wr1, wl2)
            u1 = _hread_axis(U, i, j, k, 1, axis, Val(N))
            _hwrite_axis!(Unew, i, j, k, 1, u1 .- λ .* _swap(f2 .- f1, perm), axis)
        end
    end
    return
end

function _mpair_sweep3_halo!(g::Grid3DMtlHalo{N}, dt, axis::Val{A}, perm) where {N,A}
    _mfill_periodic_halo3!(g)
    thr, grp = _mpaircfg3(g.nx, g.ny, g.nz, axis)
    λ = A == 1 ? Float32(dt) / g.dx : A == 2 ? Float32(dt) / g.dy : Float32(dt) / g.dz
    Metal.@metal threads=thr groups=grp _mstep3_pair_halo_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        λ, g.nx, g.ny, g.nz, Val(N), axis, perm)
    g.U, g.Unew = g.Unew, g.U
    return g
end

function _msource3_halo_kernel!(U, s, dt, nx, ny, nz, ::Val{N}) where {N}
    i, j, k = _mtid3()
    if i <= nx && j <= ny && k <= nz
        ii = i + 2
        jj = j + 2
        kk = k + 2
        _mwrite3!(U, ii, jj, kk, source(s, _mread3(U, ii, jj, kk, Val(N)), dt))
    end
    return
end

function _msource3_halo!(g::Grid3DMtlHalo{N}, dt) where {N}
    thr, grp = _mcfg3(g.nx, g.ny, g.nz)
    Metal.@metal threads=thr groups=grp _msource3_halo_kernel!(g.U, g.sys, Float32(dt), g.nx, g.ny, g.nz, Val(N))
    return g
end

function mstep3d_halo!(g::Grid3DMtlHalo{N}, dt; rev::Bool = false) where {N}
    px = identperm(Val(N)); py = dirperm(g.sys, N, 2); pz = dirperm(g.sys, N, 3)
    if rev
        _mpair_sweep3_halo!(g, dt, Val(3), pz)
        _mpair_sweep3_halo!(g, dt, Val(2), py)
        _mpair_sweep3_halo!(g, dt, Val(1), px)
    else
        _mpair_sweep3_halo!(g, dt, Val(1), px)
        _mpair_sweep3_halo!(g, dt, Val(2), py)
        _mpair_sweep3_halo!(g, dt, Val(3), pz)
    end
    has_source(g.sys) && _msource3_halo!(g, dt)
    return g
end

function euler_u0(n)
    s = FV.Euler(γ = 1.4f0)
    U0 = [FV.prim2cons(s, (1f0 + 0.2f0 * sinpi(2f0 * Float32(i + j + k) / n),
                            0.5f0, 0.3f0, 0.2f0, 1f0)) for i in 1:n, j in 1:n, k in 1:n]
    return s, U0
end

function mhd_u0(n)
    s = GLMMHD(γ = 5f0 / 3f0, ch = 2f0)
    B0 = 1f0 / sqrt(4f0 * Float32(π))
    icW(x, y, z) = (0.5f0, -sinpi(2f0 * y), sinpi(2f0 * x), 0.1f0,
                    0.5f0, -B0 * sinpi(2f0 * y), B0 * sinpi(4f0 * x), 0f0, 0f0)
    U0 = [FV.prim2cons(s, icW((i - 0.5f0) / n, (j - 0.5f0) / n, (k - 0.5f0) / n))
          for i in 1:n, j in 1:n, k in 1:n]
    return s, U0
end

function grids(kind, n)
    s, U0 = kind == :euler ? euler_u0(n) : mhd_u0(n)
    d = 1f0 / n
    rsol = kind == :euler ? HLLC() : HLLD()
    recon = PLM()
    g0 = Grid3DMtl(s, U0; dx = d, dy = d, dz = d, bc = :periodic, recon, rsol)
    gh = Grid3DMtlHalo(s, U0; dx = d, dy = d, dz = d, bc = :periodic, recon, rsol)
    return g0, gh, d
end

function halo_interior_array(g::Grid3DMtlHalo)
    U = Array(g.U)
    return copy(@view U[3:(g.nx + 2), 3:(g.ny + 2), 3:(g.nz + 2), :])
end

max_u_diff(g0::Grid3DMtl, gh::Grid3DMtlHalo) = maximum(abs, Array(g0.U) .- halo_interior_array(gh))

function median_seconds(f; samples = SAMPLES)
    f()
    Metal.synchronize()
    times = Float64[]
    for _ in 1:samples
        GC.gc(false)
        t0 = time_ns()
        f()
        Metal.synchronize()
        push!(times, (time_ns() - t0) / 1e9)
    end
    return median(times)
end

function print_rate(label, cells, steps, seconds)
    rate = cells * steps / seconds / 1e6
    @printf("%-28s %9.3f ms  %9.1f Mcell/s\n", label, 1e3 * seconds / steps, rate)
    return rate
end

function validate()
    println("\n== halo parity vs current ==")
    for kind in (:euler, :mhd), n in (15, 16, 24)
        g0, gh, d = grids(kind, n)
        dt = 0.04f0 * d
        steps = n == 24 ? 4 : 6
        for step in 1:steps
            mstep3d!(g0, dt; rev = isodd(step))
            mstep3d_halo!(gh, dt; rev = isodd(step))
        end
        Metal.synchronize()
        @printf("%-5s n=%-3d steps=%-2d max|ΔU| = %.9g\n", string(kind), n, steps, max_u_diff(g0, gh))
    end
end

function benchmark_kind(kind)
    println("\n== ", kind, " throughput: current vs halo ==")
    for n in SIZES
        g0, gh, d = grids(kind, n)
        dt = 0.03f0 * d
        steps = n >= 128 ? 6 : n >= 96 ? 10 : 18
        cells = n^3
        t0 = median_seconds(() -> (for step in 1:steps
            mstep3d!(g0, dt; rev = isodd(step))
        end))
        th = median_seconds(() -> (for step in 1:steps
            mstep3d_halo!(gh, dt; rev = isodd(step))
        end))
        print_rate("current $(kind) n=$n", cells, steps, t0)
        print_rate("halo $(kind) n=$n", cells, steps, th)
        @printf("%-28s %8.3f%%\n", "halo delta", 100 * (t0 / th - 1))
    end
end

println("Metal device: ", Metal.device())
println("samples: ", SAMPLES, "  sizes: ", SIZES)
validate()
benchmark_kind(:euler)
benchmark_kind(:mhd)
