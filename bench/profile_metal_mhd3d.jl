using Printf
using Statistics

include(joinpath(@__DIR__, "..", "metal", "metal.jl"))

const SAMPLES = parse(Int, get(ENV, "METAL_MHD_SAMPLES", "5"))

function mhd_u0(n)
    s = GLMMHD(γ = 5f0 / 3f0, ch = 2f0)
    B0 = 1f0 / sqrt(4f0 * Float32(π))
    icW(x, y, z) = (0.5f0, -sinpi(2f0 * y), sinpi(2f0 * x), 0.1f0,
                    0.5f0, -B0 * sinpi(2f0 * y), B0 * sinpi(4f0 * x), 0f0, 0f0)
    U0 = [FV.prim2cons(s, icW((i - 0.5f0) / n, (j - 0.5f0) / n, (k - 0.5f0) / n))
          for i in 1:n, j in 1:n, k in 1:n]
    return s, U0
end

function mhd_grid3(n)
    s, U0 = mhd_u0(n)
    d = 1f0 / n
    g = Grid3DMtl(s, U0; dx = d, dy = d, dz = d, bc = :periodic, recon = PLM(), rsol = HLLD())
    return g, d
end

function mhd_scalar_grid3(n)
    s, U0 = mhd_u0(n)
    d = 1f0 / n
    g = Grid3D(s, U0; dx = d, dy = d, dz = d, bc = :periodic, recon = PLM(), rsol = HLLD())
    return g, d
end

function mfused_step3d!(g::Grid3DMtl{N}, dt; rev::Bool = false) where {N}
    thr, grp = _mcfg3(g.nx, g.ny, g.nz)
    bc = Val(g.bc)
    px = identperm(Val(N))
    py = dirperm(g.sys, N, 2)
    pz = dirperm(g.sys, N, 3)
    sweep(kern, λ, perm) = (Metal.@metal threads=thr groups=grp kern(g.Unew, g.U, g.sys, g.recon, g.rsol,
        λ, g.nx, g.ny, g.nz, Val(N), bc, perm); (g.U, g.Unew) = (g.Unew, g.U))
    if rev
        sweep(_msweepz3_kernel!, Float32(dt) / g.dz, pz)
        sweep(_msweepy3_kernel!, Float32(dt) / g.dy, py)
        sweep(_msweepx3_kernel!, Float32(dt) / g.dx, px)
    else
        sweep(_msweepx3_kernel!, Float32(dt) / g.dx, px)
        sweep(_msweepy3_kernel!, Float32(dt) / g.dy, py)
        sweep(_msweepz3_kernel!, Float32(dt) / g.dz, pz)
    end
    has_source(g.sys) && Metal.@metal threads=thr groups=grp _msource3_kernel!(g.U, g.sys, Float32(dt), g.nx, g.ny, g.nz, Val(N))
    return g
end

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

function max_prim_diff_scalar_metal(gsc::Grid3D{N}, gm::Grid3DMtl{N}) where {N}
    Wc = FV.primitives(gsc)
    Wm = mprimitives_3d(gm)
    return maximum(maximum(abs.(Wc[i, j, k] .- Wm[i, j, k])) for i in 1:gsc.nx, j in 1:gsc.ny, k in 1:gsc.nz)
end

function max_u_diff(g1::Grid3DMtl, g2::Grid3DMtl)
    A = Array(g1.U)
    B = Array(g2.U)
    return maximum(abs, A .- B)
end

function print_rate(label, cells, steps, seconds)
    rate = cells * steps / seconds / 1e6
    @printf("%-34s %9.3f ms  %9.1f Mcell/s\n", label, 1e3 * seconds / steps, rate)
    return rate
end

function validate_mhd()
    println("\n== 3D GLM-MHD/HLLD Metal parity ==")
    for n in (12, 16, 24)
        gsc, d = mhd_scalar_grid3(n)
        gm, _ = mhd_grid3(n)
        dt = 0.05f0 * d
        steps = n == 24 ? 4 : 6
        for step in 0:(steps - 1)
            FV.step!(gsc, dt; rev = isodd(step))
            mstep3d!(gm, dt; rev = isodd(step))
        end
        Metal.synchronize()
        @printf("scalar vs current n=%-3d steps=%-2d max|ΔW| = %.9g\n", n, steps, max_prim_diff_scalar_metal(gsc, gm))
    end

    for n in (15, 16, 31, 32)
        gp, d = mhd_grid3(n)
        gf, _ = mhd_grid3(n)
        dt = 0.05f0 * d
        for step in 0:5
            mstep3d!(gp, dt; rev = isodd(step))
            mfused_step3d!(gf, dt; rev = isodd(step))
        end
        Metal.synchronize()
        @printf("current vs fused n=%-3d max|ΔU| = %.9g\n", n, max_u_diff(gp, gf))
    end
end

function benchmark_mhd()
    println("\n== 3D GLM-MHD/HLLD Metal throughput ==")
    for n in (32, 64, 96, 128)
        gp, d = mhd_grid3(n)
        gf, _ = mhd_grid3(n)
        dt = 0.03f0 * d
        steps = n >= 128 ? 5 : n >= 96 ? 8 : n >= 64 ? 12 : 30
        cells = n^3
        tp = median_seconds(() -> (for step in 1:steps
            mstep3d!(gp, dt; rev = isodd(step))
        end))
        tf = median_seconds(() -> (for step in 1:steps
            mfused_step3d!(gf, dt; rev = isodd(step))
        end))
        print_rate("current routed n=$n", cells, steps, tp)
        print_rate("fused legacy n=$n", cells, steps, tf)
        @printf("%-34s %8.3f%%\n", "current delta", 100 * (tf / tp - 1))
    end
end

println("Metal device: ", Metal.device())
validate_mhd()
benchmark_mhd()
