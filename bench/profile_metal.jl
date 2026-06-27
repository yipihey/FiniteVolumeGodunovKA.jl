using Printf
using Statistics

include(joinpath(@__DIR__, "..", "metal", "metal.jl"))

const MODE = get(ARGS, 1, "quick")
const SAMPLES = MODE == "full" ? 7 : 5

groups2(nx, ny, threads) = (cld(nx, threads[1]), cld(ny, threads[2]))
groups3(nx, ny, nz, threads) = (cld(nx, threads[1]), cld(ny, threads[2]), cld(nz, threads[3]))

function euler_grid1(n)
    s = FV.Euler(γ = 1.4f0)
    d = 1f0 / n
    U0 = [FV.prim2cons(s, (1f0 + 0.2f0 * sinpi(2f0 * Float32(i) / n),
                            0.5f0, 0.3f0, 0.2f0, 1f0)) for i in 1:n]
    return Grid1DMtl(s, U0; dx = d, bc = :periodic, recon = PLM(), rsol = HLLC()), d
end

function euler_grid2(n)
    s = FV.Euler(γ = 1.4f0)
    d = 1f0 / n
    U0 = [FV.prim2cons(s, (1f0 + 0.2f0 * sinpi(2f0 * Float32(i + j) / n),
                            0.5f0, 0.3f0, 0.2f0, 1f0)) for i in 1:n, j in 1:n]
    return Grid2DMtl(s, U0; dx = d, dy = d, bc = :periodic, recon = PLM(), rsol = HLLC()), d
end

function euler_grid3(n)
    s = FV.Euler(γ = 1.4f0)
    d = 1f0 / n
    U0 = [FV.prim2cons(s, (1f0 + 0.2f0 * sinpi(2f0 * Float32(i + j + k) / n),
                            0.5f0, 0.3f0, 0.2f0, 1f0)) for i in 1:n, j in 1:n, k in 1:n]
    return Grid3DMtl(s, U0; dx = d, dy = d, dz = d, bc = :periodic, recon = PLM(), rsol = HLLC()), d
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

function print_rate(label, cells, steps, seconds)
    rate = cells * steps / seconds / 1e6
    @printf("%-36s %9.3f ms  %9.1f Mcell/s\n", label, 1e3 * seconds / steps, rate)
    return rate
end

function launch_x2!(g::Grid2DMtl{N}, dt, threads) where {N}
    bc = Val(g.bc)
    px = identperm(Val(N))
    grp = groups2(g.nx, g.ny, threads)
    Metal.@metal threads=threads groups=grp _msweepx2_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt) / g.dx, g.nx, g.ny, Val(N), bc, px)
    g.U, g.Unew = g.Unew, g.U
    return g
end

function launch_y2!(g::Grid2DMtl{N}, dt, threads) where {N}
    bc = Val(g.bc)
    py = dirperm(g.sys, N, 2)
    grp = groups2(g.nx, g.ny, threads)
    Metal.@metal threads=threads groups=grp _msweepy2_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt) / g.dy, g.nx, g.ny, Val(N), bc, py)
    g.U, g.Unew = g.Unew, g.U
    return g
end

function step2_shape!(g, dt, tx, ty; rev = false)
    rev ? (launch_y2!(g, dt, ty); launch_x2!(g, dt, tx)) :
          (launch_x2!(g, dt, tx); launch_y2!(g, dt, ty))
    return g
end

function launch_x3!(g::Grid3DMtl{N}, dt, threads) where {N}
    bc = Val(g.bc)
    px = identperm(Val(N))
    grp = groups3(g.nx, g.ny, g.nz, threads)
    Metal.@metal threads=threads groups=grp _msweepx3_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt) / g.dx, g.nx, g.ny, g.nz, Val(N), bc, px)
    g.U, g.Unew = g.Unew, g.U
    return g
end

function launch_y3!(g::Grid3DMtl{N}, dt, threads) where {N}
    bc = Val(g.bc)
    py = dirperm(g.sys, N, 2)
    grp = groups3(g.nx, g.ny, g.nz, threads)
    Metal.@metal threads=threads groups=grp _msweepy3_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt) / g.dy, g.nx, g.ny, g.nz, Val(N), bc, py)
    g.U, g.Unew = g.Unew, g.U
    return g
end

function launch_z3!(g::Grid3DMtl{N}, dt, threads) where {N}
    bc = Val(g.bc)
    pz = dirperm(g.sys, N, 3)
    grp = groups3(g.nx, g.ny, g.nz, threads)
    Metal.@metal threads=threads groups=grp _msweepz3_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt) / g.dz, g.nx, g.ny, g.nz, Val(N), bc, pz)
    g.U, g.Unew = g.Unew, g.U
    return g
end

function step3_shape!(g, dt, tx, ty, tz; rev = false)
    rev ? (launch_z3!(g, dt, tz); launch_y3!(g, dt, ty); launch_x3!(g, dt, tx)) :
          (launch_x3!(g, dt, tx); launch_y3!(g, dt, ty); launch_z3!(g, dt, tz))
    return g
end

function cached2(g::Grid2DMtl{N}, dt, threads) where {N}
    bc = Val(g.bc)
    px = identperm(Val(N))
    py = dirperm(g.sys, N, 2)
    kx = Metal.@metal launch=false _msweepx2_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt) / g.dx, g.nx, g.ny, Val(N), bc, px)
    ky = Metal.@metal launch=false _msweepy2_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt) / g.dy, g.nx, g.ny, Val(N), bc, py)
    return (; kx, ky, threads, groups = groups2(g.nx, g.ny, threads), bc, px, py)
end

function step2_cached!(g::Grid2DMtl{N}, dt, c; rev = false) where {N}
    swx() = (c.kx(g.Unew, g.U, g.sys, g.recon, g.rsol, Float32(dt) / g.dx, g.nx, g.ny, Val(N), c.bc, c.px;
                  threads = c.threads, groups = c.groups); (g.U, g.Unew) = (g.Unew, g.U))
    swy() = (c.ky(g.Unew, g.U, g.sys, g.recon, g.rsol, Float32(dt) / g.dy, g.nx, g.ny, Val(N), c.bc, c.py;
                  threads = c.threads, groups = c.groups); (g.U, g.Unew) = (g.Unew, g.U))
    rev ? (swy(); swx()) : (swx(); swy())
    return g
end

function cached3(g::Grid3DMtl{N}, dt, threads) where {N}
    bc = Val(g.bc)
    px = identperm(Val(N))
    py = dirperm(g.sys, N, 2)
    pz = dirperm(g.sys, N, 3)
    kx = Metal.@metal launch=false _msweepx3_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt) / g.dx, g.nx, g.ny, g.nz, Val(N), bc, px)
    ky = Metal.@metal launch=false _msweepy3_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt) / g.dy, g.nx, g.ny, g.nz, Val(N), bc, py)
    kz = Metal.@metal launch=false _msweepz3_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt) / g.dz, g.nx, g.ny, g.nz, Val(N), bc, pz)
    return (; kx, ky, kz, threads, groups = groups3(g.nx, g.ny, g.nz, threads), bc, px, py, pz)
end

function step3_cached!(g::Grid3DMtl{N}, dt, c; rev = false) where {N}
    swx() = (c.kx(g.Unew, g.U, g.sys, g.recon, g.rsol, Float32(dt) / g.dx, g.nx, g.ny, g.nz, Val(N), c.bc, c.px;
                  threads = c.threads, groups = c.groups); (g.U, g.Unew) = (g.Unew, g.U))
    swy() = (c.ky(g.Unew, g.U, g.sys, g.recon, g.rsol, Float32(dt) / g.dy, g.nx, g.ny, g.nz, Val(N), c.bc, c.py;
                  threads = c.threads, groups = c.groups); (g.U, g.Unew) = (g.Unew, g.U))
    swz() = (c.kz(g.Unew, g.U, g.sys, g.recon, g.rsol, Float32(dt) / g.dz, g.nx, g.ny, g.nz, Val(N), c.bc, c.pz;
                  threads = c.threads, groups = c.groups); (g.U, g.Unew) = (g.Unew, g.U))
    rev ? (swz(); swy(); swx()) : (swx(); swy(); swz())
    return g
end

function profile_baseline()
    println("\n== Baseline full timestep throughput ==")
    for (label, maker, n, steps) in (
        ("1D n=1048576", euler_grid1, 1_048_576, 120),
        ("2D n=1024", euler_grid2, 1024, 60),
        ("3D n=128", euler_grid3, 128, 24),
    )
        g, d = maker(n)
        dt = 0.05f0 * d
        f = label[1:2] == "1D" ? (() -> (for _ in 1:steps; mstep1d!(g, dt); end)) :
            label[1:2] == "2D" ? (() -> (for i in 1:steps; mstep2d!(g, dt; rev = isodd(i)); end)) :
                                  (() -> (for i in 1:steps; mstep3d!(g, dt; rev = isodd(i)); end))
        seconds = median_seconds(f)
        cells = label[1:2] == "1D" ? n : label[1:2] == "2D" ? n^2 : n^3
        print_rate(label, cells, steps, seconds)
    end
end

function profile_phases()
    println("\n== Per-sweep phase timings ==")
    g2, d2 = euler_grid2(1024)
    dt2 = 0.05f0 * d2
    steps2 = 80
    cells2 = g2.nx * g2.ny
    tx2, ty2 = (16, 16), (16, 16)
    print_rate("2D x sweep default", cells2, steps2, median_seconds(() -> (for _ in 1:steps2; launch_x2!(g2, dt2, tx2); end)))
    print_rate("2D y sweep default", cells2, steps2, median_seconds(() -> (for _ in 1:steps2; launch_y2!(g2, dt2, ty2); end)))

    g3, d3 = euler_grid3(128)
    dt3 = 0.05f0 * d3
    steps3 = 32
    cells3 = g3.nx * g3.ny * g3.nz
    t3, _ = _mcfg3(g3.nx, g3.ny, g3.nz)
    print_rate("3D x fused sweep cfg", cells3, steps3, median_seconds(() -> (for _ in 1:steps3; launch_x3!(g3, dt3, t3); end)))
    print_rate("3D y fused sweep cfg", cells3, steps3, median_seconds(() -> (for _ in 1:steps3; launch_y3!(g3, dt3, t3); end)))
    print_rate("3D z fused sweep cfg", cells3, steps3, median_seconds(() -> (for _ in 1:steps3; launch_z3!(g3, dt3, t3); end)))
end

function profile_speed_reduce()
    println("\n== CFL speed/reduction timings ==")
    for (label, maker, n, steps) in (
        ("1D speed n=1048576", euler_grid1, 1_048_576, 80),
        ("2D speed n=1024", euler_grid2, 1024, 50),
        ("3D speed n=128", euler_grid3, 128, 24),
    )
        g, _ = maker(n)
        f = label[1:2] == "1D" ? (() -> (for _ in 1:steps; mmax_wavespeed_1d(g); end)) :
            label[1:2] == "2D" ? (() -> (for _ in 1:steps; mmax_wavespeed_2d(g); end)) :
                                  (() -> (for _ in 1:steps; mmax_wavespeed_3d(g); end))
        cells = label[1:2] == "1D" ? n : label[1:2] == "2D" ? n^2 : n^3
        print_rate(label, cells, steps, median_seconds(f))
    end
end

function profile_allocations()
    println("\n== Allocation profile ==")
    for (label, maker, n, stepper, steps) in (
        ("2D steps", euler_grid2, 256, (g, d, i) -> mstep2d!(g, 0.05f0 * d; rev = isodd(i)), 20),
        ("3D steps", euler_grid3, 64, (g, d, i) -> mstep3d!(g, 0.05f0 * d; rev = isodd(i)), 12),
    )
        g, d = maker(n)
        stepper(g, d, 1)
        Metal.synchronize()
        stats = Metal.@timed begin
            for i in 1:steps
                stepper(g, d, i)
            end
            Metal.synchronize()
        end
        @printf("%-36s CPU %8s  GPU %8s  GPU allocs %5d\n",
                label, Base.format_bytes(stats.cpu_bytes), Base.format_bytes(stats.gpu_bytes),
                stats.gpu_memstats.alloc_count)
    end
end

function profile_cached_launch()
    println("\n== Host launch path: @metal vs launch=false cached HostKernel ==")
    g2a, d2 = euler_grid2(256)
    g2b, _ = euler_grid2(256)
    dt2 = 0.05f0 * d2
    steps2 = 200
    c2 = cached2(g2b, dt2, (16, 16))
    s2a = median_seconds(() -> (for i in 1:steps2; mstep2d!(g2a, dt2; rev = isodd(i)); end))
    s2b = median_seconds(() -> (for i in 1:steps2; step2_cached!(g2b, dt2, c2; rev = isodd(i)); end))
    print_rate("2D @metal n=256", 256^2, steps2, s2a)
    print_rate("2D cached n=256", 256^2, steps2, s2b)

    g3a, d3 = euler_grid3(64)
    g3b, _ = euler_grid3(64)
    dt3 = 0.05f0 * d3
    steps3 = 120
    t3, _ = _mcfg3(g3b.nx, g3b.ny, g3b.nz)
    c3 = cached3(g3b, dt3, t3)
    s3a = median_seconds(() -> (for i in 1:steps3; step3_shape!(g3a, dt3, t3, t3, t3; rev = isodd(i)); end))
    s3b = median_seconds(() -> (for i in 1:steps3; step3_cached!(g3b, dt3, c3; rev = isodd(i)); end))
    print_rate("3D fused @metal n=64", 64^3, steps3, s3a)
    print_rate("3D fused cached n=64", 64^3, steps3, s3b)
end

function profile_threadgroups()
    println("\n== Fused-sweep threadgroup candidates, full timestep throughput ==")
    g2, d2 = euler_grid2(1024)
    dt2 = 0.05f0 * d2
    steps2 = 60
    candidates2 = (
        ("2D default 16x16", (16, 16), (16, 16)),
        ("2D axis 32x8/8x32", (32, 8), (8, 32)),
        ("2D axis 64x4/4x64", (64, 4), (4, 64)),
        ("2D slim 16x8/8x16", (16, 8), (8, 16)),
        ("2D flat 32x4/4x32", (32, 4), (4, 32)),
    )
    for (label, tx, ty) in candidates2
        seconds = median_seconds(() -> (for i in 1:steps2; step2_shape!(g2, dt2, tx, ty; rev = isodd(i)); end))
        print_rate(label, 1024^2, steps2, seconds)
    end

    g3, d3 = euler_grid3(128)
    dt3 = 0.05f0 * d3
    steps3 = 24
    candidates3 = (
        ("3D fused 16x8x2", (16, 8, 2), (16, 8, 2), (16, 8, 2)),
        ("3D legacy 8x8x4", (8, 8, 4), (8, 8, 4), (8, 8, 4)),
        ("3D axis 16x4x4/4x16x4/4x4x16", (16, 4, 4), (4, 16, 4), (4, 4, 16)),
        ("3D axis 32x4x2/4x32x2/4x4x16", (32, 4, 2), (4, 32, 2), (4, 4, 16)),
        ("3D cube 8x8x8", (8, 8, 8), (8, 8, 8), (8, 8, 8)),
        ("3D half 8x8x2", (8, 8, 2), (8, 8, 2), (8, 8, 2)),
        ("3D wide-x 16x8x2", (16, 8, 2), (16, 8, 2), (16, 8, 2)),
    )
    for (label, tx, ty, tz) in candidates3
        seconds = median_seconds(() -> (for i in 1:steps3; step3_shape!(g3, dt3, tx, ty, tz; rev = isodd(i)); end))
        print_rate(label, 128^3, steps3, seconds)
    end
end

function profile_large_threadgroups()
    println("\n== Large-grid threadgroup confirmation ==")
    for n in (128, 192, 256)
        g0, d = euler_grid3(n)
        g1, _ = euler_grid3(n)
        dt = 0.05f0 * d
        steps = n >= 256 ? 10 : n >= 192 ? 14 : 24
        default = median_seconds(() -> (for i in 1:steps
            step3_shape!(g0, dt, (8, 8, 4), (8, 8, 4), (8, 8, 4); rev = isodd(i))
        end))
        candidate = median_seconds(() -> (for i in 1:steps
            step3_shape!(g1, dt, (16, 8, 2), (16, 8, 2), (16, 8, 2); rev = isodd(i))
        end))
        cells = n^3
        print_rate("3D n=$n legacy 8x8x4", cells, steps, default)
        print_rate("3D n=$n fused 16x8x2", cells, steps, candidate)
        @printf("%-36s %8.3f%%\n", "fused delta", 100 * (default / candidate - 1))
    end
end

println("Metal device: ", Metal.device())
println("mode: ", MODE, "  samples: ", SAMPLES)

if MODE == "large"
    profile_large_threadgroups()
else
    profile_baseline()
    profile_phases()
    profile_speed_reduce()
    profile_allocations()
    profile_cached_launch()
    profile_threadgroups()
    MODE == "full" && profile_large_threadgroups()
end
