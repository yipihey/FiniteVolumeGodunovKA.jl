using Printf
using Statistics

include(joinpath(@__DIR__, "..", "metal", "metal.jl"))

const SAMPLES = parse(Int, get(ENV, "METAL_3D_SAMPLES", "7"))
const SIZES = Tuple(parse.(Int, split(get(ENV, "METAL_3D_SIZES", "64,96,128,192"), ",")))

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
    @printf("%-24s %9.3f ms  %9.1f Mcell/s\n", label, 1e3 * seconds / steps, rate)
    return rate
end

println("Metal device: ", Metal.device())
println("samples: ", SAMPLES, "  sizes: ", SIZES)

for n in SIZES
    g, d = euler_grid3(n)
    dt = 0.05f0 * d
    steps = n >= 192 ? 10 : n >= 128 ? 16 : n >= 96 ? 24 : 48
    seconds = median_seconds(() -> (for step in 1:steps
        mstep3d!(g, dt; rev = isodd(step))
    end))
    print_rate("3D Euler n=$n", n^3, steps, seconds)
end
