using FiniteVolumeGodunovKA
using FiniteVolumeGodunovKA: cons2prim, prim2cons, nconserved, vidx
using Test

const FV = FiniteVolumeGodunovKA

@testset "contract / Euler roundtrip" begin
    s = Euler(γ = 1.4f0)
    @test nconserved(s) == 5
    @test vidx(s) == ((2, 3, 4),)
    W = (1.2f0, 0.3f0, -0.5f0, 0.1f0, 0.9f0)
    @test all(isapprox.(cons2prim(s, prim2cons(s, W)), W; rtol = 1f-5))
end

@testset "packed color/species fractions" begin
    vals = Float32[1f0, 0.3f0, 1f-8, 1f-20, 1f-32]
    for x in vals
        y = unpack_color_fraction(pack_color_fraction(x))
        @test y > 0f0
        @test isapprox(log2(y), log2(max(x, FV.color_fraction_floor())); atol = 110f0 / 65535f0)
    end

    s = Euler(γ = 1.4f0)
    n = 96
    dx = 1f0 / n
    U0 = [prim2cons(s, (1f0 + 0.2f0 * sinpi(2f0 * (i - 0.5f0) / n),
                         0.7f0, 0f0, 0f0, 1f0)) for i in 1:n]
    colors = hcat(fill(0.25f0, n), fill(1f-20, n))
    g = Grid1D(s, U0; dx, bc = :periodic, recon = PLM(), rsol = HLLC(), colors)
    @test ncolors(g) == 2
    for step in 1:10
        FV.step!(g, 0.05f0 * dx)
    end
    X = color_fractions(g)
    @test maximum(abs.(X[:, 1] .- X[1, 1])) <= 2f-4
    @test maximum(abs.(log2.(X[:, 2]) .- log2(X[1, 2]))) <= 2f-3

    colors2 = hcat([0.3f0 + 0.1f0 * sinpi(2f0 * (i - 0.5f0) / n) for i in 1:n],
                   [1f-20 * (1f0 + 0.4f0 * sinpi(2f0 * (i - 0.5f0) / n)) for i in 1:n])
    g2 = Grid1D(s, copy(U0); dx, bc = :periodic, recon = PLM(), rsol = HLLC(), colors = colors2)
    color_mass(g, X, q) = sum(Float32(g.U[i][1]) * X[i, q] for i in 1:g.nx) * dx
    X20 = color_fractions(g2)
    m0 = (color_mass(g2, X20, 1), color_mass(g2, X20, 2))
    for step in 1:20
        FV.step!(g2, 0.03f0 * dx)
    end
    X21 = color_fractions(g2)
    m1 = (color_mass(g2, X21, 1), color_mass(g2, X21, 2))
    rel = abs.((m1 .- m0) ./ m0)
    @test rel[1] <= 5f-5
    @test rel[2] <= 5f-4

    m = 12
    d = 1f0 / m
    U3 = [prim2cons(s, (1f0 + 0.1f0 * sinpi(2f0 * (i + j + k) / m),
                         0.3f0, 0.2f0, 0.1f0, 1f0)) for i in 1:m, j in 1:m, k in 1:m]
    C3 = Array{Float32,4}(undef, m, m, m, 2)
    C3[:, :, :, 1] .= 0.4f0
    C3[:, :, :, 2] .= 1f-18
    g3 = Grid3D(s, U3; dx = d, dy = d, dz = d, bc = :periodic, recon = PLM(), rsol = HLLC(), colors = C3)
    FV.step!(g3, 0.03f0 * d)
    X3 = color_fractions(g3)
    @test size(X3) == (m, m, m, 2)
    @test maximum(abs.(X3[:, :, :, 1] .- X3[1, 1, 1, 1])) <= 3f-4
    @test maximum(abs.(log2.(X3[:, :, :, 2]) .- log2(X3[1, 1, 1, 2]))) <= 3f-3
end

@testset "EulerColors generic backends" begin
    s = EulerColors{2}(γ = 5f0 / 3f0)
    @test nconserved(s) == 7
    W = (1.2f0, 0.3f0, -0.2f0, 0.1f0, 0.9f0, 0.42f0, 0.58f0)
    @test all(isapprox.(cons2prim(s, prim2cons(s, W)), W; rtol = 1f-5))

    n = 12
    d = 1f0 / n
    U0 = [prim2cons(s, (1f0 + 0.1f0 * sinpi(2f0 * Float32(i + j + k) / n),
                         0.3f0, 0.2f0, 0.1f0, 1f0, 0.42f0, 0.58f0))
          for i in 1:n, j in 1:n, k in 1:n]
    g = Grid3D(s, U0; dx = d, dy = d, dz = d, bc = :periodic) # default HLLC must carry colors.
    FV.step!(g, 0.03f0 * d)
    X = primitives(g)
    @test maximum(abs(X[i, j, k][6] - 0.42f0) for i in 1:n, j in 1:n, k in 1:n) <= 1f-5
    @test maximum(abs(X[i, j, k][6] + X[i, j, k][7] - 1f0) for i in 1:n, j in 1:n, k in 1:n) <= 1f-5
end

# ---------------------------------------------------------------------------
# Sod shock tube (Float32, HLLC) — exercises cons2prim/prim2cons/physflux/HLLC,
# the limiter, and the conservative update against the known star state.
# ---------------------------------------------------------------------------
@testset "Sod shock tube (Float32)" begin
    nx = 400; dx = 1f0 / nx
    xs = Float32[(i - 0.5f0) * dx for i in 1:nx]
    s  = Euler(γ = 1.4f0)
    U0 = [prim2cons(s, (x < 0.5f0 ? 1f0 : 0.125f0, 0f0, 0f0, 0f0,
                        x < 0.5f0 ? 1f0 : 0.1f0)) for x in xs]
    g  = Grid1D(s, U0; dx = dx, bc = :outflow, recon = PLM(), rsol = HLLC(), cfl = 0.4f0)

    m0 = sum(u[1] for u in U0) * dx
    FV.evolve!(g, 0.2f0)
    W  = FV.primitives(g)

    @test all(w -> w[1] > 0 && w[5] > 0, W)                 # positivity
    @test isapprox(FV.conserved_total(g)[1], m0; rtol = 1f-4)  # mass conserved

    sample(xc) = begin
        i = argmin(abs.(xs .- xc))
        mean5(f) = sum(f(W[k]) for k in i-2:i+2) / 5
        (ρ = mean5(w -> w[1]), u = mean5(w -> w[2]), P = mean5(w -> w[5]))
    end
    postshock = sample(0.75f0)   # between contact (~0.686) and shock (~0.850)
    leftstar  = sample(0.60f0)   # between rarefaction tail and contact
    @test isapprox(postshock.ρ, 0.26557; rtol = 0.05)
    @test isapprox(postshock.P, 0.30313; rtol = 0.05)
    @test isapprox(postshock.u, 0.92745; rtol = 0.05)
    @test isapprox(leftstar.ρ,  0.42632; rtol = 0.06)
    @test isapprox(leftstar.P,  0.30313; rtol = 0.05)
end

# ---------------------------------------------------------------------------
# Smooth convergence — entropy wave advected one period. Run in Float64 to
# demonstrate the SAME physics is element-type-generic; unlimited PLM → 2nd order
# (TVD limiters clip smooth extrema to ~1st, so :none is the right choice here).
# ---------------------------------------------------------------------------
@testset "entropy-wave convergence (Float64, 2nd order)" begin
    ρ0(x) = 1.0 + 0.2 * sinpi(2x)
    run(nx) = begin
        dx = 1.0 / nx
        xs = [(i - 0.5) * dx for i in 1:nx]
        s  = Euler(γ = 1.4f0)
        U0 = [prim2cons(s, (ρ0(x), 1.0, 0.0, 0.0, 1.0)) for x in xs]
        g  = Grid1D(s, U0; dx = dx, bc = :periodic, recon = PLM(:none), rsol = HLL(), cfl = 0.4)
        FV.evolve!(g, 1.0)
        W = FV.primitives(g)
        sum(abs(W[i][1] - ρ0(xs[i])) for i in 1:nx) * dx     # L1, exact = IC at t=1
    end

    ns   = [16, 32, 64, 128]
    errs = [run(n) for n in ns]
    ord  = [log2(errs[k] / errs[k+1]) for k in 1:length(ns)-1]
    @info "entropy-wave convergence" errs ord
    @test all(diff(errs) .< 0)        # errors decrease with resolution
    @test ord[end] ≥ 1.85             # 2nd-order at the finest pair
end

# ---------------------------------------------------------------------------
# SIMD CPU backend — must be BIT-IDENTICAL to the scalar backend (same physics,
# same Float32 ops, just Vec{8} lanes + a scalar tail). Bit-identity is the
# strongest possible proof the vectorized path runs the same code.
# ---------------------------------------------------------------------------
@testset "SIMD backend ≡ scalar (bit-identical)" begin
    s = Euler(γ = 1.4f0)
    cmp(U0, dx, bc; kw...) = begin
        gsc = Grid1D(s, copy(U0); dx = dx, bc = bc, kw...)
        gsi = Grid1DSoA(s, copy(U0); dx = dx, bc = bc, kw...)
        FV.evolve!(gsc, 0.2f0); FV.evolve_simd!(gsi, 0.2f0)
        Wsc, Wsi = FV.primitives(gsc), FV.primitives_soa(gsi)
        maximum(maximum(abs.(Wsc[i] .- Wsi[i])) for i in 1:length(U0))
    end
    nx = 437; dx = 1f0 / nx                       # deliberately not a multiple of 8 → tail
    xs = Float32[(i - 0.5f0) * dx for i in 1:nx]
    sod  = [prim2cons(s, (x < 0.5f0 ? 1f0 : 0.125f0, 0f0,0f0,0f0, x < 0.5f0 ? 1f0 : 0.1f0)) for x in xs]
    wave = [prim2cons(s, (1f0 + 0.2f0*sinpi(2f0*x), 1f0, 0f0, 0f0, 1f0)) for x in xs]
    @test cmp(sod,  dx, :outflow;  recon = PLM(),     rsol = HLLC()) == 0f0
    @test cmp(wave, dx, :periodic; recon = PLM(),     rsol = HLL())  == 0f0
    @test cmp(wave, dx, :periodic; recon = PLM(:none), rsol = LLF()) == 0f0
end

# ---------------------------------------------------------------------------
# 2D + rotation — the design-defining test. The user wrote only physflux_x; the
# y-flux is obtained by rotating the marked vector components. Isotropy must be
# bit-exact, and the Strang-split 2D scheme must be 2nd order.
# ---------------------------------------------------------------------------
@testset "2D rotation isotropy + convergence" begin
    s = Euler(γ = 1.4f0)

    # one x-sweep on an x-varying Sod ≡ one y-sweep on the same profile along y (u↔v).
    n = 96; d = 1f0/n; m = 4
    xprob = [prim2cons(s, (i <= n÷2 ? 1f0 : 0.125f0, 0.3f0,0f0,0f0, i <= n÷2 ? 1f0 : 0.1f0)) for i in 1:n, _ in 1:m]
    gx = Grid2D(s, xprob; dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC()); FV._sweep_x!(gx, 0.1f0*d)
    yprob = [prim2cons(s, (j <= n÷2 ? 1f0 : 0.125f0, 0f0,0.3f0,0f0, j <= n÷2 ? 1f0 : 0.1f0)) for _ in 1:m, j in 1:n]
    gy = Grid2D(s, yprob; dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC()); FV._sweep_y!(gy, 0.1f0*d)
    iso = maximum(maximum(abs.(gx.U[a,b] .- (gy.U[b,a][1], gy.U[b,a][3], gy.U[b,a][2], gy.U[b,a][4], gy.U[b,a][5])))
                  for a in 1:n, b in 1:m)
    @test iso == 0f0                                   # y-flux-via-rotation ≡ x-flux, bit-exact

    # diagonal entropy wave (Float64), exact at t=1 → 2nd order.
    ρ0(x, y) = 1.0 + 0.2 * sinpi(2 * (x + y))
    err2d(nn) = begin
        dx = 1.0/nn
        U0 = [prim2cons(s, (ρ0((i-0.5)*dx, (j-0.5)*dx), 1.0, 1.0, 0.0, 1.0)) for i in 1:nn, j in 1:nn]
        g = Grid2D(s, U0; dx=dx, dy=dx, bc=:periodic, recon=PLM(:none), rsol=HLL(), cfl=0.4)
        FV.evolve2d!(g, 1.0); W = FV.primitives(g)
        sum(abs(W[i,j][1] - ρ0((i-0.5)*dx, (j-0.5)*dx)) for i in 1:nn, j in 1:nn) * dx * dx
    end
    es = [err2d(nn) for nn in (16, 32, 64)]
    @test all(diff(es) .< 0)
    @test log2(es[2] / es[3]) ≥ 1.9                    # 2nd order at the finest pair
end

# ---------------------------------------------------------------------------
# GLM-MHD through the SAME contract — 9 vars, TWO rotating vectors (momentum + B).
# The payoff test: add variables + a param + the flux + vidx-as-two-triples, and the
# library's rotation handles y/z automatically.
# ---------------------------------------------------------------------------
@testset "GLM-MHD via the contract (Brio-Wu + rotation)" begin
    s = GLMMHD(γ = 2f0, ch = 2f0)
    nx = 800; dx = 1f0/nx; xs = Float32[(i-0.5f0)*dx for i in 1:nx]
    L = (1f0,   0f0,0f0,0f0, 1f0, 0.75f0,  1f0, 0f0, 0f0)
    R = (0.125f0, 0f0,0f0,0f0, 0.1f0, 0.75f0, -1f0, 0f0, 0f0)
    U0 = [prim2cons(s, x < 0.5f0 ? L : R) for x in xs]
    g  = Grid1D(s, U0; dx=dx, bc=:outflow, recon=PLM(), rsol=LLF(), cfl=0.4f0)
    m0 = sum(u[1] for u in U0) * dx
    FV.evolve!(g, 0.1f0); W = FV.primitives(g)
    @test all(w -> all(isfinite, w), W)                       # stable
    @test all(w -> w[1] > 0 && w[5] > 0, W)                   # positivity
    @test maximum(abs(w[6] - 0.75f0) for w in W) == 0f0       # normal field Bx exactly preserved
    @test isapprox(sum(u[1] for u in g.U)*dx, m0; rtol = 1f-4)
    # @source hook: the GLM ψ-damping decays ψ (Euler's default source is identity).
    @test FV.source(s, prim2cons(s, (1f0,0f0,0f0,0f0,1f0, 0.5f0,0.5f0,0f0, 1f0)), 0.1f0)[9] < 1f0
    @test FV.source(Euler(γ=1.4f0), (1f0,2f0,3f0,4f0,5f0), 0.1f0) == (1f0,2f0,3f0,4f0,5f0)

    # rotation isotropy with TWO vectors: one x-sweep ≡ one y-sweep (momentum + B swapped).
    n = 96; d = 1f0/n; m = 4
    xL = (1f0,0f0,0f0,0f0,1f0, 0.75f0, 1f0,0f0,0f0); xR = (0.125f0,0f0,0f0,0f0,0.1f0, 0.75f0,-1f0,0f0,0f0)
    yL = (1f0,0f0,0f0,0f0,1f0, 1f0, 0.75f0,0f0,0f0); yR = (0.125f0,0f0,0f0,0f0,0.1f0, -1f0,0.75f0,0f0,0f0)
    gx = Grid2D(s, [prim2cons(s, i<=n÷2 ? xL : xR) for i in 1:n, _ in 1:m];
                dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=LLF()); FV._sweep_x!(gx, 0.1f0*d)
    gy = Grid2D(s, [prim2cons(s, j<=n÷2 ? yL : yR) for _ in 1:m, j in 1:n];
                dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=LLF()); FV._sweep_y!(gy, 0.1f0*d)
    iso = maximum(begin
              u = gy.U[b,a]
              maximum(abs.(gx.U[a,b] .- (u[1],u[3],u[2],u[4],u[5],u[7],u[6],u[8],u[9])))
          end for a in 1:n, b in 1:m)
    @test iso == 0f0                                          # two-vector rotation, bit-exact
end

# ---------------------------------------------------------------------------
# CT (constrained transport) — face-staggered B + edge-EMF curl → div·B at machine
# zero (vs GLM cleaning's ~2 on the same Orszag-Tang).
# ---------------------------------------------------------------------------
@testset "CT: machine-zero div·B (Orszag-Tang)" begin
    n = 96; dx = 1f0/n; γ = 5f0/3f0; B0 = 1f0/sqrt(4f0*Float32(π))
    ρ0 = 25f0/(36f0*Float32(π)); P0 = 5f0/(12f0*Float32(π)); s = GLMMHD(γ=γ, ch=0f0)
    bx = [(-B0*sinpi(2f0*(j-0.5f0)*dx)) for i in 1:n, j in 1:n]   # Bx on x-faces (y-dependent)
    by = [( B0*sinpi(4f0*(i-0.5f0)*dx)) for i in 1:n, j in 1:n]   # By on y-faces (x-dependent)
    U = Matrix{NTuple{5,Float32}}(undef, n, n)
    for j in 1:n, i in 1:n
        u = -sinpi(2f0*(j-0.5f0)*dx); v = sinpi(2f0*(i-0.5f0)*dx)
        Bxc = 0.5f0*(bx[i,j]+bx[mod1(i+1,n),j]); Byc = 0.5f0*(by[i,j]+by[i,mod1(j+1,n)])
        U[i,j] = (ρ0, ρ0*u, ρ0*v, 0f0, P0/(γ-1) + 0.5f0*ρ0*(u*u+v*v) + 0.5f0*(Bxc*Bxc+Byc*Byc))
    end
    g = Grid2DCT(s, U, bx, by; dx=dx, dy=dx, rsol=LLF(), cfl=0.4f0)
    @test FV.divB_max(g) == 0f0                       # IC divergence-free exactly
    FV.evolve_ct!(g, 0.2f0)
    @test FV.divB_max(g) < 1f-3                       # machine-zero (Float32 roundoff) vs GLM ~2
    @test all(g.U[i,j][1] > 0 for i in 1:n, j in 1:n) # stable, positive density
end

# ---------------------------------------------------------------------------
# 2D SIMD backend (Grid2DSoA) — vectorized along x for both sweeps. Bit-identical
# to the 2D scalar backend for Euler/HLLC AND GLM-MHD/HLLD (the Vec path of every solver).
# ---------------------------------------------------------------------------
@testset "2D SIMD ≡ 2D scalar (bit-identical)" begin
    cmp2d(s, U0, recon, rsol, nsteps) = begin
        n = size(U0, 1); d = 1f0/n
        gsc = Grid2D(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=recon, rsol=rsol)
        gsi = Grid2DSoA(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=recon, rsol=rsol)
        for _ in 1:nsteps; FV.step!(gsc, 0.1f0*d); FV.step!(gsi, 0.1f0*d); end
        Wc = FV.primitives(gsc); Wi = FV.primitives_soa(gsi)
        maximum(maximum(abs.(Wc[i,j] .- Wi[i,j])) for i in 1:n, j in 1:n)
    end
    se = Euler(γ = 1.4f0); n = 64
    U0e = [prim2cons(se, (1f0 + 0.3f0*sinpi(2f0*((i-0.5f0)/n + (j-0.5f0)/n)), 0.5f0, 0.3f0, 0f0, 1f0)) for i in 1:n, j in 1:n]
    @test cmp2d(se, U0e, PLM(), HLLC(), 15) == 0f0
    sm = GLMMHD(γ = 5f0/3f0, ch = 2f0); B0 = 1f0/sqrt(4f0*Float32(π))
    U0m = [prim2cons(sm, (0.5f0, -sinpi(2f0*(j-0.5f0)/n), sinpi(2f0*(i-0.5f0)/n), 0f0, 0.5f0,
                          -B0*sinpi(2f0*(j-0.5f0)/n), B0*sinpi(4f0*(i-0.5f0)/n), 0f0, 0f0)) for i in 1:n, j in 1:n]
    @test cmp2d(sm, U0m, PLM(), HLLD(), 12) == 0f0
end

# ---------------------------------------------------------------------------
# 3D backend (Grid3D) — symmetric Strang x·y·z·y·x. The z-sweep uses dirperm(s,N,3); the
# rotation machinery generalizes to all 3 axes with no new code.
# ---------------------------------------------------------------------------
@testset "3D backend + z-rotation" begin
    se = Euler(γ = 1.4f0)
    # z-rotation isotropy: x-sweep ≡ z-sweep on the transposed problem (u↔w), bit-exact.
    n = 48; d = 1f0/n; m = 4
    xp = [prim2cons(se, (i <= n÷2 ? 1f0 : 0.125f0, 0.3f0,0f0,0f0, i <= n÷2 ? 1f0 : 0.1f0)) for i in 1:n, _ in 1:m, _ in 1:m]
    gx = Grid3D(se, xp; dx=d,dy=d,dz=d, bc=:periodic, recon=PLM(), rsol=HLLC()); FV._sweep_x3d!(gx, 0.1f0*d)
    zp = [prim2cons(se, (k <= n÷2 ? 1f0 : 0.125f0, 0f0,0f0,0.3f0, k <= n÷2 ? 1f0 : 0.1f0)) for _ in 1:m, _ in 1:m, k in 1:n]
    gz = Grid3D(se, zp; dx=d,dy=d,dz=d, bc=:periodic, recon=PLM(), rsol=HLLC()); FV._sweep_z3d!(gz, 0.1f0*d)
    @test maximum(begin u = gz.U[b,c,a]; maximum(abs.(gx.U[a,b,c] .- (u[1],u[4],u[3],u[2],u[5]))) end for a in 1:n, b in 1:m, c in 1:m) == 0f0

    # 3D diagonal entropy wave (Float64) → 2nd order.
    ρ0(x,y,z) = 1.0 + 0.2*sinpi(2*(x+y+z))
    err3d(nn) = begin
        dx = 1.0/nn
        U0 = [prim2cons(se, (ρ0((i-0.5)*dx,(j-0.5)*dx,(k-0.5)*dx), 1.0,1.0,1.0,1.0)) for i in 1:nn, j in 1:nn, k in 1:nn]
        g = Grid3D(se, U0; dx=dx,dy=dx,dz=dx, bc=:periodic, recon=PLM(:none), rsol=HLL(), cfl=0.4)
        FV.evolve3d!(g, 1.0); W = FV.primitives(g)
        sum(abs(W[i,j,k][1] - ρ0((i-0.5)*dx,(j-0.5)*dx,(k-0.5)*dx)) for i in 1:nn, j in 1:nn, k in 1:nn) * dx^3
    end
    es = [err3d(nn) for nn in (12, 18, 24)]
    @test es[3] < es[2] < es[1]
    @test log(es[2]/es[3])/log(24/18) ≥ 1.85

    # GLM-MHD in 3D with HLLD: stable + positive.
    sm = GLMMHD(γ=5f0/3f0, ch=2f0); nn = 16; dd = 1f0/nn; B0 = 1f0/sqrt(4f0*Float32(π))
    icW(x,y,z) = (0.5f0, -sinpi(2f0*y), sinpi(2f0*x), 0f0, 0.5f0, -B0*sinpi(2f0*y), B0*sinpi(4f0*x), 0f0, 0f0)
    U0 = [prim2cons(sm, icW((i-0.5f0)*dd,(j-0.5f0)*dd,(k-0.5f0)*dd)) for i in 1:nn, j in 1:nn, k in 1:nn]
    g = Grid3D(sm, U0; dx=dd,dy=dd,dz=dd, bc=:periodic, recon=PLM(), rsol=HLLD(), cfl=0.4f0); FV.evolve3d!(g, 0.03f0)
    @test all(w -> all(isfinite, w) && w[1] > 0 && w[5] > 0, FV.primitives(g))

    # 3D SIMD bit-identical to scalar (Euler/HLLC + GLM/HLLD).
    nn = 24; dd = 1f0/nn
    U0e = [prim2cons(se, (1f0+0.3f0*sinpi(2f0*(i+j+k-1.5f0)/nn), 0.5f0,0.3f0,0.2f0, 1f0)) for i in 1:nn, j in 1:nn, k in 1:nn]
    cmp3d(s, U0, rsol, ns) = begin
        gsc = Grid3D(s, copy(U0); dx=dd,dy=dd,dz=dd, bc=:periodic, recon=PLM(), rsol=rsol)
        gsi = Grid3DSoA(s, copy(U0); dx=dd,dy=dd,dz=dd, bc=:periodic, recon=PLM(), rsol=rsol)
        for _ in 1:ns; FV.step!(gsc, 0.1f0*dd); FV.step!(gsi, 0.1f0*dd); end
        maximum(maximum(abs.(FV.primitives(gsc)[i,j,k] .- FV.primitives_soa(gsi)[i,j,k])) for i in 1:nn, j in 1:nn, k in 1:nn)
    end
    @test cmp3d(se, U0e, HLLC(), 8) == 0f0
    U0g = [prim2cons(sm, icW((i-0.5f0)/nn,(j-0.5f0)/nn,(k-0.5f0)/nn)) for i in 1:nn, j in 1:nn, k in 1:nn]
    @test cmp3d(sm, U0g, HLLD(), 6) == 0f0
end

# ---------------------------------------------------------------------------
# HLLD — the Miyoshi-Kusano MHD solver, keyed to GLMMHD. Consistency, Brio-Wu
# stability/positivity/conservation, and bit-exact rotation.
# ---------------------------------------------------------------------------
@testset "HLLD MHD Riemann solver" begin
    s = GLMMHD(γ = 2f0, ch = 2f0)
    W = (1f0, 0.3f0, -0.2f0, 0.1f0, 1f0, 0.75f0, 0.5f0, -0.4f0, 0f0)
    @test maximum(abs.(FV.riemann(HLLD(), s, W, W) .- FV.physflux_x(s, W))) == 0f0   # L=R consistency

    nx = 800; dx = 1f0/nx
    bw(i) = i <= nx÷2 ? (1f0,0f0,0f0,0f0,1f0,0.75f0,1f0,0f0,0f0) : (0.125f0,0f0,0f0,0f0,0.1f0,0.75f0,-1f0,0f0,0f0)
    U0 = [prim2cons(s, bw(i)) for i in 1:nx]; m0 = sum(u[1] for u in U0)*dx
    g = Grid1D(s, U0; dx=dx, bc=:outflow, recon=PLM(), rsol=HLLD(), cfl=0.4f0)
    FV.evolve!(g, 0.1f0); W2 = FV.primitives(g)
    @test all(w -> all(isfinite, w), W2)                      # stable
    @test all(w -> w[1] > 0 && w[5] > 0, W2)                  # positive
    @test maximum(abs(w[6] - 0.75f0) for w in W2) == 0f0      # Bx preserved exactly
    @test isapprox(sum(u[1] for u in g.U)*dx, m0; rtol = 1f-4)

    n = 96; d = 1f0/n; m = 4                                  # rotation isotropy, bit-exact
    xL=(1f0,0f0,0f0,0f0,1f0,0.75f0,1f0,0f0,0f0); xR=(0.125f0,0f0,0f0,0f0,0.1f0,0.75f0,-1f0,0f0,0f0)
    yL=(1f0,0f0,0f0,0f0,1f0,1f0,0.75f0,0f0,0f0); yR=(0.125f0,0f0,0f0,0f0,0.1f0,-1f0,0.75f0,0f0,0f0)
    gx = Grid2D(s, [prim2cons(s, i<=n÷2 ? xL : xR) for i in 1:n, _ in 1:m]; dx=d,dy=d,bc=:periodic,recon=PLM(),rsol=HLLD()); FV._sweep_x!(gx, 0.1f0*d)
    gy = Grid2D(s, [prim2cons(s, j<=n÷2 ? yL : yR) for _ in 1:m, j in 1:n]; dx=d,dy=d,bc=:periodic,recon=PLM(),rsol=HLLD()); FV._sweep_y!(gy, 0.1f0*d)
    iso = maximum(begin u = gy.U[b,a]; maximum(abs.(gx.U[a,b] .- (u[1],u[3],u[2],u[4],u[5],u[7],u[6],u[8],u[9]))) end for a in 1:n, b in 1:m)
    @test iso == 0f0
end

# ---------------------------------------------------------------------------
# CUDA backend — same physics, T = Float32 on a GPU thread. Must be bit-identical
# to the scalar backend. Skipped automatically when no functional GPU is present.
# ---------------------------------------------------------------------------
using CUDA
if CUDA.functional()
    @testset "CUDA backend ≡ scalar (bit-identical)" begin
        s = Euler(γ = 1.4f0)
        cmp(U0, dx, bc; kw...) = begin
            gsc = Grid1D(s, copy(U0); dx = dx, bc = bc, kw...)
            gcu = Grid1DCU(s, copy(U0); dx = dx, bc = bc, kw...)
            FV.evolve!(gsc, 0.2f0); FV.evolve_cuda!(gcu, 0.2f0)
            Wsc, Wcu = FV.primitives(gsc), FV.primitives_cuda(gcu)
            maximum(maximum(abs.(Wsc[i] .- Wcu[i])) for i in 1:length(U0))
        end
        nx = 4001; dx = 1f0 / nx
        xs = Float32[(i - 0.5f0) * dx for i in 1:nx]
        sod  = [prim2cons(s, (x < 0.5f0 ? 1f0 : 0.125f0, 0f0,0f0,0f0, x < 0.5f0 ? 1f0 : 0.1f0)) for x in xs]
        wave = [prim2cons(s, (1f0 + 0.2f0*sinpi(2f0*x), 1f0, 0f0, 0f0, 1f0)) for x in xs]
        @test cmp(sod,  dx, :outflow;  recon = PLM(), rsol = HLLC()) == 0f0
        @test cmp(wave, dx, :periodic; recon = PLM(), rsol = HLL())  == 0f0
    end

    @testset "3D CUDA ≡ 3D scalar" begin
        s = Euler(γ = 1.4f0); n = 24; d = 1f0/n
        U0 = [prim2cons(s, (1f0+0.3f0*sinpi(2f0*(i+j+k-1.5f0)/n), 0.5f0,0.3f0,0.2f0, 1f0)) for i in 1:n, j in 1:n, k in 1:n]
        gsc = Grid3D(s, copy(U0); dx=d,dy=d,dz=d, bc=:periodic, recon=PLM(), rsol=HLLC())
        gg  = Grid3DCU(s, copy(U0); dx=d,dy=d,dz=d, bc=:periodic, recon=PLM(), rsol=HLLC())
        for _ in 1:8; FV.step!(gsc, 0.1f0*d); FV.step!(gg, 0.1f0*d); end
        Wc = FV.primitives(gsc); Wg = FV.primitives(gg)
        @test maximum(maximum(abs.(Wc[i,j,k] .- Wg[i,j,k])) for i in 1:n, j in 1:n, k in 1:n) == 0f0
    end

    @testset "2D CUDA ≡ 2D CPU (rotation on GPU)" begin
        s = Euler(γ = 1.4f0); n = 64; d = 1f0/n
        ρ0(x, y) = 1f0 + 0.3f0*sinpi(2f0*(x + y))
        U0 = [prim2cons(s, (ρ0((i-0.5f0)*d, (j-0.5f0)*d), 0.5f0, 0.3f0, 0f0, 1f0)) for i in 1:n, j in 1:n]
        gc = Grid2D(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC())
        gg = Grid2DCU(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC())
        for _ in 1:15; FV.step!(gc, 0.1f0*d); FV.step!(gg, 0.1f0*d); end
        Wc = FV.primitives(gc); Wg = FV.primitives(gg)
        @test maximum(maximum(abs.(Wc[i,j] .- Wg[i,j])) for i in 1:n, j in 1:n) == 0f0
    end

    # transpile-to-CUDA-C performance backend (needs nvcc — skipped if absent)
    if (try; FV._find_nvcc(); true; catch; false; end)
        @testset "transpile-to-CUDA-C (Grid3DCuMarch)" begin
            for s in (Euler(γ=1.4f0), GLMMHD(γ=5f0/3f0, ch=2f0))
                @test transpile_selfcheck(s) == 0f0          # transpiled C physics ≡ Julia, bit-exact
            end

            @testset "EulerColors{$NC}: passive color/species advection" for NC in (1, 2, 3)
                @test transpile_selfcheck(EulerColors{NC}(γ=5f0/3f0)) == 0f0   # unrolled color C ≡ Julia, bit-exact
                γc = 5f0/3f0; nc3 = (48, 16, 16); dxc = 1f0/nc3[1]
                # smooth periodic hydro IC; colors X₁=0.42 (uniform), X₂=1-X₁+blob, X₃=blob' (trace ~1e-20)
                mk(i,j,k) = begin
                    x=2f0π*(i-1)/nc3[1]; y=2f0π*(j-1)/nc3[2]; z=2f0π*(k-1)/nc3[3]
                    ρ=1f0+0.3f0*sin(x)*cos(y)+0.2f0*sin(z); u=0.6f0+0.05f0*sin(y); v=0.04f0*cos(z); w=0.03f0*sin(x)
                    P=1f0+0.1f0*cos(x)*sin(y); E=P/(γc-1f0)+0.5f0*ρ*(u*u+v*v+w*w)
                    blob=0.2f0*exp(-(Float32(i-24)^2+Float32(j-8)^2+Float32(k-8)^2)/8f0)
                    (ρ,ρ*u,ρ*v,ρ*w,E, ρ*0.42f0, ρ*(0.58f0+blob), ρ*(1f-20+blob*1f-20))
                end
                Uc = [ mk(i,j,k)[1:5+NC] for i in 1:nc3[1], j in 1:nc3[2], k in 1:nc3[3] ]
                Uh = [ mk(i,j,k)[1:5]     for i in 1:nc3[1], j in 1:nc3[2], k in 1:nc3[3] ]
                gc = Grid3DCuMarch(EulerColors{NC}(γ=γc), Uc; dx=dxc)
                gh = Grid3DCuMarch(Euler(γ=γc), Uh; dx=dxc)
                dt = 0.4f0*dxc/2f0
                for _ in 1:6; FV.run_ctu!(gc, dt, 1); FV.run_ctu!(gh, dt, 1); end
                VOL=prod(nc3); blk(R,c)=reshape(Array(R)[(c-1)*VOL+1:c*VOL], nc3...)
                rel(a,b)=maximum(abs.(Float64.(a).-Float64.(b)))/(maximum(abs.(Float64.(b)))+eps())
                # colors are PASSIVE: the 5 hydro vars track a plain Euler run to f32 round-off
                for c in 1:5; @test rel(blk(gc.R,c), blk(gh.R,c)) < 1f-5; end
                ρ = blk(gc.R,1)
                @test maximum(abs.((blk(gc.R,6)./ρ) .- 0.42f0)) < 1f-5      # uniform color stays uniform (CMA)
                NC >= 2 && @test maximum(abs.(((blk(gc.R,6).+blk(gc.R,7))./ρ) .- 1f0)) < 1f-4  # ΣX preserved
                for c in 6:5+NC                                              # conservation + finiteness per color
                    @test all(isfinite, blk(gc.R,c))
                    @test isapprox(sum(Float64.(blk(gc.R,c))), sum(Float64(Uc[i,j,k][c]) for i in 1:nc3[1],j in 1:nc3[2],k in 1:nc3[3]); rtol=1f-4)
                end
            end

            # The fast f16-TILED CTU (run_ctus!) carries colours too, via an exact f32 side-channel that
            # advects the species in float (so a trace X~1e-25 does NOT underflow __half) and re-attaches
            # them to the f16 hydro density — uniform-X / ΣX=1 preserved, X matches the f32 run_ctu! path.
            @testset "EulerColors{$NC}: f16-tiled run_ctus! colours" for NC in (1, 2, 3)
                γc = 5f0/3f0; nc3 = (48, 16, 16); dxc = 1f0/nc3[1]
                mk(i,j,k) = begin                                            # ΣX₁₂=1 IC + a trace X₃~1e-25
                    x=2f0π*(i-1)/nc3[1]; y=2f0π*(j-1)/nc3[2]; z=2f0π*(k-1)/nc3[3]
                    ρ=1f0+0.3f0*sin(x)*cos(y)+0.2f0*sin(z); u=0.6f0+0.05f0*sin(y); v=0.04f0*cos(z); w=0.03f0*sin(x)
                    P=1f0+0.1f0*cos(x)*sin(y); E=P/(γc-1f0)+0.5f0*ρ*(u*u+v*v+w*w)
                    X1=0.3f0+0.2f0*exp(-(Float32(i-24)^2)/20f0); X2=1f0-X1
                    (ρ,ρ*u,ρ*v,ρ*w,E, ρ*X1, ρ*X2, ρ*1f-25)[1:5+NC]
                end
                Uc = [ mk(i,j,k) for i in 1:nc3[1], j in 1:nc3[2], k in 1:nc3[3] ]
                g16 = Grid3DCuMarch(EulerColors{NC}(γ=γc), copy(Uc); dx=dxc)
                g32 = Grid3DCuMarch(EulerColors{NC}(γ=γc), copy(Uc); dx=dxc)
                dt = 0.4f0*dxc/2f0
                for _ in 1:8; FV.run_ctus!(g16, dt, 1); FV.run_ctu!(g32, dt, 1); end
                VOL=prod(nc3); blk(R,c)=reshape(Array(R)[(c-1)*VOL+1:c*VOL], nc3...)
                ρ = blk(g16.R,1)
                for c in 6:5+NC                                              # all colours finite + conserved
                    @test all(isfinite, blk(g16.R,c))
                    @test isapprox(sum(Float64.(blk(g16.R,c))), sum(Float64(Uc[i,j,k][c]) for i in 1:nc3[1],j in 1:nc3[2],k in 1:nc3[3]); rtol=1f-4)
                end
                # X matches the f32 path to f16-level; the trace species advected (didn't underflow to 0)
                for c in 6:5+NC; @test maximum(abs.((blk(g16.R,c)./ρ) .- (blk(g32.R,c)./blk(g32.R,1)))) < 5f-3; end
                NC >= 3 && @test minimum(blk(g16.R,8)./ρ) > 1f-26          # trace X₃~1e-25 stays O(1e-25), not 0
                NC >= 2 && @test maximum(abs.(((blk(g16.R,6).+blk(g16.R,7))./ρ) .- 1f0)) < 1f-4  # ΣX preserved
            end

            # DUAL-ENERGY: a COLD gas (eint ≪ KE, E in the f16 subnormal range) that NaNs the single-energy
            # f16-tiled run_ctus! survives ALL-f16 in EulerDE — pressure comes from the evolved Ge, never the
            # cancelling E−½ρv².  Premise + headline + eint preservation + (rho,v) match to the f32 DE path.
            @testset "EulerDE: ALL-f16 dual-energy on cold gas (no NaN, eint preserved)" begin
                γd=5f0/3f0; nd=(32,16,16); dxd=Float32(2π/nd[1]); ρ0=0.17f0; e0=6.96f-8; v0=4f-4
                # KE ≫ eint and E ~ 2e-7 (f16 subnormal) — the cancellation regime of the real cosmology state.
                mkd(i,j,k) = begin
                    x=2f0π*(i-1)/nd[1]; y=2f0π*(j-1)/nd[2]; z=2f0π*(k-1)/nd[3]
                    ρ=ρ0*(1f0+0.03f0*sin(x)); u=v0*(1f0+0.1f0*sin(y)); v=v0*0.1f0*cos(z); w=v0*0.1f0*sin(x+y)
                    eint=e0*(1f0+0.02f0*cos(z)); Ge=ρ*eint; E=Ge+0.5f0*ρ*(u*u+v*v+w*w)
                    (ρ,ρ*u,ρ*v,ρ*w,E,Ge)
                end
                Ud=[mkd(i,j,k) for i in 1:nd[1], j in 1:nd[2], k in 1:nd[3]]
                # premise: single-energy 5-var Euler all-f16 run_ctus! NaNs this cold state
                U5=[Ud[i,j,k][1:5] for i in 1:nd[1], j in 1:nd[2], k in 1:nd[3]]
                gE=Grid3DCuMarch(Euler(γ=γd), U5; dx=dxd); dtd=dt_cfl(gE; cfl=0.3f0)
                FV.run_ctus!(gE, dtd, 1)
                @test any(x->!isfinite(x), Array(gE.R))                    # single-energy f16 DOES NaN
                # headline: all-f16 dual-energy runs cold gas with NO NaN, eint preserved (~6.96e-8), rho>0
                sde=EulerDE(γ=γd, η=1f-3)
                g16=Grid3DCuMarch(sde, copy(Ud); dx=dxd, de_prec=:f16)
                gf=Grid3DCuMarch(sde, copy(Ud); dx=dxd)                    # f32 DE reference (run_ctu!)
                FV.run_ctus_de16!(g16, dtd, 20); FV.run_ctu!(gf, dtd, 20)
                W16=FV.primitives(g16); Wf=FV.primitives(gf)
                @test all(w->all(isfinite,w), W16)                        # no NaN/Inf
                @test all(w->w[1]>0, W16)                                 # rho>0
                e16=[w[6] for w in W16]
                @test all(0.9f0*e0 .< e16 .< 1.1f0*e0)                    # eint preserved ≈6.96e-8, not flushed to 0
                @test minimum(e16) > 1f-9                                  # explicitly NOT lost to subnormal underflow
                # (rho,v) match the f32 dual-energy path to f16 tolerance
                @test maximum(abs(W16[i][1]-Wf[i][1]) for i in eachindex(Wf)) < 5f-3*ρ0
                @test maximum(abs(W16[i][2]-Wf[i][2]) for i in eachindex(Wf)) < 5f-2*v0 + 1f-5
            end

            # DUAL-ENERGY + PASSIVE COLOURS (EulerDEColors): the all-f16 de16 path carries the colour
            # slots in the SAME f16 tile (normal-valued X≈0.05, no underflow) and advects them on the mass
            # flux — so a COLD gas (eint ≪ KE) runs all-f16 with NO NaN AND a tracer that advects, stays
            # bounded, and conserves ΣρX.  Mirrors the EulerDE cold-gas test with a colour added.
            @testset "EulerDEColors{1}: all-f16 dual-energy + advected colour (cold gas)" begin
                @test transpile_selfcheck(EulerDEColors{1}(γ=5f0/3f0, η=1f-3)) == 0f0   # C ≡ Julia, bit-exact
                γd=5f0/3f0; nd=(64,64,64); dxd=Float32(2π/nd[1]); ρ0=0.17f0; e0=1f-6; v0=1.5f-3
                sph(i,j,k) = ((i-32f0)^2+(j-32f0)^2+(k-32f0)^2) < 12f0^2   # sharp colour sphere
                mkc(i,j,k) = begin
                    x=2f0π*(i-1)/nd[1]; y=2f0π*(j-1)/nd[2]; z=2f0π*(k-1)/nd[3]
                    ρ=ρ0*(1f0+0.03f0*sin(x)); u=v0*(1f0+0.1f0*sin(y)); v=v0*0.1f0*cos(z); w=v0*0.1f0*sin(x+y)
                    eint=e0*(1f0+0.02f0*cos(z)); Ge=ρ*eint; E=Ge+0.5f0*ρ*(u*u+v*v+w*w)
                    (ρ,ρ*u,ρ*v,ρ*w,E,Ge, ρ*(sph(i,j,k) ? 0.09f0 : 0.05f0))
                end
                Uc=[mkc(i,j,k) for i in 1:nd[1], j in 1:nd[2], k in 1:nd[3]]
                sdc=EulerDEColors{1}(γ=γd, η=1f-3)
                g16=Grid3DCuMarch(sdc, copy(Uc); dx=dxd, de_prec=:f16, riemann=:hllc)
                gf =Grid3DCuMarch(sdc, copy(Uc); dx=dxd, riemann=:hllc)   # f32 DE+colour reference (run_ctu!)
                dtd=dt_cfl(g16; cfl=0.3f0)
                X0=[Uc[i,j,k][7]/Uc[i,j,k][1] for i in 1:nd[1], j in 1:nd[2], k in 1:nd[3]]
                mass0=sum(Float64(Uc[i,j,k][7]) for i in 1:nd[1],j in 1:nd[2],k in 1:nd[3])
                FV.run_ctus_de16!(g16, dtd, 30); FV.run_ctu!(gf, dtd, 30)
                W16=FV.primitives(g16); Wf=FV.primitives(gf)
                @test all(w->all(isfinite,w), W16)                       # no NaN/Inf
                @test all(w->w[1]>0, W16)                                # rho>0
                e16=[w[6] for w in W16]
                @test all(0.9f0*e0 .< e16 .< 1.1f0*e0)                   # eint=Ge/ρ preserved ≈1e-6
                @test minimum(e16) > 1f-9                                # not flushed to subnormal 0
                X16=[w[7] for w in W16]; Xf=[w[7] for w in Wf]
                @test all(0f0 .<= X16 .<= 1f0)                           # CMA boundedness 0≤X≤1
                @test maximum(abs.(X16 .- Xf)) < 5f-3                    # colour matches f32 to f16 tol
                @test maximum(abs.(X16 .- X0)) > 1f-2                    # colour ADVECTED (didn't freeze)
                VOL=prod(nd); blk(R,c)=reshape(Array(R)[(c-1)*VOL+1:c*VOL], nd...)
                @test isapprox(sum(Float64.(blk(g16.R,7))), mass0; rtol=1f-6)   # ΣρX conserved to round-off
                @test maximum(abs(W16[i][1]-Wf[i][1]) for i in eachindex(Wf)) < 5f-3*ρ0   # ρ match f32
                @test maximum(abs(W16[i][2]-Wf[i][2]) for i in eachindex(Wf)) < 5f-2*v0+1f-5  # u match f32
            end

            # f16-STORAGE grid buffer (store=:f16): the conserved state lives in __half GLOBAL memory (not just
            # the compute tile), halving the largest persistent allocation.  Energies (E,Ge) are GE_SCALE-lifted
            # into f16's normal range on store and un-lifted on load; ρ, momenta, colours stored raw.  Validates
            # vs the f32-buffer de16 (same IC, same GE_SCALE): memory halving + zero NaN + eint preserved.
            @testset "EulerDEColors{1}: f16-STORAGE de16 (halved buffer, cold gas)" begin
                γd=5f0/3f0; GS=1f7; nd=(64,64,64); dxd=Float32(2π/nd[1]); ρ0=0.17f0; e0=6.96f-8; v0=4f-4
                sph(i,j,k) = ((i-32f0)^2+(j-32f0)^2+(k-32f0)^2) < 12f0^2
                mks(i,j,k) = begin
                    x=2f0π*(i-1)/nd[1]; y=2f0π*(j-1)/nd[2]; z=2f0π*(k-1)/nd[3]
                    ρ=ρ0*(1f0+0.03f0*sin(x)); u=v0*(1f0+0.1f0*sin(y)); v=v0*0.1f0*cos(z); w=v0*0.1f0*sin(x+y)
                    eint=e0*(1f0+0.02f0*cos(z)); Ge=ρ*eint; E=Ge+0.5f0*ρ*(u*u+v*v+w*w)
                    (ρ,ρ*u,ρ*v,ρ*w,E,Ge, ρ*(sph(i,j,k) ? 0.09f0 : 0.05f0))
                end
                Us=[mks(i,j,k) for i in 1:nd[1], j in 1:nd[2], k in 1:nd[3]]
                sds=EulerDEColors{1}(γ=γd, η=1f-3)
                g32=Grid3DCuMarch(sds, copy(Us); dx=dxd, de_prec=:f16, ge_scale=GS, riemann=:hllc)               # f32 store
                g16=Grid3DCuMarch(sds, copy(Us); dx=dxd, de_prec=:f16, ge_scale=GS, riemann=:hllc, store=:f16)   # f16 store
                @test eltype(g16.R) == Float16                                  # 1. storage IS f16
                @test sizeof(g16.R) == sizeof(g32.R) ÷ 2                        #    and HALF the bytes of f32 store
                @test g16.store === :f16
                dtd=dt_cfl(g32; cfl=0.3f0)
                FV.run_ctus_de16!(g32, dtd, 30); FV.run_ctus_de16!(g16, dtd, 30)
                W32=FV.primitives(g32; ge_scale=GS); W16=FV.primitives(g16; ge_scale=GS)
                @test all(w->all(isfinite,w), W16)                             # 2. zero NaN/Inf
                @test all(w->w[1]>0, W16)                                       #    rho>0
                e16=[w[6] for w in W16]
                @test all(0.5f0*e0 .< e16 .< 1.5f0*e0)                          #    eint preserved ≈6.96e-8
                @test minimum(e16) > 1f-9                                       #    NOT flushed to subnormal 0
                # f16-store vs f32-store de16: ρ,v,X track to f16 STORAGE tolerance (lossier than f16-compute alone:
                # the momenta sit near the f16 subnormal floor ρu≈6.8e-5, so KE — hence the energy slots — carries
                # f16 mantissa noise; eint's relative diff is the largest, but its ABSOLUTE value is preserved).
                @test maximum(abs(W16[i][1]-W32[i][1]) for i in eachindex(W32)) < 3f-2*ρ0   # ρ to f16-store tol
                @test maximum(abs(W16[i][2]-W32[i][2]) for i in eachindex(W32)) < 5f-2*v0 + 1f-4
                X16=[w[7] for w in W16]; X32=[w[7] for w in W32]
                @test all(0f0 .<= X16 .<= 1f0)                                  #    colour bounded
                @test maximum(abs.(X16 .- X32)) < 5f-3                          #    colour matches f32 store
                # momenta stored RAW in f16 are NOT flushed to zero (above the subnormal floor)
                VOL=prod(nd); ru16=Float32.(Array(g16.R)[VOL+1:2VOL])
                @test count(==(0f0), ru16) == 0
            end

            # f16-STORAGE on WARM/SHOCK gas (Sod): energies are normal-valued (GE_SCALE=1), so the f16 store
            # tracks the f32 store tightly — the f16-store error here is pure f16 quantization of normal values.
            @testset "EulerDE: f16-STORAGE de16 warm/shock (Sod) matches f32 store" begin
                γd=5f0/3f0; nd=(64,64,64); dxd=Float32(1f0/nd[1])
                mkw(i,j,k)=begin
                    left = i <= nd[1]÷2; ρ=left ? 1f0 : 0.125f0; P=left ? 1f0 : 0.1f0
                    Ge=P/(γd-1f0); E=Ge; (ρ,0f0,0f0,0f0,E,Ge)
                end
                Uw=[mkw(i,j,k) for i in 1:nd[1], j in 1:nd[2], k in 1:nd[3]]; sdw=EulerDE(γ=γd, η=1f-3)
                g32=Grid3DCuMarch(sdw, copy(Uw); dx=dxd, de_prec=:f16)
                g16=Grid3DCuMarch(sdw, copy(Uw); dx=dxd, de_prec=:f16, store=:f16)
                @test eltype(g16.R) == Float16
                dtd=dt_cfl(g32; cfl=0.3f0)
                FV.run_ctus_de16!(g32, dtd, 30); FV.run_ctus_de16!(g16, dtd, 30)
                W32=FV.primitives(g32); W16=FV.primitives(g16)
                @test all(w->all(isfinite,w), W16)
                @test maximum(abs(W16[i][1]-W32[i][1]) for i in eachindex(W32)) < 5f-3   # ρ to f16-store tol
                @test maximum(abs(W16[i][5]-W32[i][5]) for i in eachindex(W32)) < 5f-3   # P to f16-store tol
            end

            s = Euler(γ=1.4f0); n = 32; d = 1f0/n
            U0 = [prim2cons(s, (1f0+0.1f0*sinpi(2f0*Float32(i+j+k)/n), 0.2f0,0.1f0,0.1f0, 1f0)) for i in 1:n, j in 1:n, k in 1:n]
            g = Grid3DCuMarch(s, U0; dx=d); FV.run!(g, 0.2f0*d, 20); W = FV.primitives(g)
            @test all(w -> all(isfinite, w) && w[1] > 0 && w[5] > 0, W)

            # the SOLVER guarantees: evolve! is 2nd-order (MUSCL+SSP-RK2) and exactly conservative
            ρe(x,y,z,t) = 1f0 + 0.2f0*sinpi(2f0*(x+y+z-3f0*t))
            ent(m) = [prim2cons(s, (ρe((i-.5f0)/m,(j-.5f0)/m,(k-.5f0)/m,0f0),1f0,1f0,1f0,1f0)) for i in 1:m,j in 1:m,k in 1:m]
            l1(m) = (g2=Grid3DCuMarch(s, ent(m); dx=1f0/m); evolve!(g2, 0.1f0); ρ=[w[1] for w in primitives(g2)];
                     sum(abs(ρ[i,j,k]-ρe((i-.5f0)/m,(j-.5f0)/m,(k-.5f0)/m,0.1f0)) for i in 1:m,j in 1:m,k in 1:m)/m^3)
            e16, e32 = l1(16), l1(32)
            @test log(e16/e32)/log(2) > 1.7            # genuine 2nd-order convergence

            gc = Grid3DCuMarch(s, ent(32); dx=1f0/32); c0 = conserved_total(gc)
            evolve!(gc, 0.15f0); c1 = conserved_total(gc)
            @test abs(c1[1]-c0[1]) ≤ 1f-5*abs(c0[1])   # mass conserved to machine precision
            @test abs(c1[5]-c0[5]) ≤ 1f-5*abs(c0[5])   # energy conserved

            # single-pass CTU schemes (naive + shared-memory-tiled f16): both genuinely 2nd-order too
            ctconv(stepper, m) = (g3=Grid3DCuMarch(s, ent(m); dx=1f0/m); t=0f0;
                     while t<0.1f0; dt=min(dt_cfl(g3), 0.1f0-t); stepper(g3,dt,1); t+=dt; end;
                     ρ=[w[1] for w in primitives(g3)];
                     sum(abs(ρ[i,j,k]-ρe((i-.5f0)/m,(j-.5f0)/m,(k-.5f0)/m,0.1f0)) for i in 1:m,j in 1:m,k in 1:m)/m^3)
            @test log(ctconv(run_ctu!,16)/ctconv(run_ctu!,32))/log(2) > 1.7     # naive CTU 2nd-order
            @test log(ctconv(run_ctus!,16)/ctconv(run_ctus!,32))/log(2) > 1.7   # tiled f16 CTU 2nd-order
            @test log(ctconv(run_ctum!,16)/ctconv(run_ctum!,32))/log(2) > 1.7   # streaming z-march 2nd-order
            @test log(ctconv(run_ctumh!,16)/ctconv(run_ctumh!,32))/log(2) > 1.5  # f16-arith march: 2nd-order to the f16 floor
        end
    else
        @info "nvcc not found — skipping transpile backend tests"
    end
else
    @info "CUDA not functional — skipping GPU backend tests"
end
