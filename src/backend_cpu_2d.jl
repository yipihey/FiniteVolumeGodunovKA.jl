# 2D CPU backend (v0): scalar reference, Strang dimensional splitting.
#
# A step is x(dt/2) · y(dt) · x(dt/2) — each sweep a full 1D MUSCL-Hancock update along its
# direction, 2nd-order in time by Strang composition. The y-sweep reuses the SAME per-cell
# physics through `_update_dir` with the rotation perm: the user wrote only physflux_x. This
# is the test that "the library rotates for y/z" actually holds for a real 2D flow.

mutable struct Grid2D{N,T,S<:FVSystem,R,RS}
    sys::S
    recon::R
    rsol::RS
    U::Matrix{NTuple{N,T}}     # (nx, ny)
    Ut::Matrix{NTuple{N,T}}    # ping-pong buffer
    colors::Union{Nothing,Array{UInt16,3}}
    colorst::Union{Nothing,Array{UInt16,3}}
    nx::Int
    ny::Int
    dx::T
    dy::T
    bc::Symbol
    cfl::T
end

function Grid2D(sys::FVSystem, U0::Matrix{NTuple{N,T}};
                dx, dy, bc::Symbol = :outflow, recon = PLM(), rsol = HLLC(),
                cfl = T(0.4), colors=nothing) where {N,T}
    nx, ny = size(U0)
    C, Ct = _pack_colors(colors, (nx, ny))
    Grid2D{N,T,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, copy(U0), similar(U0), C, Ct, nx, ny, T(dx), T(dy), bc, T(cfl))
end

function _sweep_x!(g::Grid2D{N,T}, dt) where {N,T}
    s, r, rs = g.sys, g.recon, g.rsol
    λ = T(dt) / g.dx; nx, ny = g.nx, g.ny; bc = Val(g.bc); perm = identperm(Val(N))
    U = g.U
    @inbounds for j in 1:ny, i in 1:nx
        im2 = U[_gidx(i-2, nx, bc), j]; im1 = U[_gidx(i-1, nx, bc), j]; i0 = U[i, j]
        ip1 = U[_gidx(i+1, nx, bc), j]; ip2 = U[_gidx(i+2, nx, bc), j]
        g.Ut[i, j] = _update_dir(s, r, rs, im2, im1, i0, ip1, ip2, λ, perm)
    end
    if g.colors !== nothing
        C = g.colors; Ct = g.colorst
        @inbounds for q in 1:size(C, 3), j in 1:ny, i in 1:nx
            Ct[i, j, q] = _update_packed_color(s, r, rs,
                U[_gidx(i - 2, nx, bc), j], U[_gidx(i - 1, nx, bc), j], U[i, j],
                U[_gidx(i + 1, nx, bc), j], U[_gidx(i + 2, nx, bc), j],
                C[_gidx(i - 2, nx, bc), j, q], C[_gidx(i - 1, nx, bc), j, q], C[i, j, q],
                C[_gidx(i + 1, nx, bc), j, q], C[_gidx(i + 2, nx, bc), j, q],
                λ, perm, g.Ut[i, j][1])
        end
        g.colors, g.colorst = g.colorst, g.colors
    end
    g.U, g.Ut = g.Ut, g.U
end

function _sweep_y!(g::Grid2D{N,T}, dt) where {N,T}
    s, r, rs = g.sys, g.recon, g.rsol
    λ = T(dt) / g.dy; nx, ny = g.nx, g.ny; bc = Val(g.bc); perm = dirperm(s, N, 2)
    U = g.U
    @inbounds for i in 1:nx, j in 1:ny
        jm2 = U[i, _gidx(j-2, ny, bc)]; jm1 = U[i, _gidx(j-1, ny, bc)]; j0 = U[i, j]
        jp1 = U[i, _gidx(j+1, ny, bc)]; jp2 = U[i, _gidx(j+2, ny, bc)]
        g.Ut[i, j] = _update_dir(s, r, rs, jm2, jm1, j0, jp1, jp2, λ, perm)
    end
    if g.colors !== nothing
        C = g.colors; Ct = g.colorst
        @inbounds for q in 1:size(C, 3), i in 1:nx, j in 1:ny
            Ct[i, j, q] = _update_packed_color(s, r, rs,
                U[i, _gidx(j - 2, ny, bc)], U[i, _gidx(j - 1, ny, bc)], U[i, j],
                U[i, _gidx(j + 1, ny, bc)], U[i, _gidx(j + 2, ny, bc)],
                C[i, _gidx(j - 2, ny, bc), q], C[i, _gidx(j - 1, ny, bc), q], C[i, j, q],
                C[i, _gidx(j + 1, ny, bc), q], C[i, _gidx(j + 2, ny, bc), q],
                λ, perm, g.Ut[i, j][1])
        end
        g.colors, g.colorst = g.colorst, g.colors
    end
    g.U, g.Ut = g.Ut, g.U
end

# 2 full-dt sweeps + alternating order across steps (x·y then y·x → 2nd-order pair over 2 steps),
# then the source (skipped when the system has none). 3 passes → 2.
function step!(g::Grid2D, dt; rev::Bool = false)
    if rev
        _sweep_y!(g, dt); _sweep_x!(g, dt)
    else
        _sweep_x!(g, dt); _sweep_y!(g, dt)
    end
    if has_source(g.sys)
        s = g.sys
        @inbounds for j in 1:g.ny, i in 1:g.nx
            g.U[i, j] = source(s, g.U[i, j], dt)
        end
    end
    return g
end

function max_wavespeed(g::Grid2D{N}) where {N}
    s = g.sys; a = zero(g.dx); py = dirperm(s, N, 2)
    @inbounds for j in 1:g.ny, i in 1:g.nx
        W = cons2prim(s, g.U[i, j])
        a = max(a, fastspeed_x(s, W), fastspeed_x(s, _swap(W, py)))
    end
    return a   # max fast SPEED over cells & directions
end

function evolve2d!(g::Grid2D, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        c = max_wavespeed(g)
        g.sys = prestep(g.sys, c)                          # dynamic cleaning speed
        dt = min(g.cfl * min(g.dx, g.dy) / c, tend - t)    # exact for dx=dy
        step!(g, dt; rev = isodd(n))
        t += dt; n += 1
    end
    return g
end

primitives(g::Grid2D{N}) where {N} = [cons2prim(g.sys, g.U[i, j]) for i in 1:g.nx, j in 1:g.ny]

function conserved_total(g::Grid2D{N}) where {N}
    acc = ntuple(_ -> 0.0, Val(N))
    @inbounds for j in 1:g.ny, i in 1:g.nx
        acc = acc .+ ntuple(k -> Float64(g.U[i, j][k]), Val(N))
    end
    return acc .* (Float64(g.dx) * Float64(g.dy))
end
