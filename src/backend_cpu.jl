# Reference CPU backend (v0): scalar, 1D, single-stage MUSCL-Hancock.
#
# This is the correctness anchor for the contract — deliberately simple (allocates
# per step, no SIMD/threads yet). The SIMD-CPU backend reuses the IDENTICAL physics
# with the element type `T = Vec{W,Float32}` over lane-packed cells; the CUDA backend
# reuses it with `T = Float32` in a staged shared-memory cube. Only this driver loop
# changes per backend; `cons2prim`/`faces`/`riemann` are shared verbatim.

const _NG = 2  # ghost layers (PLM slope stencil + interface reach)

mutable struct Grid1D{N,T,S<:FVSystem,R,RS}
    sys::S
    recon::R
    rsol::RS
    U::Vector{NTuple{N,T}}
    colors::Union{Nothing,Array{UInt16,2}}
    colorst::Union{Nothing,Array{UInt16,2}}
    nx::Int
    dx::T
    bc::Symbol      # :periodic | :outflow
    cfl::T
end

function Grid1D(sys::FVSystem, U0::Vector{NTuple{N,T}};
                dx, bc::Symbol=:outflow, recon=PLM(), rsol=HLLC(), cfl=T(0.4),
                colors=nothing) where {N,T}
    C, Ct = _pack_colors(colors, (length(U0),))
    Grid1D{N,T,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, copy(U0), C, Ct, length(U0), T(dx), bc, T(cfl))
end

primitives(g::Grid1D) = [cons2prim(g.sys, u) for u in g.U]

function conserved_total(g::Grid1D{N,T}) where {N,T}
    acc = g.U[1]
    @inbounds for i in 2:g.nx
        acc = acc .+ g.U[i]
    end
    return acc .* g.dx
end

function max_wavespeed(g::Grid1D)
    s = g.sys
    a = fastspeed_x(s, cons2prim(s, g.U[1]))
    @inbounds for i in 2:g.nx
        a = max(a, fastspeed_x(s, cons2prim(s, g.U[i])))
    end
    return a
end

# Fill a padded primitive array (length nx + 2*_NG) with ghosts per the BC.
function _padded_primitives(g::Grid1D{N,T}) where {N,T}
    nx, s = g.nx, g.sys
    Wp = Vector{NTuple{N,T}}(undef, nx + 2_NG)
    @inbounds for i in 1:nx
        Wp[i + _NG] = cons2prim(s, g.U[i])
    end
    if g.bc === :periodic
        @inbounds for k in 1:_NG
            Wp[k]              = Wp[nx + k]          # left ghosts ← right interior
            Wp[nx + _NG + k]   = Wp[_NG + k]         # right ghosts ← left interior
        end
    elseif g.bc === :outflow
        @inbounds for k in 1:_NG
            Wp[k]              = Wp[_NG + 1]         # zero-gradient
            Wp[nx + _NG + k]   = Wp[nx + _NG]
        end
    else
        error("Grid1D: unknown bc $(g.bc)")
    end
    return Wp
end

function step!(g::Grid1D{N,T}, dt) where {N,T}
    s, dx, nx, ng = g.sys, g.dx, g.nx, _NG
    λ = T(dt) / dx
    Uold = g.U
    Wp = _padded_primitives(g)

    # MUSCL-Hancock predictor: half-step the limited face states by dt/2.
    np  = nx + 2ng
    WLh = Vector{NTuple{N,T}}(undef, np)
    WRh = Vector{NTuple{N,T}}(undef, np)
    @inbounds for j in 2:np-1
        WL, WR = faces(g.recon, Wp[j-1], Wp[j], Wp[j+1])
        FL, FR = physflux_x(s, WL), physflux_x(s, WR)
        dUh = (T(0.5) * λ) .* (FR .- FL)
        WLh[j] = cons2prim(s, prim2cons(s, WL) .- dUh)
        WRh[j] = cons2prim(s, prim2cons(s, WR) .- dUh)
    end

    # Godunov corrector: Riemann flux at each interface, conservative update.
    Unew = Vector{NTuple{N,T}}(undef, nx)
    @inbounds for i in 1:nx
        j  = i + ng
        Fl = riemann(g.rsol, s, WRh[j-1], WLh[j])    # interface i-1/2
        Fr = riemann(g.rsol, s, WRh[j],   WLh[j+1])  # interface i+1/2
        Unew[i] = Uold[i] .- λ .* (Fr .- Fl)
    end
    if g.colors !== nothing
        C = g.colors
        Ct = g.colorst
        bc = Val(g.bc)
        perm = identperm(Val(N))
        @inbounds for q in 1:size(C, 2), i in 1:nx
            Ct[i, q] = _update_packed_color(s, g.recon, g.rsol,
                Uold[_gidx(i - 2, nx, bc)], Uold[_gidx(i - 1, nx, bc)], Uold[i],
                Uold[_gidx(i + 1, nx, bc)], Uold[_gidx(i + 2, nx, bc)],
                C[_gidx(i - 2, nx, bc), q], C[_gidx(i - 1, nx, bc), q], C[i, q],
                C[_gidx(i + 1, nx, bc), q], C[_gidx(i + 2, nx, bc), q],
                λ, perm, Unew[i][1])
        end
        g.colors, g.colorst = g.colorst, g.colors
    end
    if has_source(s)
        @inbounds for i in 1:nx
            Unew[i] = source(s, Unew[i], dt)    # operator-split source
        end
    end
    copyto!(g.U, Unew)
    return g
end

"""
    evolve!(g, tend; maxsteps=10^7) -> g

Advance to `tend` with CFL-limited steps. Returns the grid (state in `g.U`).
"""
function evolve!(g::Grid1D{N,T}, tend; maxsteps::Int = 10^7) where {N,T}
    t = zero(T); tend = T(tend); n = 0
    while t < tend && n < maxsteps
        c = max_wavespeed(g)
        g.sys = prestep(g.sys, c)          # dynamic cleaning speed (no-op unless GLM)
        dt = min(g.cfl * g.dx / c, tend - t)
        step!(g, dt)
        t += dt; n += 1
    end
    return g
end
