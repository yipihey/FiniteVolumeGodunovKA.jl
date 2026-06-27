# Packed passive color/species fractions.
#
# Colors are stored out-of-band from the hydro/MHD conserved tuple, following the
# reference kernels' CMA path: the hydro Riemann solve provides one mass flux and
# every color rides that flux as F_color = F_mass * X_upwind.  The stored quantity
# is the intensive fraction X, encoded as UInt16 log2(X) over [2^-110, 1].

const COLOR_LOG2_MIN = -110f0
const COLOR_LOG2_RANGE = 110f0
const COLOR_LOG2_FLOOR = 7.703719777548943f-34 # 2^-110

@inline color_log2_floor() = COLOR_LOG2_MIN
@inline color_fraction_floor() = COLOR_LOG2_FLOOR

@inline function pack_color_log2(l2)
    t = (Float32(l2) - COLOR_LOG2_MIN) * (65535f0 / COLOR_LOG2_RANGE)
    return Base.unsafe_trunc(UInt16, floor(clamp(t, 0f0, 65535f0) + 0.5f0))
end

@inline pack_color_fraction(x) = pack_color_log2(log2(max(Float32(x), COLOR_LOG2_FLOOR)))

@inline unpack_color_log2(u::UInt16) = COLOR_LOG2_MIN + Float32(u) * (COLOR_LOG2_RANGE / 65535f0)
@inline unpack_color_fraction(u::UInt16) = exp2(unpack_color_log2(u))

function _pack_colors(colors, dims::NTuple{D,Int}) where {D}
    colors === nothing && return nothing, nothing
    A = colors
    nd = ndims(A)
    nd == D || nd == D + 1 || error("colors must have $D or $(D + 1) dimensions for grid dims $dims")
    ntuple(i -> size(A, i), Val(D)) == dims ||
        error("colors leading dimensions $(ntuple(i -> size(A, i), Val(D))) do not match grid dims $dims")
    nc = nd == D ? 1 : size(A, D + 1)
    C = Array{UInt16}(undef, dims..., nc)
    if eltype(A) === UInt16
        if nd == D
            @inbounds for I in CartesianIndices(dims)
                C[Tuple(I)..., 1] = A[Tuple(I)...]
            end
        else
            copyto!(C, A)
        end
    else
        if nd == D
            @inbounds for I in CartesianIndices(dims)
                C[Tuple(I)..., 1] = pack_color_fraction(A[Tuple(I)...])
            end
        else
            @inbounds for q in 1:nc, I in CartesianIndices(dims)
                C[Tuple(I)..., q] = pack_color_fraction(A[Tuple(I)..., q])
            end
        end
    end
    return C, similar(C)
end

ncolors(g) = hasproperty(g, :colors) && getproperty(g, :colors) !== nothing ?
    size(getproperty(g, :colors), ndims(getproperty(g, :colors))) : 0

function color_fractions(g)
    hasproperty(g, :colors) || return nothing
    C = getproperty(g, :colors)
    C === nothing && return nothing
    F = Array{Float32}(undef, size(C))
    @inbounds for I in eachindex(C)
        F[I] = unpack_color_fraction(C[I])
    end
    return F
end

@inline function _normal_velocity(s, U, perm)
    W = cons2prim(s, _swap(U, perm))
    return W[2]
end

@inline function _mass_fluxes_dir(s, r, rs, im2, im1, i0, ip1, ip2, lambda, perm)
    sm2 = _swap(im2, perm)
    sm1 = _swap(im1, perm)
    s0 = _swap(i0, perm)
    sp1 = _swap(ip1, perm)
    sp2 = _swap(ip2, perm)
    WRm = _halfstep(s, r, sm2, sm1, s0, lambda)[2]
    WL0, WR0 = _halfstep(s, r, sm1, s0, sp1, lambda)
    WLp = _halfstep(s, r, s0, sp1, sp2, lambda)[1]
    return riemann(rs, s, WRm, WL0)[1], riemann(rs, s, WR0, WLp)[1]
end

@inline function _color_faces(r, lm, l0, lp, un, lambda)
    d = slope(r, lm, l0, lp)
    c = l0 - 0.5f0 * Float32(lambda) * Float32(un) * d
    return c - 0.5f0 * d, c + 0.5f0 * d
end

@inline _color_faces(::PCM, lm, l0, lp, un, lambda) = (l0, l0)

@inline function _update_packed_color(s, r, rs, im2, im1, i0, ip1, ip2,
                                      cm2::UInt16, cm1::UInt16, c0::UInt16,
                                      cp1::UInt16, cp2::UInt16,
                                      lambda, perm, rho_new)
    lm2 = unpack_color_log2(cm2)
    lm1 = unpack_color_log2(cm1)
    l0 = unpack_color_log2(c0)
    lp1 = unpack_color_log2(cp1)
    lp2 = unpack_color_log2(cp2)

    _, rm = _color_faces(r, lm2, lm1, l0, _normal_velocity(s, im1, perm), lambda)
    lz, rz = _color_faces(r, lm1, l0, lp1, _normal_velocity(s, i0, perm), lambda)
    lp, _ = _color_faces(r, l0, lp1, lp2, _normal_velocity(s, ip1, perm), lambda)

    fm, fp = _mass_fluxes_dir(s, r, rs, im2, im1, i0, ip1, ip2, lambda, perm)
    zero_f = zero(fp)
    xl = ifelse(fm >= zero(fm), exp2(rm), exp2(lz))
    xr = ifelse(fp >= zero_f, exp2(rz), exp2(lp))

    rho_x = Float32(i0[1]) * exp2(l0) - Float32(lambda) * (Float32(fp) * xr - Float32(fm) * xl)
    xnew = rho_x / max(Float32(rho_new), eps(Float32))
    return pack_color_fraction(xnew)
end
