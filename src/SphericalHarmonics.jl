module SphericalHarmonics

using IrrationalConstants
using SphericalHarmonicModes
using SphericalHarmonicArrays
using Printf
using StaticArrays
using Setfield
using SpecialFunctions: loggamma

export computeYlm, computeYlm!
export computePlmcostheta, computePlmcostheta!
export computePlmx, computePlmx!

using Base: @propagate_inbounds

include("irrationals.jl")

abstract type HarmonicType end
struct RealHarmonics <: HarmonicType end
struct ComplexHarmonics <: HarmonicType end

"""
    SphericalHarmonicsCache

Preallocate arrays of associated Legendre polynomials and spherical harmonics.
Such an object may be allocated using [`cache`](@ref).
"""
mutable struct SphericalHarmonicsCache{T, C, PLM<:AbstractVector{T}, YLM<:AbstractVector, SHT}
    C :: C
    P :: PLM
    Y :: YLM
    SHType :: SHT
end

struct AssociatedLegendrePolynomials{T, PLM<:AbstractVector{T}} <: AbstractVector{T}
    cosθ :: T
    lmax :: Int
    P :: PLM
    initialized :: Bool
end
AssociatedLegendrePolynomials(cosθ, lmax, P::AbstractVector, initialized::Bool) =
    AssociatedLegendrePolynomials(convert(eltype(P), cosθ), Int(lmax), P, initialized)
Base.parent(P::AssociatedLegendrePolynomials) = P.P
Base.size(P::AssociatedLegendrePolynomials) = size(parent(P))
Base.axes(P::AssociatedLegendrePolynomials) = axes(parent(P))
Base.IndexStyle(::Type{<:AssociatedLegendrePolynomials{<:Any, PLM}}) where {PLM} = IndexStyle(PLM)
@propagate_inbounds Base.getindex(P::AssociatedLegendrePolynomials, I...) = getindex(parent(P), I...)
function Base.summary(io::IO, P::AssociatedLegendrePolynomials)
    print(io, "$(length(P))-element normalized AssociatedLegendrePolynomials{$(eltype(P))} for lmax = $(Int(P.lmax))")
    if P.initialized
        print(io, " and cosθ = ")
        @printf io "%.4g" P.cosθ
    else
        print(io, " (uninitialized)")
    end
end

function SphericalHarmonicsCache(T::Type, lmax::Int, ::Type{m_range}, SHType) where {m_range}
    C = compute_coefficients(T, lmax)
    P = AssociatedLegendrePolynomials(T(0), lmax, allocate_p(T, lmax), false)
    Y = allocate_y(eltypeY(T, SHType), lmax, m_range)
    SphericalHarmonicsCache(C, P, Y, SHType)
end
SphericalHarmonicsCache(lmax::Int, T::Type, SHType) = SphericalHarmonicsCache(Float64, lmax, T, SHType)

function Base.show(io::IO, S::SphericalHarmonicsCache)
    print(io, "$SphericalHarmonicsCache($(eltypeP(S)), $(_lmax(S)), $(_mrange_basetype(S)), $(S.SHType))")
end

"""
    cache([T::Type = Float64], lmax, [m_range = SphericalHarmonicModes.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])

Allocate arrays to evaluate associated Legendre polynomials and spherical harmonics.
The returned object may be passed to [`computePlmcostheta!`](@ref) and [`computeYlm!`](@ref).
The coefficients are cached and need not be recomputed.

# Examples
```jldoctest cache
julia> S = SphericalHarmonics.cache(1);

julia> computePlmcostheta!(S, pi/3, 1)
3-element normalized AssociatedLegendrePolynomials{Float64} for lmax = 1 and cosθ = 0.5:
  0.3989422804014327
  0.34549414947133555
 -0.4231421876608172

julia> computeYlm!(S, pi/3, pi/4, 1)
4-element SHArray(::Vector{ComplexF64}, (ML(0:1, -1:1),)):
   0.2820947917738782 + 0.0im
  0.21157109383040865 - 0.2115710938304086im
  0.24430125595146002 + 0.0im
 -0.21157109383040865 - 0.2115710938304086im
```

Choosing a new `lmax` in `computePlmcostheta!` expands the cache if necessary.

```jldoctest cache
julia> computePlmcostheta!(S, pi/3, 2)
6-element normalized AssociatedLegendrePolynomials{Float64} for lmax = 2 and cosθ = 0.5:
  0.3989422804014327
  0.34549414947133555
 -0.4231421876608172
 -0.11150775725954817
 -0.4730873478787801
  0.40970566147202964
```
"""
cache(args...; m_range = FullRange, SHType = ComplexHarmonics()) = cache(args..., m_range, SHType)
cache(::Type{T}, lmax::Int, ::Type{m_range}, SHType::HarmonicType = ComplexHarmonics()) where {T,m_range} = SphericalHarmonicsCache(T, lmax, m_range, SHType)
cache(::Type{T}, lmax::Int, SHType::HarmonicType = ComplexHarmonics()) where {T} = SphericalHarmonicsCache(T, lmax, FullRange, SHType)
cache(lmax::Int, ::Type{m_range}, SHType::HarmonicType = ComplexHarmonics()) where {m_range} = SphericalHarmonicsCache(lmax, m_range, SHType)
cache(lmax::Int, SHType::HarmonicType = ComplexHarmonics()) = SphericalHarmonicsCache(lmax, FullRange, SHType)

const SHMRange = Union{Type{FullRange}, Type{ZeroTo}}

sizeP(maxDegree::Int) = div((maxDegree + 1) * (maxDegree + 2), 2)
sizeY(maxDegree::Int, ::Type{FullRange} = FullRange) = (maxDegree + 1) * (maxDegree + 1)
sizeY(maxDegree::Int, ::Type{ZeroTo}) = sizeP(maxDegree)

"""
    SphericalHarmonics.allocate_p([T::Type = Float64], lmax::Integer)

Allocate an array large enough to store an entire set of Associated Legendre
Polynomials ``\\bar{P}_ℓ^m(x)`` of maximum degree ``ℓ``.
"""
allocate_p(T::Type, lmax::Integer) = SHArray{T,1}(undef, (ML(ZeroTo(lmax), ZeroTo),))
allocate_p(lmax::Integer) = allocate_p(Float64, lmax)

"""
    SphericalHarmonics.allocate_y([T::Type = ComplexF64], lmax::Integer)

Allocate an array large enough to store an entire set of spherical harmonics
``Y_{ℓ,m}(θ,ϕ)`` of maximum degree ``ℓ``.
"""
allocate_y(T::Type, lmax::Integer, m_range = FullRange) = SHArray{T,1}(undef, (ML(ZeroTo(lmax), m_range),))
allocate_y(lmax::Integer, m_range = FullRange) = allocate_y(ComplexF64, lmax, m_range)

# This method uses the somewhat complicated promote_type mechanism instead of complex(R), as
# some types such as Dual numbers currently do not have complex(R) defined for them.
# We effectively evaluate typeof(zero(R) * im) instead.
eltypeY(::Type{R}, ::ComplexHarmonics) where {R} = promote_type(R, complex(Bool))
eltypeY(::Type{R}, ::RealHarmonics) where {R} = R

# define accessor methods that may be used by wrappers
eltypeP(S::SphericalHarmonicsCache{T}) where {T} = T
eltypeY(S::SphericalHarmonicsCache{<:Any,<:Any,<:Any,Y}) where {Y} = eltype(Y)
getP(S::SphericalHarmonicsCache) = S.P
getY(S::SphericalHarmonicsCache) = S.Y

function _mrange_basetype(S::SphericalHarmonicsCache)
    Y = getY(S)
    modes = first(SphericalHarmonicArrays.shmodes(Y))
    _mrange_basetype(m_range(modes))
end
_mrange_basetype(::FullRange) = FullRange
_mrange_basetype(::ZeroTo) = ZeroTo

function _lmax(S::SphericalHarmonicsCache)
    modes = first(SphericalHarmonicArrays.shmodes(S.C))
    maximum(l_range(modes))
end

@doc raw"""
    SphericalHarmonics.compute_coefficients(lmax)

Precompute coefficients ``a_ℓ^m`` and ``b_ℓ^m`` for all ``2 ≤ ℓ ≤ ℓ_\mathrm{max}`` and ``0 ≤ m ≤ ℓ-2``.

    SphericalHarmonics.compute_coefficients(lmax, m)

Precompute coefficients ``a_ℓ^m`` and ``b_ℓ^m`` for all ``|m| + 2 ≤ ℓ ≤ ℓ_\mathrm{max}`` and the specified ``m``.
"""
compute_coefficients(lmax::Integer) = compute_coefficients(typeof(sqrt(lmax)), lmax)
compute_coefficients(lmax::Integer, ::Nothing) = compute_coefficients(lmax)
compute_coefficients(lmax::Integer, m::Integer) = compute_coefficients(typeof(sqrt(lmax)), lmax, m)
function compute_coefficients(T::Type, lmax::Integer)
    @assert lmax >= 0 "degree must be non-negative"

    shmodes = ML(ZeroTo(Int(lmax)), ZeroTo)
    A = zeros(SVector{2,T}, length(shmodes))
    coeff = SHArray(A, (shmodes,))

    @inbounds _compute_coefficients!(coeff, 2, Int(lmax))
    return coeff
end
compute_coefficients(T::Type, lmax::Integer, ::Nothing) = compute_coefficients(T, lmax)
function compute_coefficients(T::Type, lmax::Integer, m::Integer)
    @assert lmax >= 0 "degree must be non-negative"
    @assert m >= 0 "m must be non-negative"

    shmodes = ML(ZeroTo(Int(lmax)), SingleValuedRange(m))
    coeff = zeros(SVector{2,T}, shmodes)

    @inbounds _compute_coefficients!(coeff, abs(Int(m)) + 2, Int(lmax), abs(Int(m)), abs(Int(m)))
    return coeff
end
function compute_coefficients!(S::SphericalHarmonicsCache, lmax::Integer)
    @assert lmax >= 0 "degree must be >= 0"
    if lmax > _lmax(S)
        shmodesP = ML(ZeroTo(Int(lmax)), ZeroTo)
        shmodesY = ML(ZeroTo(Int(lmax)), _mrange_basetype(S))
        A = parent(S.C)
        resize!(A, length(shmodesP))
        coeff = SHArray(A, (shmodesP,))

        @inbounds _compute_coefficients!(coeff, Int(_lmax(S))+1, Int(lmax))

        S.C = coeff
        P_len = length(S.P)
        P_new = resize!(parent(parent(S.P)), length(shmodesP))
        P_new[P_len+1:end] .= zero(eltype(P_new))
        P_new_SA = SHArray(P_new, (shmodesP,))
        P = S.P
        P = @set P.P = P_new_SA
        S.P = P

        Y_len = length(S.Y)
        Y_new = resize!(parent(S.Y), length(shmodesY))
        Y_new[Y_len+1:end] .= zero(eltype(Y_new))
        S.Y = SHArray(Y_new, (shmodesY,))
    end
    return S.C
end

@propagate_inbounds function _compute_coefficients!(coeff, lmin, lmax, m_min = 0, m_max = lmax - 2)
    T = eltype(eltype(coeff))
    for l in lmin:lmax
        # pre-compute certain terms to improve performance
        invlm1 = 1/T(l-1)
        invl = 1/T(l)
        Anum = 4 - invl^2
        Bden = 1/√(4 - invlm1^2)
        for m in max(0, m_min):min(l-2, m_max)
            coeff[(l, m)] = SVector{2}(√(Anum / (1 - (m * invl)^2)), -√(1 - (m * invlm1)^2) * Bden)
        end
    end
end

function checksize(sz, minsize)
    @assert sz >= minsize "array needs to have a minimum size of $minsize, received size $sz"
end
function checksizesP(P, lmax)
    lmax >= 0 || throw(ArgumentError("lmax = $lmax does not correspond to a valid mode"))
    checksize(length(P), sizeP(Int(lmax)))
end

@propagate_inbounds readcoeffs(coeff::SHArray, T::Type, l, m) = coeff[(l,m)]

function readcoeffs(coeff::Nothing, T::Type, l, m)
    @assert abs(m) <= l - 2 "m must be <= l - 2"
    invl = 1/T(l)
    invlm1 = 1/T(l-1)
    alm = √(T(4 - invl^2)/T(1 - (m * invl)^2))
    blm = -√(T(1 - (m * invlm1)^2 )/T(4 - invlm1^2))
    return alm, blm
end

######################################################################################
# Associated Legendre polynomials
######################################################################################

_promotetype(costheta, sintheta, coeff::Nothing) = float(promote_type(typeof(costheta), typeof(sintheta)))
_promotetype(costheta, sintheta, coeff::AbstractArray) = float(reduce(promote_type, (typeof(costheta), typeof(sintheta), eltype(eltype(coeff)))))

_viewmodes(P, modes) = @view P[(firstindex(P) - 1) .+ (1:length(modes))]

function _wrapSHArray(P, modes)
    SHArray(_viewmodes(P, modes), (modes,))
end
function _wrapSHArray(P, lmax, m_range)
    modes = ML(ZeroTo(Int(lmax)), m_range)
    _wrapSHArray(P, modes)
end

abstract type PLMnorm end
struct Orthonormal <: PLMnorm cs::Bool end
struct Unnormalized <: PLMnorm cs::Bool end
struct LMnorm <: PLMnorm cs::Bool end # Limpanuparb-Milthorpe norm as used in the paper

(::Type{T})(; csphase = true) where {T<:PLMnorm} = T(csphase)

# Condon-Shortey phase is included by default. This may be disabled by passing
# the flag to the normalization specifier
csphase(norm, m) = !norm.cs ? (-1)^m : 1
function invnorm(l, m, norm::Unnormalized)
    f = exp(-(log(2l+1) - log2π + loggamma(l-m+1) - loggamma(l+m+1))/2)
    f * csphase(norm, m)
end
invnorm(l, m, norm::Orthonormal) = (f = sqrtπ; f * csphase(norm, m))
invnorm(l, m, norm::LMnorm) = (f = true; f * csphase(norm, m))

m_l(modes, l, ::Nothing) =  m_range(modes, l)
m_l(modes, l, m) =  intersect(m_range(modes, l), m)
function normalize!(P, norm, l, m = nothing)
    norm isa LMnorm && norm.cs && return P # short-circuit the standard case
    modes = LM(l, m === nothing ? ZeroTo : m)
    Plm = _wrapSHArray(P, maximum(l), ZeroTo)
    T = eltype(Plm)
    for l in l_range(modes)
        for m in m_l(modes, l, m)
            Plm[(l,m)] *= invnorm(T(l), T(m), norm)
        end
    end
    return P
end

@propagate_inbounds function _computePlmcostheta!(P, costheta, sintheta, l_range,
    coeff::Union{AbstractArray,Nothing} = nothing; norm::PLMnorm = LMnorm())

    lmin, lmax = map(Int, extrema(l_range))
    @assert lmin >= 0 "degree must be non-negative"

    T = _promotetype(costheta, sintheta, coeff)
    Plm = _wrapSHArray(P, lmax, ZeroTo)

    if lmin == 0
        Plm[(0, 0)] = invsqrt2π
    end

    if (lmax > 0)
        P11 = -(_sqrt3by4pi * sintheta)
        temp = T(P11)
        if lmin <= 1
            Plm[(1, 0)] = _sqrt3by2pi * costheta
            Plm[(1, 1)] = P11
        end

        # compute temp without losing precision
        for l in 2:lmin - 1
            temp *= -√(T(2l + 1)/2l) * sintheta
        end

        for l in max(2, lmin):lmax
            for m in 0:(l-2)
                alm, blm = readcoeffs(coeff, T, l, m)
                Plm[(l,m)] = alm * (costheta * Plm[(l - 1, m)] + blm * Plm[(l - 2, m)])
            end
            Plm[(l, l - 1)] = costheta * √(T(2l + 1)) * temp
            temp *= -√(T(2l + 1)/2l) * sintheta
            Plm[(l, l)] = temp
        end
    end

    normalize!(P, norm, l_range)
    return P
end

@propagate_inbounds function _computePlmcostheta!(P, costheta, sintheta, l_range, m::Integer,
    coeff::Union{AbstractArray,Nothing} = nothing; norm::PLMnorm = LMnorm())

    lmin, lmax = map(Int, extrema(l_range))
    @assert lmin == 0 "minimum degree must be zero"
    # @assert lmin >= 0 "degree must be non-negative"
    0 <= m <= lmax || throw(ArgumentError("m = $m does not satisfy 0 ≤ m ≤ lmax = $lmax"))

    T = _promotetype(costheta, sintheta, coeff)

    Plm = _wrapSHArray(P, lmax, ZeroTo)

    if m == 0 && lmin == 0
        Plm[(0, 0)] = invsqrt2π
    end
    temp = T(invsqrt2π)

    if (lmax > 0)
        P11 = -(_sqrt3by4pi * sintheta)
        if lmin <= 1 && m > 0
            temp = T(P11)
            Plm[(1, 1)] = temp
        end

        # This is necessary for lmin > 2
        # for l in 2:lmin - 1
        #     temp *= -√(T(2l + 1)/2l) * sintheta
        # end

        # Compute Pmm using recursion over l and m
        for mi in max(lmin, 2):Int(m)
            temp *= -√(T(2mi + 1)/2mi) * sintheta
            Plm[(mi, mi)] = temp
        end

        if m == lmax
            normalize!(Plm, norm, l_range, m)
            return P
        end

        Plm[(m + 1, m)] = √(T(2m + 3)) * costheta * temp

        # Compute Plm using recursion over l at a fixed m
        for l in max(lmin, Int(m) + 2):lmax
            alm, blm = readcoeffs(coeff, T, l, m)
            Plm[(l,m)] = alm * (costheta * Plm[(l - 1, m)] + blm * Plm[(l - 2, m)])
        end
    end

    normalize!(Plm, norm, l_range, m)
    return P
end

@propagate_inbounds function _computePlmcostheta!(P, costheta, sintheta, l_range,
        m::Nothing, coeff::Union{AbstractArray,Nothing}; norm::PLMnorm = LMnorm())
    _computePlmcostheta!(P, costheta, sintheta, l_range, coeff; norm = norm)
end

@propagate_inbounds function Plmrecursion(l, m, costheta, Pmm, Pmp1m, coeff, T)
    Plm2_m, Plm1_m = Pmm, Pmp1m

    # Compute Plm using recursion over l at a fixed m, starting from l = m + 2
    for li in Int(m) + 2:Int(l)
        alm, blm = readcoeffs(coeff, T, li, m)

        Pl_m = alm * (costheta * Plm1_m + blm * Plm2_m)

        Plm2_m, Plm1_m = Plm1_m, Pl_m
    end

    return Plm1_m
end

@propagate_inbounds function _computePlmcostheta(costheta, sintheta, l, m, coeff; norm::PLMnorm = LMnorm())
    0 <= m <= l || throw(ArgumentError("m = $m does not satisfy 0 ≤ m ≤ l = $l"))

    T = _promotetype(costheta, sintheta, coeff)

    Pmm = T(invsqrt2π)

    if m > 0
        Pmm = -T(_sqrt3by4pi * sintheta)
    end

    # Compute Pmm using recursion over l and m
    for mi in 2:Int(m)
        Pmm = -√(1 + 1 / 2T(mi)) * sintheta * Pmm
    end

    if m == l
        Pmm *= invnorm(T(l), T(m), norm)
        return Pmm
    end

    Pmp1m = √(T(2m + 3)) * costheta * Pmm

    # Recursion at a constant m to compute Pl,m from Pm,m and Pm+1,m
    Plm = Plmrecursion(l, m, costheta, Pmm, Pmp1m, coeff, T)
    Plm *= invnorm(T(l), T(m), norm)
    return Plm
end

function zeroP!(P, lmax)
    checksizesP(P, lmax)
    N = sizeP(Int(lmax))
    Pv = @view P[firstindex(P) .+ (0:N-1)]
    fill!(Pv, zero(eltype(Pv)))
    return P
end

"""
    computePlmx!(P::AbstractVector, x, lmax::Integer, coeff::AbstractMatrix; [norm = SphericalHarmonics.LMnorm()])

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_ℓ^m(x)``
using the given coefficients, and store in the array `P`.
The matrix `coeff` may be computed using [`compute_coefficients`](@ref).

The argument `x` needs to lie in ``-1 ≤ x ≤ 1``. The function implicitly assumes that
``x = \\cos\\theta`` where ``0 ≤ \\theta ≤ π``.

The keyword argument `norm` may be used to specify the how the polynomials are normalized.
See [`computePlmcostheta`](@ref) for the possible normalization options.

    computePlmx!(P::AbstractVector, x, lmax::Integer, m::Integer, coeff::AbstractMatrix)

Compute the set of normalized Associated Legendre Polynomials ``\\bar{P}_ℓ^m(x)`` for for the specified ``m``
and all ``ℓ`` lying in ``|m| ≤ ℓ ≤ ℓ_\\mathrm{max}`` .
"""
function computePlmx!(P::AbstractVector, x, lmax::Integer, args...; norm::PLMnorm = LMnorm())
    @assert lmax >= 0 "degree must be non-negative"
    -1 <= x <= 1 || throw(DomainError("x", "The argument to associated Legendre polynomials must satisfy -1 <= x <= 1"))
    zeroP!(P, lmax)
    _computePlmx_range!(P, x, 0:lmax, args...; norm = norm)
    return P
end

"""
    computePlmx!(S::SphericalHarmonicsCache, x[, lmax::Integer]; [norm = SphericalHarmonics.LMnorm()])

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_ℓ^m(x)``
using the pre-computed coefficients in `S`, and store the result in `S`. If `lmax` is not provided,
the value of `lmax` for which coefficients have been computed in `S` is used.

The keyword argument `norm` may be used to specify the how the polynomials are normalized.
See [`computePlmcostheta`](@ref) for the possible normalization options.
"""
function computePlmx!(S::SphericalHarmonicsCache, x, lmax::Integer = _lmax(S); norm::PLMnorm = LMnorm())
    compute_coefficients!(S, lmax)
    P = computePlmx!(S.P, x, lmax, S.C, norm = norm)
    S.P = P
    return S.P
end
function computePlmx!(P::AssociatedLegendrePolynomials, x, lmax::Integer, coeff = nothing; kw...)
    @assert lmax >= 0 "degree must be non-negative"
    -1 <= x <= 1 || throw(DomainError("x", "The argument to associated Legendre polynomials must satisfy -1 <= x <= 1"))
    if P.initialized && P.cosθ == x && P.lmax >= lmax
        return P
    elseif P.initialized && P.cosθ == x && P.lmax < lmax
        _computePlmx_range!(parent(P), x, P.lmax+1:lmax, coeff; kw...)
    else
        computePlmx!(parent(P), x, lmax, coeff; kw...)
        P = @set P.cosθ = x
    end
    P = @set P.lmax = lmax
    P = @set P.initialized = true
    return P
end
function _computePlmx_range!(P::AbstractVector, x, l_range::AbstractUnitRange{<:Integer}, args...; norm::PLMnorm = LMnorm())
    lmin, lmax = map(Int, extrema(l_range))
    cosθ, sinθ  = promote(x, √(1-x^2))
    @inbounds _computePlmcostheta!(P, cosθ, sinθ, lmin:lmax, args...; norm = norm)
    return P
end

"""
    computePlmx(x; lmax::Integer, [m::Integer], [norm = SphericalHarmonics.LMnorm()])
    computePlmx(x, lmax::Integer[, m::Integer]; [norm = SphericalHarmonics.LMnorm()])

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_ℓ^m(x)`` where
``0 ≤ ℓ ≤ ℓ_\\mathrm{max}`` and ``0 ≤ m ≤ ℓ``. If `m` is provided then only the polynomials for that
azimuthal order are computed.

The argument `x` needs to lie in ``-1 ≤ x ≤ 1``. The function implicitly assumes that
``x = \\cos\\theta`` where ``0 ≤ \\theta ≤ π``.

The keyword argument `norm` may be used to specify the how the polynomials are normalized.
See [`computePlmcostheta`](@ref) for the possible normalization options.
"""
computePlmx(x; lmax::Integer, m::Union{Integer, Nothing} = nothing, norm::PLMnorm = LMnorm()) = computePlmx(x, lmax, m; norm = norm)
function computePlmx(x, lmax::Integer, m::Union{Integer, Nothing} = nothing; norm::PLMnorm = LMnorm())
    P = allocate_p(float(typeof(x)), lmax)
    coeff = compute_coefficients(lmax, m)
    _applymaybeallm(computePlmx!, P, x, lmax, m, coeff; norm = norm)
    return AssociatedLegendrePolynomials(x, lmax, P, true)
end

"""
    computePlmcostheta!(P::AbstractVector, θ, lmax::Integer, coeff; [norm = SphericalHarmonics.LMnorm()])
    computePlmcostheta!(P::AbstractVector, θ::SphericalHarmonics.Pole, lmax::Integer; [norm = SphericalHarmonics.LMnorm()])

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_ℓ^m(\\cos θ)``
using the given coefficients, and store in the array `P`. The matrix `coeff` may be computed
using [`compute_coefficients`](@ref).

The keyword argument `norm` may be used to specify the how the polynomials are normalized.
See [`computePlmcostheta`](@ref) for the possible normalization options.

    computePlmcostheta!(P::AbstractVector, θ, lmax::Integer, m::Integer, coeff)

Compute the Associated Legendre Polynomials ``\\bar{P}_ℓ^m(\\cos θ)`` for for the specified ``m``
and all ``ℓ`` lying in ``|m| ≤ ℓ ≤ ℓ_\\mathrm{max}``. The array `P` needs to be large enough to hold all the polynomials
for ``0 ≤ ℓ ≤ ℓ_\\mathrm{max}`` and ``0 ≤ m ≤ ℓ``, as the recursive evaluation requires the computation of extra elements.
Pre-existing values in `P` may be overwritten, even for azimuthal orders not equal to ``m``.
"""
function computePlmcostheta!(P::AbstractVector, θ, lmax, args...; norm::PLMnorm = LMnorm())
    zeroP!(P, lmax)
    _computePlmcostheta_range!(P, θ, 0:lmax, args...; norm = norm)
    return P
end
function _computePlmcostheta_range!(P::AbstractVector, θ, l_range::AbstractUnitRange{<:Integer}, args...; norm::PLMnorm = LMnorm())
    lmin, lmax = map(Int, extrema(l_range))
    cosθ, sinθ  = promote(cos(θ), sin(θ))
    @inbounds _computePlmcostheta!(P, cosθ, sinθ, lmin:lmax, args...; norm = norm)
    return P
end

function computePlmcostheta!(P::AbstractVector, θ::Pole, lmax, args...; norm::PLMnorm = LMnorm())
    zeroP!(P, lmax)
    _computePlmcostheta_range!(P, θ, 0:lmax, args...; norm = norm)
    return P
end
function _computePlmcostheta_range!(P::AbstractVector, θ::Pole, l_range::AbstractUnitRange{<:Integer}, m::Integer, args...; kw...)
    lmin, lmax = map(Int, extrema(l_range))
    if !iszero(m)
        return P
    end
    _computePlmcostheta_m0_range!(P, θ, lmin:lmax; kw...)
    return P
end
function _computePlmcostheta_range!(P::AbstractVector, θ::Pole, l_range::AbstractUnitRange{<:Integer}, args...; kw...)
    _computePlmcostheta_m0_range!(P, θ, l_range; kw...)
    return P
end

function _computePlmcostheta_m0_range!(P::AbstractVector{T}, ::NorthPole, l_range::AbstractUnitRange{<:Integer}; norm::PLMnorm = LMnorm()) where {T}
    lmin, lmax = map(Int, extrema(l_range))
    Plm = _wrapSHArray(P, lmax, ZeroTo)
    @inbounds for l in lmin:lmax
        Plm[(l, 0)] = invsqrt2π * √(T(2l + 1))
    end
    normalize!(Plm, norm, l_range, 0)
    return P
end

function _computePlmcostheta_m0_range!(P::AbstractVector{T}, ::SouthPole, l_range::AbstractUnitRange{<:Integer}; norm::PLMnorm = LMnorm()) where {T}
    lmin, lmax = map(Int, extrema(l_range))
    Plm = _wrapSHArray(P, lmax, ZeroTo)
    phase = 1
    @inbounds for l in lmin:lmax
        Plm[(l, 0)] = phase * invsqrt2π * √(T(2l + 1))
        phase *= -1
    end
    normalize!(Plm, norm, l_range, 0)
    return P
end

function computePlmcostheta!(P::AssociatedLegendrePolynomials, θ::Pole, lmax, args...; kw...)
    P = _computePlmcostheta_alp!(P, θ, lmax, args...; kw...)
    return P
end
function computePlmcostheta!(P::AssociatedLegendrePolynomials, θ, lmax, args...; kw...)
    P = _computePlmcostheta_alp!(P, θ, lmax, args...; kw...)
    return P
end
function _computePlmcostheta_alp!(P::AssociatedLegendrePolynomials, θ, lmax, args...; kw...)
    if P.initialized && P.cosθ == cos(θ) && P.lmax >= lmax
        return P
    elseif P.initialized && P.cosθ == cos(θ) && P.lmax < lmax
        _computePlmcostheta_range!(parent(P), θ, P.lmax+1:lmax, args...; kw...)
    else
        computePlmcostheta!(parent(P), θ, lmax, args...; kw...)
        P = @set P.cosθ = cos(θ)
    end
    P = @set P.lmax = lmax
    P = @set P.initialized = true
    return P
end

"""
    computePlmcostheta!(S::SphericalHarmonicsCache, θ, [lmax::Integer]; [norm = SphericalHarmonics.LMnorm()])

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_ℓ^m(\\cos θ)``
using the pre-computed coefficients in `S`, and store the result in `S`. If `lmax` is not provided,
the value of `lmax` for which coefficients have been computed in `S` is used.

The keyword argument `norm` may be used to specify the how the polynomials are normalized.
See [`computePlmcostheta`](@ref) for the possible normalization options.
"""
function computePlmcostheta!(S::SphericalHarmonicsCache, θ, lmax::Integer = _lmax(S); kw...)
    compute_coefficients!(S, lmax)
    P = computePlmcostheta!(S.P, θ, lmax, S.C; kw...)
    S.P = P
    return P
end

"""
    computePlmcostheta(θ; lmax::Integer, [m::Integer], [norm = SphericalHarmonics.LMnorm()])
    computePlmcostheta(θ, lmax::Integer[, m::Integer]; [norm = SphericalHarmonics.LMnorm()])

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_ℓ^m(\\cos θ)`` where
``0 ≤ ℓ ≤ ℓ_\\mathrm{max}`` and ``0 ≤ m ≤ ℓ`` for colatitude ``\\theta``. If `m` is provided then only the
polynomials corresponding to the specified `m` are computed.

The polynomials are normalized as

```math
\\bar{P}_{\\ell}^m = \\sqrt{\\frac{(2\\ell + 1)(\\ell-m)!}{2\\pi (\\ell+m)!}} P_{\\ell m},
```

where ``P_{\\ell m}`` are the standard
[associated Legendre polynomials](https://en.wikipedia.org/wiki/Associated_Legendre_polynomials#Alternative_notations),
and are defined in terms of Legendre polynomials ``P_\\ell(x)`` as

```math
P_{\\ell m}\\left(x\\right)=(-1)^m \\left(1-x^{2}\\right)^{m/2}\\frac{d^{m}}{dx^{m}}P_{\\ell}\\left(x\\right).
```

The normalized polynomials ``\\bar{P}_{\\ell}^m`` satisfy

```math
\\int_{-1}^{1} dx\\,\\left| \\bar{P}_{\\ell}^m(x) \\right|^2 = \\frac{1}{\\pi}
```

A different normalization may be chosen by specifying the keyword argument `norm`, with optionally the
Condon-Shortley phase disabled by passing the keyword argument `csphase` to the constructor of the
normalization specifier. The possible normalization options are:

* `SphericalHarmonics.LMnorm([; csphase = true])`: the default normalization described above
* `SphericalHarmonics.Orthonormal([; csphase = true])`: Orthonormal polynomials that are defined as
```math
\\tilde{P}_{\\ell}^m = \\sqrt{\\frac{(2\\ell + 1)(\\ell-m)!}{2(\\ell+m)!}} P_{\\ell m} =
\\sqrt{\\pi} \\bar{P}_{\\ell m},
```
and satisfy
```math
\\int_{-1}^{1} \\tilde{P}_{\\ell m}(x) \\tilde{P}_{k m}(x) dx = \\delta_{ℓk}
```
* `SphericalHarmonics.Unnormalized([; csphase = true])`: The polynomials ``P_{ℓm}`` that satisfy ``P_{ℓm}(1)=\\delta_{m0}``
within numerical precision bounds, as well as

```math
\\int_{-1}^{1} P_{\\ell m}(x) P_{k m}(x) dx = \\frac{2(\\ell+m)!}{(2\\ell+1)(\\ell-m)!}\\delta_{ℓk}
```

In each case setting `csphase = false` will lead to an overall factor of ``(-1)^m`` being multiplied
to the polynomials.

Returns an `AbstractVector` that may be indexed using `(ℓ,m)` pairs aside from the
canonical indexing with `Int`s.

The precision of the result may be increased by using arbitrary-precision arguments.

# Examples
```jldoctest
julia> P = computePlmcostheta(pi/2, lmax = 1)
3-element normalized AssociatedLegendrePolynomials{Float64} for lmax = 1 and cosθ = 6.123e-17:
  0.3989422804014327
  4.231083042742082e-17
 -0.4886025119029199

julia> P[(0,0)]
0.3989422804014327

julia> P = computePlmcostheta(big(pi)/2, lmax = 1)
3-element normalized AssociatedLegendrePolynomials{BigFloat} for lmax = 1 and cosθ = 5.485e-78:
  0.3989422804014326779399460599343818684758586311649346576659258296706579258993008
  3.789785583114350800838137317730900078444216599640987847808409161681770236721676e-78
 -0.4886025119029199215863846228383470045758856081942277021382431574458410003616367
```
"""
computePlmcostheta(θ; lmax::Integer, m::Union{Integer, Nothing} = nothing, norm::PLMnorm = LMnorm()) = computePlmcostheta(θ, lmax, m; norm = norm)

_applymaybeallm(f, P, θ, lmax, m::Nothing, coeff...; kw...) = f(P, θ, lmax, coeff...; kw...)
_applymaybeallm(f, P, θ, lmax, m, coeff...; kw...) = f(P, θ, lmax, m, coeff...; kw...)

_maybecomputecoeff(lmax, θ::Pole, m) = nothing
_maybecomputecoeff(lmax, θ, m) = compute_coefficients(promote_type(Float64, float(typeof(θ))), Int(lmax), m)

function computePlmcostheta(θ, lmax::Integer, m::Union{Integer, Nothing} = nothing; norm::PLMnorm = LMnorm())
    coeff = _maybecomputecoeff(lmax, θ, m)
    P = allocate_p(float(typeof(θ)), lmax)
    _applymaybeallm(computePlmcostheta!, P, θ, lmax, m, coeff; norm = norm)
    return AssociatedLegendrePolynomials(cos(θ), lmax, P, true)
end

@doc raw"""
    SphericalHarmonics.associatedLegendrex(x; l::Integer, m::Integer, [coeff = nothing], [norm = SphericalHarmonics.LMnorm()])

Evaluate the normalized associated Legendre polynomial ``\bar{P}_ℓ^m(x)``.
Optionally a matrix of coefficients returned by [`compute_coefficients`](@ref) may be provided.

The keyword argument `norm` may be used to specify the how the polynomials are normalized.
See [`computePlmcostheta`](@ref) for the possible normalization options.
"""
associatedLegendrex(x; l::Integer, m::Integer, coeff = nothing, norm::PLMnorm = LMnorm()) =
    associatedLegendrex(x, l, m, coeff; norm = norm)
function associatedLegendrex(x, l::Integer, m::Integer, coeff = nothing; norm::PLMnorm = LMnorm())
    _computePlmcostheta(x, √(1-x^2), l, m, coeff; norm = norm)
end

@doc raw"""
    SphericalHarmonics.associatedLegendre(θ; l::Integer, m::Integer, [coeff = nothing], [norm = SphericalHarmonics.LMnorm()])

Evaluate the normalized associated Legendre polynomial ``\bar{P}_ℓ^m(\cos \theta)``.
Optionally a matrix of coefficients returned by [`compute_coefficients`](@ref) may be provided.

The keyword argument `norm` may be used to specify the how the polynomials are normalized.
See [`computePlmcostheta`](@ref) for the possible normalization options.
"""
associatedLegendre(θ; l::Integer, m::Integer, coeff = nothing, norm::PLMnorm = LMnorm()) = associatedLegendre(θ, l, m, coeff; norm = norm)
function associatedLegendre(θ, l::Integer, m::Integer, coeff = nothing; norm::PLMnorm = LMnorm())
    _computePlmcostheta(cos(θ), sin(θ), l, m, coeff; norm = norm)
end

_Plm_pole(T, l, m, norm) = invsqrt2π * √(T(2l + 1)) * invnorm(T(l), T(m), norm)

function associatedLegendre(::NorthPole, l::Integer, m::Integer, coeff = nothing; norm::PLMnorm = LMnorm())
    T = promote_type(Float64, float(typeof(l)))
    if m != 0
        return zero(T)
    end
    _Plm_pole(T, l, m, norm)
end

function associatedLegendre(::SouthPole, l::Integer, m::Integer, coeff = nothing; norm::PLMnorm = LMnorm())
    T = promote_type(Float64, float(typeof(l)))
    if m != 0
        return zero(T)
    end
    (-1)^Int(l) * _Plm_pole(T, l, m, norm)
end

######################################################################################
# Spherical Harmonics
######################################################################################

function phase(::RealHarmonics, ::Type{FullRange}, m, ϕ, norm, CSphase)
    S, C = sincos(abs(Int(m))*ϕ)
    C, S
end

function phase(::ComplexHarmonics, ::Type{FullRange}, m, ϕ, norm, CSphase)
    ep = cis(Int(m)*ϕ) * norm
    em = CSphase * conj(ep)
    ep, em
end

phase(::RealHarmonics, ::Type{ZeroTo}, m, ϕ, norm, CSphase) = cos(m*ϕ)
phase(::ComplexHarmonics, ::Type{ZeroTo}, m, ϕ, norm, CSphase) = cis(m*ϕ) * norm

@propagate_inbounds function fill_m_maybenegm!(Y, P, lmax, m, ϕ, CSphase, ::Type{FullRange}, SHType)
    m >= 0 || throw(ArgumentError("m must be ≥ 0"))
    @assert lmax >= 0 "degree must be non-negative"
    phasempos, phasemneg = phase(SHType, FullRange, m, ϕ, invsqrt2, CSphase)

    PS = _wrapSHArray(P, lmax, ZeroTo)
    Ylm = _wrapSHArray(Y, lmax, FullRange)

    for l in m:Int(lmax)
        Plm = PS[(l, m)]
        Ylm[(l, -m)] = phasemneg * Plm
        Ylm[(l,  m)] = phasempos * Plm
    end
    return Y
end

@propagate_inbounds function fill_m_maybenegm!(Y, P, lmax, m, ϕ, CSphase, ::Type{ZeroTo}, SHType)
    m >= 0 || throw(ArgumentError("m must be ≥ 0"))
    @assert lmax >= 0 "degree must be non-negative"
    phasem = phase(SHType, ZeroTo, m, ϕ, invsqrt2, CSphase)

    PS = _wrapSHArray(P, lmax, ZeroTo)
    Ylm = _wrapSHArray(Y, lmax, ZeroTo)

    for l in abs(Int(m)):Int(lmax)
        Ylm[(l,m)] = phasem * PS[(l,m)]
    end
    return Y
end

@propagate_inbounds function fill_m!(Y, P, lmax, m, ϕ, CSphase, ::Type{FullRange}, SHType::ComplexHarmonics)
    @assert lmax >= 0 "degree must be non-negative"
    phasempos, phasemneg = phase(SHType, FullRange, m, ϕ, invsqrt2, CSphase)

    PS = _wrapSHArray(P, lmax, ZeroTo)
    Ylm = _wrapSHArray(Y, lmax, FullRange)

    for l in abs(Int(m)):Int(lmax)
        Plm = PS[(l, abs(m))]
        if m >= 0
            Ylm[(l, m)] = phasempos * Plm
        else
            Ylm[(l, m)] = (-1)^m * phasempos * Plm
        end
    end
    return Y
end

@propagate_inbounds function fill_m!(Y, P, lmax, m, ϕ, CSphase, ::Type{FullRange}, SHType::RealHarmonics)
    @assert lmax >= 0 "degree must be non-negative"
    C, S = phase(SHType, FullRange, m, ϕ, invsqrt2, CSphase)

    PS = _wrapSHArray(P, lmax, ZeroTo)
    Ylm = _wrapSHArray(Y, lmax, FullRange)

    for l in abs(Int(m)):Int(lmax)
        Plm = PS[(l, abs(m))]
        if m >= 0
            Ylm[(l, m)] = C * Plm
        else
            Ylm[(l, m)] = S * Plm
        end
    end
    return Y
end

@propagate_inbounds function fill_m!(Y, P, lmax, m, ϕ, CSphase, ::Type{ZeroTo}, SHType)
    m >= 0 || throw(ArgumentError("m must be ≥ 0"))
    @assert lmax >= 0 "degree must be non-negative"

    phasem = phase(SHType, ZeroTo, m, ϕ, invsqrt2, CSphase)
    PS = _wrapSHArray(P, lmax, ZeroTo)
    Ylm = _wrapSHArray(Y, lmax, ZeroTo)

    for l in Int(m):Int(lmax)
        Ylm[(l, m)] = phasem * P[(l, m)]
    end
    return Y
end

@propagate_inbounds function computeYlm_maybeallm!(Y, P, ϕ, lmax, ::Nothing, m_range, SHType)
    @assert lmax >= 0 "degree must be non-negative"
    PS = _wrapSHArray(P, lmax, ZeroTo)
    Ylm = _wrapSHArray(Y, lmax, m_range)
    for l in 0:Int(lmax)
        Ylm[(l, 0)] = PS[(l, 0)] * invsqrt2
    end

    CSphase = 1
    for m in 1:Int(lmax)
        CSphase *= -1
        fill_m_maybenegm!(Y, P, lmax, m, ϕ, CSphase, m_range, SHType)
    end
    return Y
end
@propagate_inbounds function computeYlm_maybeallm!(Y, P, ϕ, lmax, m::Integer, m_range, SHType)
    @assert lmax >= 0 "degree must be non-negative"
    -lmax <= m <= lmax || throw(ArgumentError("m = $m does not satisfy 0 ≤ |m| ≤ lmax = $lmax"))

    PS = _wrapSHArray(P, lmax, ZeroTo)
    Ylm = _wrapSHArray(Y, lmax, m_range)
    if iszero(m)
        for l in 0:Int(lmax)
            Ylm[(l, 0)] = PS[(l, 0)] * invsqrt2
        end
    else
        fill_m!(Y, P, lmax, m, ϕ, (-1)^m, m_range, SHType)
    end
    return Y
end

_maybeabs(::Nothing) = nothing
_maybeabs(m::Integer) = abs(m)

function checksize_zeroY!(Y, P, lmax, m_range)
    checksize(length(P), sizeP(Int(lmax)))
    Ny = sizeY(Int(lmax), m_range)
    checksize(length(Y), Ny)
    Yv = @view Y[firstindex(Y) .+ (0:Ny-1)]
    fill!(Yv, zero(eltype(Yv)))
    return nothing
end

"""
    computeYlm!(Y::AbstractVector, P::AbstractVector, θ, ϕ; lmax::Integer, [m::Integer], [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])
    computeYlm!(Y::AbstractVector, P::AbstractVector, θ, ϕ, lmax::Integer, [m::Integer, [m_range = SphericalHarmonics.FullRange, [SHType = SphericalHarmonics.ComplexHarmonics()]]])

Compute an entire set of spherical harmonics ``Y_{ℓ,m}(θ,ϕ)``
using the precomputed associated Legendre Polynomials ``\\bar{P}_ℓ^m(\\cos θ)``,
and store in the array `Y`. The array `P` may be computed using [`computePlmcostheta`](@ref).

The optional argument `m_range` decides if the spherical harmonics for negative `m` values are computed.
By default the functions for all values of `m` are evaluated. Setting `m_range` to `SphericalHarmonics.ZeroTo` would result
in only functions for `m ≥ 0` being evaluated. Providing `m` would override this, in which case only the polynomials
corresponding to the azimuthal order `m` would be computed.

The optional argument `SHType` may be used to choose between real and complex harmonics.
To compute real spherical harmonics, set this to `RealHarmonics()`.

!!! note
    This function assumes that the associated Legendre Polynomials have been pre-computed, and does not perform any
    check on the values of `P`.
"""
function computeYlm!(Y::AbstractVector, P::AbstractVector, θ,
    ϕ, lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())

    checksize_zeroY!(Y, P, lmax, m_range)
    @inbounds computeYlm_maybeallm!(Y, P, ϕ, lmax, m, m_range, SHType)
    return Y
end
function computeYlm!(Y::AbstractVector, P::AbstractVector, θ,
    ϕ, lmax::Integer, m_range::SHMRange, SHType::HarmonicType = ComplexHarmonics())

    computeYlm!(Y, P, θ, ϕ, lmax, nothing, m_range, SHType)
    return Y
end

function computeYlm!(Y::AbstractVector, P::AbstractVector, θ,
    ϕ; lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())

    computeYlm!(Y, P, θ, ϕ, lmax, m, m_range, SHType)
    return Y
end

function _computeYlm_poles!(Y, P, ϕ, lmax, m_range)
    PS = _wrapSHArray(P, lmax, ZeroTo)
    Ylm = _wrapSHArray(Y, lmax, m_range)

    @inbounds for l in 0:lmax
        Ylm[(l, 0)] = PS[(l, 0)] * invsqrt2
    end
    return Y
end

function computeYlm!(Y::AbstractVector, P::AbstractVector, θ::Pole,
    ϕ, lmax::Integer, m_range::SHMRange = FullRange,
    SHType::HarmonicType = ComplexHarmonics())

    checksize_zeroY!(Y, P, lmax, m_range)
    _computeYlm_poles!(Y, P, ϕ, lmax, m_range)
    return Y
end
function computeYlm!(Y::AbstractVector, P::AbstractVector, θ::Pole,
    ϕ, lmax::Integer, m::Integer, m_range::SHMRange = FullRange,
    SHType::HarmonicType = ComplexHarmonics())

    checksize_zeroY!(Y, P, lmax, m_range)
    if iszero(m)
        _computeYlm_poles!(Y, P, ϕ, lmax, m_range)
    end
    return Y
end

"""
    computeYlm!(S::SphericalHarmonicsCache, θ, ϕ, [lmax::Integer])

Compute an entire set of spherical harmonics ``Y_{ℓ,m}(θ,ϕ)`` for ``0 ≤ ℓ ≤ ℓ_\\mathrm{max}`` using the
pre-computed associated Legendre polynomials saved in `S`, and store the result in `S`. If `lmax` is not provided,
the value of `lmax` for which associated Legendre polynomials have been computed in `S` is used.

!!! note
    This function assumes that the associated Legendre polynomials have been pre-computed, and does not perform
    any check on their values. In general `computeYlm!(S::SphericalHarmonicsCache, θ, ϕ, lmax)` should only be
    called after a preceeding call to `computePlmcostheta!(S, θ, lmax)` in order to obtain meaningful results.
"""
function computeYlm!(S::SphericalHarmonicsCache, θ, ϕ, lmax::Integer = _lmax(S))
    @assert lmax <= _lmax(S) "Plm for lmax = $lmax is not available, please run computePlmcostheta!(S, θ, lmax) first"
    !S.P.initialized && throw(ArgumentError("please run computePlmcostheta!(S, θ, lmax) first"))
    computeYlm!(S.Y, S.P, θ, ϕ, lmax, nothing, _mrange_basetype(S), S.SHType)
    return S.Y
end

"""
    computeYlm!(Y::AbstractVector, θ, ϕ; lmax::Integer, [m::Integer] [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])

Compute an entire set of spherical harmonics ``Y_{ℓ,m}(θ,ϕ)`` for ``0 ≤ ℓ ≤ ℓ_\\mathrm{max}``,
and store them in the array `Y`.

The optional argument `m_range` decides if the spherical harmonics for negative `m` values are computed.
By default the functions for all values of `m` are evaluated. Setting `m_range` to `SphericalHarmonics.ZeroTo` would result
in only functions for `m ≥ 0` being evaluated. Providing `m` would override this, in which case only the polynomials
corresponding to the azimuthal order `m` would be computed.

The optional argument `SHType` may be used to choose between real and complex harmonics.
To compute real spherical harmonics, set this to `SphericalHarmonics.RealHarmonics()`.
"""
function computeYlm!(Y::AbstractVector, θ::Number, ϕ, lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())
    P = computePlmcostheta(θ, lmax, _maybeabs(m))
    computeYlm!(Y, P, θ, ϕ, lmax, m, m_range, SHType)
    return Y
end
function computeYlm!(Y::AbstractVector, θ::Number, ϕ, lmax::Integer, m_range::SHMRange,
    SHType::HarmonicType = ComplexHarmonics())
    computeYlm!(Y, θ, ϕ, lmax, nothing, m_range, SHType)
    return Y
end

function computeYlm!(Y::AbstractVector, θ::Number, ϕ; lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())
    computeYlm!(Y, θ, ϕ, lmax, m, m_range, SHType)
    return Y
end

function _computeYlm(P, θ, ϕ, lmax, m, m_range, SHType)
    T = eltypeY(promote_type(eltype(P), typeof(ϕ)), SHType)
    Y = allocate_y(T, lmax, m_range)
    computeYlm!(Y, P, θ, ϕ, lmax, m, m_range, SHType)
    return Y
end

@doc raw"""
    computeYlm(θ, ϕ; lmax::Integer, [m::Integer], [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])
    computeYlm(θ::SphericalHarmonics.Pole; lmax::Integer, [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])

Compute an entire set of spherical harmonics ``Y_{ℓ,m}(θ,ϕ)`` for
``0 ≤ ℓ ≤ ℓ_\mathrm{max}`` and ``-ℓ ≤ m ≤ ℓ``, for colatitude ``\theta`` and
azimuth ``\phi``. If ``m`` is provided then only the polynomials for the specified ``m`` are computed.

Returns an `AbstractVector` that may be indexed using `(l,m)` pairs aside from the canonical
indexing with `Int`s.

The optional argument `m_range` decides if the spherical harmonics for negative `m` values are computed.
By default the functions for all values of `m` are evaluated. Setting `m_range` to `SphericalHarmonics.ZeroTo` would result
in only functions for `m ≥ 0` being evaluated.

The optional argument `SHType` may be used to choose between real and complex harmonics.
To compute real spherical harmonics, set this to `SphericalHarmonics.RealHarmonics()`. The complex harmonics
are defined as

```math
Y_{\ell,m}\left(\theta,\phi\right)=\begin{cases}
\frac{1}{\sqrt{2}}\bar{P}_{\ell}^{m}\left(\cos\theta\right)\exp\left(im\phi\right), & m\geq0,\\
\left(-1\right)^{m}Y_{\ell,-m}^{*}\left(\theta,\phi\right), & m<0.
\end{cases}
```

This definition corresponds to [Laplace spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics#Conventions),
whereas the quantum mechanical definition prepends a Condon-Shortley phase on the harmonics.

The real spherical harmonics are defined as

```math
Y_{\ell,m}\left(\theta,\phi\right)=\begin{cases}
\bar{P}_{\ell}^{\left|m\right|}\left(\cos\theta\right)\sin\left|m\right|\phi, & m<0,\\
\bar{P}_{\ell}^{0}\left(\cos\theta\right)/\sqrt{2}, & m=0,\\
\bar{P}_{\ell}^{m}\left(\cos\theta\right)\cos m\phi, & m>0.
\end{cases}
```

The precision of the result may be increased by using arbitrary-precision arguments.
"""*
"""
# Examples
```jldoctest
julia> Y = computeYlm(pi/2, 0, lmax = 1)
4-element SHArray(::$(Array{Complex{Float64},1}), (ML(0:1, -1:1),)):
     0.2820947917738782 + 0.0im
     0.3454941494713355 - 0.0im
 2.9918275112863375e-17 + 0.0im
    -0.3454941494713355 - 0.0im

julia> Y[(1,-1)] # index using (l,m)
0.3454941494713355 - 0.0im

julia> Y = computeYlm(big(pi)/2, big(0), lmax = big(1)) # Arbitrary precision
4-element SHArray(::$(Array{Complex{BigFloat},1}), (ML(0:1, -1:1),)):
    0.2820947917738781434740397257803862929220253146644994284220428608553212342207478 + 0.0im
    0.3454941494713354792652446460318896831393773703262433134867073548945156550201567 - 0.0im
 2.679783085063171668225419916118067917387251852939708540164955895366691604430101e-78 + 0.0im
   -0.3454941494713354792652446460318896831393773703262433134867073548945156550201567 - 0.0im

julia> computeYlm(SphericalHarmonics.NorthPole(), 0, lmax = 1)
4-element SHArray(::$(Array{Complex{Float64},1}), (ML(0:1, -1:1),)):
 0.2820947917738782 + 0.0im
               -0.0 + 0.0im
   0.48860251190292 + 0.0im
                0.0 + 0.0im

julia> Y = computeYlm(pi/2, pi/3, lmax = 1, m_range = SphericalHarmonics.ZeroTo, SHType = SphericalHarmonics.RealHarmonics())
3-element SHArray(::$(Array{Float64,1}), (ML(0:1, 0:1),)):
  0.2820947917738782
  2.9918275112863375e-17
 -0.24430125595146002
```
"""
function computeYlm(θ, ϕ, lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())

    T = float(promote_type(typeof(θ), typeof(ϕ)))
    P = computePlmcostheta(T(θ), lmax, _maybeabs(m))
    Y = _computeYlm(P, T(θ), T(ϕ), lmax, m, m_range, SHType)
    return Y
end
function computeYlm(θ, ϕ, lmax::Integer,
    m_range::SHMRange, SHType::HarmonicType = ComplexHarmonics())

    computeYlm(θ, ϕ, lmax, nothing, m_range, SHType)
end

function computeYlm(θ::Pole, ϕ, lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange,
    SHType::HarmonicType = ComplexHarmonics())

    P = computePlmcostheta(θ, lmax, _maybeabs(m))
    Y = _computeYlm(P, θ, ϕ, lmax, m, m_range, SHType)
    return Y
end
function computeYlm(θ::Pole, ϕ, lmax::Integer, m_range::SHMRange,
    SHType::HarmonicType = ComplexHarmonics())

    computeYlm(θ, ϕ, lmax, nothing, m_range, SHType)
end

function computeYlm(θ::Pole, lmax::Integer, m_range::SHMRange = FullRange,
    SHType::HarmonicType = ComplexHarmonics())
    computeYlm(θ, 0, lmax, nothing, m_range, SHType)
end

function computeYlm(θ, ϕ; lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())

    computeYlm(θ, ϕ, lmax, m, m_range, SHType)
end
function computeYlm(θ::Pole; lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())
    computeYlm(θ, 0, lmax, m, m_range, SHType)
end

"""
    SphericalHarmonics.sphericalharmonic(θ, ϕ; l, m, [SHType = ComplexHarmonics()], [coeff = nothing])

Evaluate the spherical harmonic ``Y_{ℓ,m}(θ, ϕ)``. The flag `SHType` sets the type of the harmonic computed,
and setting this to `RealHarmonics()` would evaluate real spherical harmonics. Optionally a precomputed
matrix of coefficients returned by [`compute_coefficients`](@ref) may be provided.

# Example
```jldoctest
julia> SphericalHarmonics.sphericalharmonic(π/3, π/3, l = 500, m = 250)
-0.18910100312194328 - 0.32753254516944075im

julia> SphericalHarmonics.sphericalharmonic(π/3, π/3, l = 500, m = 250, SHType = SphericalHarmonics.RealHarmonics())
-0.26742920327340913
```
"""
function sphericalharmonic(θ, ϕ; l::Integer, m::Integer,
    SHType::HarmonicType = ComplexHarmonics(), coeff = nothing)

    sphericalharmonic(θ, ϕ, l, m, SHType, coeff)
end
function sphericalharmonic(θ, ϕ, l::Integer, m::Integer, SHType::HarmonicType = ComplexHarmonics(),
    coeff = nothing)

    P = associatedLegendre(θ, l, abs(m), coeff)
    if m == 0
        return invsqrt2 * P
    end
    phasepos, phaseneg = phase(SHType, FullRange, abs(m), ϕ, invsqrt2, (-1)^Int(m))
    norm = m >= 0 ? phasepos : phaseneg
    norm * P
end

include("precompile.jl")

end
