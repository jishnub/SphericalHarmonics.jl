module SphericalHarmonics

using SphericalHarmonicModes
using SphericalHarmonicArrays
using ElasticArrays

export computeYlm, computeYlm!, computePlmcostheta, computePlmcostheta!

import Base: @propagate_inbounds

include("irrationals.jl")

abstract type HarmonicType end
struct RealHarmonics <: HarmonicType end
struct ComplexHarmonics <: HarmonicType end

mutable struct SphericalHarmonicsCache{T, M, SHT, C<:AbstractMatrix{<:Real}, PLM<:AbstractVector{<:Real}, YLM<:AbstractVector}
    lmax :: UInt
    C :: C
    P :: PLM
    Y :: YLM
end

function SphericalHarmonicsCache(T::Type, lmax::UInt; m_range = FullRange, SHType::HarmonicType = ComplexHarmonics())
    C = compute_coefficients(T, lmax)
    P = allocate_p(T, lmax)
    Y = allocate_y(eltypeY(T, SHType), lmax, m_range)
    SphericalHarmonicsCache{T,m_range,typeof(SHType),typeof(C),typeof(P),typeof(Y)}(lmax, C, P, Y)
end
SphericalHarmonicsCache(T::Type, lmax; kw...) = SphericalHarmonicsCache(T, UInt(lmax); kw...)
SphericalHarmonicsCache(lmax; kw...) = SphericalHarmonicsCache(Float64, lmax; kw...)

"""
    cache([T::Type = Float64], lmax; [m_range = SphericalHarmonicModes.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])

Allocate arrays to evaluate associated Legendre polynomials and spherical harmonics.
The returned object may be passed to [`computePlmcostheta!`](@ref) and [`computeYlm!`](@ref).
The coefficients are cached and need not be recomputed.

# Examples
```jldoctest cache
julia> S = SphericalHarmonics.cache(1);

julia> computePlmcostheta!(S, pi/3, 1)
3-element SHArray(::Vector{Float64}, (ML(0:1, 0:1),)):
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
6-element SHArray(::Vector{Float64}, (ML(0:2, 0:2),)):
  0.3989422804014327
  0.34549414947133555
 -0.4231421876608172
 -0.11150775725954817
 -0.4730873478787801
  0.40970566147202964
```
"""
cache(args...; kw...) = SphericalHarmonicsCache(args...; kw...)

const SHMRange = Union{Type{FullRange}, Type{ZeroTo}}

sizeP(maxDegree::Int) = div((maxDegree + 1) * (maxDegree + 2), 2)
sizeY(maxDegree::Int, ::Type{FullRange} = FullRange) = (maxDegree + 1) * (maxDegree + 1)
sizeY(maxDegree::Int, ::Type{ZeroTo}) = sizeP(maxDegree)

"""
    SphericalHarmonics.allocate_p([T::Type = Float64], L::Integer)

Allocate an array large enough to store an entire set of Associated Legendre
Polynomials ``\\bar{P}_l^m(x)`` of maximum degree ``L``.
"""
allocate_p(T::Type, L::Integer) = SHArray{T,1}(undef, (ML(ZeroTo(L), ZeroTo),))
allocate_p(L::Integer) = allocate_p(Float64, L)

"""
    SphericalHarmonics.allocate_y([T::Type = ComplexF64], L::Integer)

Allocate an array large enough to store an entire set of spherical harmonics
``Y_{l,m}(θ,ϕ)`` of maximum degree ``L``.
"""
allocate_y(T::Type, L::Integer, m_range = FullRange) = SHArray{T,1}(undef, (ML(ZeroTo(L), m_range),))
allocate_y(L::Integer, m_range = FullRange) = allocate_y(ComplexF64, L, m_range)

eltypeY(::Type{R}, ::ComplexHarmonics) where {R} = Complex{R}
eltypeY(::Type{R}, ::RealHarmonics) where {R} = R

"""
    SphericalHarmonics.compute_coefficients(L)

Precompute coefficients ``a_l^m`` and ``b_l^m`` for all ``2 ≤ l ≤ L`` and ``0 ≤ m ≤ l-2``.

    SphericalHarmonics.compute_coefficients(L, m)

Precompute coefficients ``a_l^m`` and ``b_l^m`` for all ``|m| + 2 ≤ l ≤ L`` and the specified ``m``.
"""
compute_coefficients(L::Integer) = compute_coefficients(typeof(sqrt(L)), L)
compute_coefficients(L::Integer, ::Nothing) = compute_coefficients(L)
compute_coefficients(L::Integer, m::Integer) = compute_coefficients(typeof(sqrt(L)), L, m)
function compute_coefficients(T::Type, L::Integer)
    @assert L >= 0 "degree must be >= 0"

    shmodes = ML(ZeroTo(Int(L)), ZeroTo)
    A = ElasticArray(zeros(T, 2, length(shmodes)))
    coeff = SHArray(A, (2, shmodes))

    @inbounds for l in 2:Int(L)
        # pre-compute certain terms to improve performance
        invlm1 = 1/T((l-1))
        invl = 1/T(l)
        Anum = 4 - invl^2
        Bden = 1/√(4 - invlm1^2)
        for m in 0:(l-2)
            coeff[1, (l, m)] = √(Anum / (1 - (m * invl)^2))
            coeff[2, (l, m)] = -√(1 - (m * invlm1)^2) * Bden
        end
    end
    return coeff
end
function compute_coefficients(T::Type, L::Integer, m::Integer)
    @assert L >= 0 "degree must be >= 0"
    @assert m >= 0 "m must be >= 0"

    shmodes = ML(ZeroTo(Int(L)), SingleValuedRange(m))
    coeff = zeros(T, 2, shmodes)

    @inbounds for l in abs(Int(m)) + 2:Int(L)
        # pre-compute certain terms to improve performance
        invlm1 = 1/T(l-1)
        invl = 1/T(l)
        coeff[1, (l,m)] = √((4 - invl^2) / (1 - (m * invl)^2))
        coeff[2, (l,m)] = -√((1 - (m * invlm1)^2) / (4 - invlm1^2))
    end
    return coeff
end
function compute_coefficients!(S::SphericalHarmonicsCache{T,M}, L::Integer) where {T,M}
    @assert L >= 0 "degree must be >= 0"
    if L > S.lmax
        shmodesP = ML(ZeroTo(Int(L)), ZeroTo)
        shmodesY = ML(ZeroTo(Int(L)), M)
        A = parent(S.C)
        resize!(A, 2, length(shmodesP))
        coeff = SHArray(ElasticArray(A), (2, shmodesP))

        @inbounds for l in (S.lmax+1):Int(L)
            # pre-compute certain terms to improve performance
            invlm1 = 1/T((l-1))
            invl = 1/T(l)
            Anum = 4 - invl^2
            Bden = 1/√(4 - invlm1^2)
            for m in 0:(l-2)
                coeff[1, (l, m)] = √(Anum / (1 - (m * invl)^2))
                coeff[2, (l, m)] = -√(1 - (m * invlm1)^2) * Bden
            end
        end
        S.lmax = L
        S.C = coeff
        P_new = resize!(parent(S.P), length(shmodesP))
        S.P = SHArray(P_new, (shmodesP,))
        Y_new = resize!(parent(S.Y), length(shmodesY))
        S.Y = SHArray(Y_new, (shmodesY,))
    end
    return S
end

function checksize(sz, minsize)
    @assert sz >= minsize "array needs to have a minimum size of $minsize, received size $sz"
end
function checksizesP(P, L)
    L >= 0 || throw(ArgumentError("lmax = $L does not correspond to a valid mode"))
    checksize(length(P), sizeP(Int(L)))
end

@propagate_inbounds function readcoeffs(coeff::SHArray, l, m)
    alm = coeff[1, (l,m)]
    blm = coeff[2, (l,m)]
    return alm, blm
end

function readcoeffs(coeff::Nothing, l, m)
    @assert abs(m) <= l - 2 "m must be <= l - 2"
    invl = 1/l
    invlm1 = 1/(l-1)
    alm = √((4 - invl^2)/(1 - (m * invl)^2))
    blm = -√((1 - (m * invlm1)^2 )/(4 - invlm1^2))
    return alm, blm
end

######################################################################################
# Associated Legendre polynomials
######################################################################################

_promotetype(costheta, sintheta, coeff::Nothing) = promote_type(typeof(costheta), typeof(sintheta))
_promotetype(costheta, sintheta, coeff::AbstractArray) = reduce(promote_type, (typeof(costheta), typeof(sintheta), eltype(coeff)))

_viewmodes(P, modes) = @view P[(firstindex(P) - 1) .+ (1:length(modes))]

function _wrapSHArray(P, modes)
    SHArray(_viewmodes(P, modes), (modes,))
end
function _wrapSHArray(P, L, m_range)
    modes = ML(ZeroTo(Int(L)), m_range)
    _wrapSHArray(P, modes)
end

@propagate_inbounds function _computePlmcostheta!(P::AbstractVector, costheta::Real, sintheta::Real, L::Integer,
    coeff::Union{AbstractArray,Nothing} = nothing)

    fill!(P, zero(eltype(P)))

    T = _promotetype(costheta, sintheta, coeff)
    Plm = _wrapSHArray(P, L, ZeroTo)

    Plm[(0, 0)] = _invsqrt2pi

    if (L > 0)
        Plm[(1, 0)] = _sqrt3by2pi * costheta
        P11 = -(_sqrt3by4pi * sintheta)
        Plm[(1, 1)] = P11
        temp = T(P11)

        for l in 2:Int(L)
            for m in 0:(l-2)
                alm, blm = readcoeffs(coeff, l, m)
                Plm[(l,m)] = alm * (costheta * Plm[(l - 1, m)] + blm * Plm[(l - 2, m)])
            end
            Plm[(l, l - 1)] = costheta * √(T(2l + 1)) * temp
            temp = -√(1 + 1 / 2T(l)) * sintheta * temp
            Plm[(l, l)] = temp
        end
    end

    return P
end

@propagate_inbounds function _computePlmcostheta!(P::AbstractVector, costheta::Real, sintheta::Real, L::Integer, m::Integer,
    coeff::Union{AbstractArray,Nothing} = nothing)

    0 <= m <= L || throw(ArgumentError("m = $m does not satisfy 0 ≤ m ≤ lmax = $L"))

    T = _promotetype(costheta, sintheta, coeff)

    fill!(P, zero(eltype(P)))
    Plm = _wrapSHArray(P, L, ZeroTo)

    if m == 0
        Plm[(0, 0)] = _invsqrt2pi
    end

    if (L > 0)
        P11 = -(_sqrt3by4pi * sintheta)
        Plm[(1, 1)] = P11

        # Compute Pmm using recursion over l and m
        for mi in 2:Int(m)
            Plm[(mi, mi)] = -√(1 + 1 / 2T(mi)) * sintheta * Plm[(mi-1, mi-1)]
        end

        if m == L
            return P
        end

        Plm[(m + 1, m)] = √(T(2m + 3)) * costheta * Plm[(m, m)]

        # Compute Plm using recursion over l at a fixed m
        for l in Int(m) + 2:Int(L)
            alm, blm = readcoeffs(coeff, l, m)
            Plm[(l,m)] = alm * (costheta * Plm[(l - 1, m)] + blm * Plm[(l - 2, m)])
        end
    end

    return P
end

@propagate_inbounds function _computePlmcostheta!(P::AbstractVector, costheta::Real, sintheta::Real, L::Integer, m::Nothing, coeff)
    _computePlmcostheta!(P, costheta, sintheta, L, coeff)
end

@propagate_inbounds function Plmrecursion(l, m, costheta, Pmm, Pmp1m, coeff)
    Plm2_m, Plm1_m = Pmm, Pmp1m

    # Compute Plm using recursion over l at a fixed m, starting from l = m + 2
    for li in Int(m) + 2:Int(l)
        alm, blm = readcoeffs(coeff, li, m)

        Pl_m = alm * (costheta * Plm1_m + blm * Plm2_m)

        Plm2_m, Plm1_m = Plm1_m, Pl_m
    end

    return Plm1_m
end

@propagate_inbounds function _computePlmcostheta(costheta::Real, sintheta::Real, l::Integer, m::Integer, coeff)
    0 <= m <= l || throw(ArgumentError("m = $m does not satisfy 0 ≤ m ≤ l = $l"))

    T = _promotetype(costheta, sintheta, coeff)

    Pmm = T(_invsqrt2pi)

    if m > 0
        Pmm = -T(_sqrt3by4pi * sintheta)
    end

    # Compute Pmm using recursion over l and m
    for mi in 2:Int(m)
        Pmm = -√(1 + 1 / 2T(mi)) * sintheta * Pmm
    end

    if m == l
        return Pmm
    end

    Pmp1m = √(T(2m + 3)) * costheta * Pmm

    # Recursion at a constant m to compute Pl,m from Pm,m and Pm+1,m
    Plm = Plmrecursion(l, m, costheta, Pmm, Pmp1m, coeff)
end

"""
    computePlmx!(P::AbstractVector{<:Real}, x::Real, lmax::Integer, coeff::AbstractMatrix)

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_l^m(x)``
using the given coefficients, and store in the array `P`. The matrix `coeff` may be computed
using [`compute_coefficients`](@ref).

The argument `x` needs to lie in ``-1 ≤ x ≤ 1``. The function implicitly assumes that
``x = \\cos(\\theta)`` where ``0 ≤ \\theta ≤ π``.

See [`computePlmcostheta`](@ref) for the normalization used.

    computePlmx!(P::AbstractVector{<:Real}, x::Real, lmax::Integer, m::Integer, coeff::AbstractMatrix)

Compute the set of normalized Associated Legendre Polynomials ``\\bar{P}_l^m(x)`` for for the specified ``m``
and all ``l`` lying in ``|m| ≤ l ≤ l_\\mathrm{max}`` .
"""
function computePlmx!(P::AbstractVector{<:Real}, x::Real, L::Integer, args...)
    checksizesP(P, L)
    -1 <= x <= 1 || throw(ArgumentError("x needs to lie in [-1,1]"))
    cosθ, sinθ, _L = promote(x, √(1-x^2), L)
    @inbounds _computePlmcostheta!(P, cosθ, sinθ, Integer(_L), args...)
    return P
end

function computePlmx!(S::SphericalHarmonicsCache, x::Real, L::Integer)
    compute_coefficients!(S, L)
    computePlmx!(S.P, x, L, S.C)
    return S.P
end

"""
    computePlmcostheta!(P::AbstractVector{<:Real}, θ::Real, lmax::Integer, coeff)
    computePlmcostheta!(P::AbstractVector{<:Real}, θ::SphericalHarmonics.Pole, lmax::Integer)

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_l^m(\\cos θ)``
using the given coefficients, and store in the array `P`. The matrix `coeff` may be computed
using [`compute_coefficients`](@ref).

See [`computePlmcostheta`](@ref) for the normalization used.

    computePlmcostheta!(P::AbstractVector{<:Real}, θ::Real, lmax::Integer, m::Integer, coeff)

Compute the Associated Legendre Polynomials ``\\bar{P}_l^m(\\cos θ)`` for for the specified ``m``
and all ``l`` lying in ``|m| ≤ l ≤ l_\\mathrm{max}``. The array `P` needs to be large enough to hold all the polynomials
for ``0 ≤ l ≤ l_\\mathrm{max}`` and ``0 ≤ m ≤ l``, as the recursive evaluation requires the computation of extra elements.
Pre-existing values in `P` may be overwritten, even for azimuthal orders not equal to ``m``.
"""
function computePlmcostheta!(P::AbstractVector{<:Real}, θ::Real, L::Integer, args...)
    checksizesP(P, L)
    cosθ, sinθ, _L = promote(cos(θ), sin(θ), L)
    @inbounds _computePlmcostheta!(P, cosθ, sinθ, Integer(_L), args...)
    return P
end

function computePlmcostheta!(P::AbstractVector{<:Real}, θ::Pole, L::Integer, m::Integer, args...)
    checksizesP(P, L)
    if !iszero(m)
        fill!(P, zero(eltype(P)))
        return P
    end
    computePlmcostheta!(P, θ, L)
    return P
end
computePlmcostheta!(P::AbstractVector{<:Real}, θ::Pole, L::Integer, args...) =
    computePlmcostheta!(P, θ, L)

function computePlmcostheta!(P::AbstractVector{T}, ::NorthPole, L::Integer) where {T<:Real}
    checksizesP(P, L)
    fill!(P, zero(eltype(P)))
    Plm = _wrapSHArray(P, L, ZeroTo)
    @inbounds for l in 0:L
        Plm[(l, 0)] = _invsqrt2pi * √(T(2l + 1))
    end
    return P
end

function computePlmcostheta!(P::AbstractVector{T}, ::SouthPole, L::Integer) where {T<:Real}
    checksizesP(P, L)
    fill!(P, zero(eltype(P)))
    Plm = _wrapSHArray(P, L, ZeroTo)
    phase = 1
    @inbounds for l in 0:L
        Plm[(l, 0)] = phase * _invsqrt2pi * √(T(2l + 1))
        phase *= -1
    end
    return P
end

function computePlmcostheta!(S::SphericalHarmonicsCache, θ::Real, L::Integer, m::Union{Integer,Nothing} = nothing)
    compute_coefficients!(S, L)
    computePlmcostheta!(S.P, θ, L, m, S.C)
    return S.P
end

"""
    computePlmcostheta(θ::Real; lmax::Integer, [m::Integer])
    computePlmcostheta(θ::Real, lmax::Integer, [m::Integer])

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_l^m(\\cos θ)`` where
``0 ≤ l ≤ l_\\mathrm{max}`` and ``0 ≤ m ≤ l`` for colatitude ``\\theta``. If `m` is provided then only the
polynomials corresponding to the specified `m` are computed.

The polynomials are normalized as

```math
\\bar{P}_{\\ell}^m = \\sqrt{\\frac{(2\\ell + 1)(\\ell-m)!}{2\\pi (\\ell+m)!}} P_{\\ell m},
```

where ``P_{\\ell m}`` are the standard
[associated Legendre polynomials](https://en.wikipedia.org/wiki/Associated_Legendre_polynomials#Alternative_notations),
and are defined in terms of Legendre polynomials ``P_\\ell(x)`` as

```math
P_{\\ell m}\\left(x\\right)=\\left(1-x^{2}\\right)^{m/2}\\frac{d^{m}}{dx^{m}}P_{\\ell}\\left(x\\right).
```

The normalized polynomials ``\\bar{P}_{\\ell}^m`` satisfy

```math
\\int_{0}^{\\pi} \\sin θ d\\theta\\,\\left| \\bar{P}_{\\ell}^m(\\cos θ) \\right|^2 = \\frac{1}{\\pi}
```

!!! info
    The Condon-Shortley phase factor ``(-1)^m`` is not included in the definition of the polynomials.

Returns an `SHVector` that may be
indexed using `(l,m)` pairs aside from the canonical indexing with `Int`s.

The precision of the result may be increased by using arbitrary-precision arguments.

# Examples
```jldoctest
julia> P = computePlmcostheta(pi/2, lmax = 1)
3-element SHArray(::$(Array{Float64,1}), (ML(0:1, 0:1),)):
  0.3989422804014327
  4.231083042742082e-17
 -0.4886025119029199

julia> P[(0,0)]
0.3989422804014327

julia> P = computePlmcostheta(big(pi)/2, lmax = big(1)) # Arbitrary precision
3-element SHArray(::$(Array{BigFloat,1}), (ML(0:1, 0:1),)):
  0.3989422804014326779399460599343818684758586311649346576659258296706579258993008
  3.789785583114350800838137317730900078444216599640987847808409161681770236721676e-78
 -0.4886025119029199215863846228383470045758856081942277021382431574458410003616367
```
"""
computePlmcostheta(θ::Real; lmax::Integer, m::Union{Integer, Nothing} = nothing) = computePlmcostheta(θ, lmax, m)

_applymaybeallm(f, P, θ, L, m::Nothing, coeff...) = f(P, θ, L, coeff...)
_applymaybeallm(f, P, θ, L, m, coeff...) = f(P, θ, L, m, coeff...)

_maybecomputecoeff(L, θ::Pole, m) = nothing
_maybecomputecoeff(L, θ, m) = compute_coefficients(L, m)

function computePlmcostheta(θ::Real, L::Integer, m::Union{Integer, Nothing} = nothing)
    T = promote_type(float(typeof(L)), float(typeof(θ)))
    P = allocate_p(T, L)
    coeff = _maybecomputecoeff(L, θ, m)
    _applymaybeallm(computePlmcostheta!, P, θ, L, m, coeff)
    return P
end

@doc raw"""
    SphericalHarmonics.associatedLegendre(θ::Real; l::Integer, m::Integer, [coeff = nothing])

Evaluate the normalized associated Legendre polynomial ``\bar{P}_l^m(\cos \theta)``.
Optionally a `Matrix` of coefficients returned by [`compute_coefficients`](@ref) may be provided.

See [`computePlmcostheta`](@ref) for the specific choice of normalization used here.
"""
associatedLegendre(θ::Real; l::Integer, m::Integer, coeff = nothing) = associatedLegendre(θ, l, m, coeff)
function associatedLegendre(θ::Real, l::Integer, m::Integer, coeff = nothing)
    _computePlmcostheta(cos(θ), sin(θ), l, m, coeff)
end

function associatedLegendre(θ::NorthPole, l::Integer, m::Integer, coeff = nothing)
    T = promote_type(Float64, float(typeof(l)))
    if m != 0
        return zero(T)
    end
    P = _invsqrt2pi * √(T(2l + 1))
end

function associatedLegendre(θ::SouthPole, l::Integer, m::Integer, coeff = nothing)
    T = promote_type(Float64, float(typeof(l)))
    if m != 0
        return zero(T)
    end
    P = _invsqrt2pi * (-1)^Int(l) * √(T(2l + 1))
end

"""
    computePlmx(x::Real; lmax::Integer, [m::Integer])
    computePlmx(x::Real, lmax::Integer, [m::Integer])

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_l^m(x)`` where
``0 ≤ l ≤ l_\\mathrm{max}`` and ``0 ≤ m ≤ l``. If ``m`` is provided then only the polynomials for that
azimuthal order are computed.

The argument `x` needs to lie in ``-1 ≤ x ≤ 1``. The function implicitly assumes that
``x = \\cos(\\theta)`` where ``0 ≤ \\theta ≤ π``.
"""
computePlmx(x::Real; lmax::Integer, m::Union{Integer, Nothing} = nothing) = computePlmx(x, lmax, m)
function computePlmx(x::Real, L::Integer, m::Union{Integer, Nothing} = nothing)
    T = promote_type(float(typeof(L)), float(typeof(x)))
    P = allocate_p(T, L)
    coeff = compute_coefficients(L, m)
    _applymaybeallm(computePlmx!, P, x, L, m, coeff)
    return P
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

@propagate_inbounds function fill_m_maybenegm!(Y, P, L, m, ϕ, CSphase, ::Type{FullRange}, SHType)
    m >= 0 || throw(ArgumentError("m must be ≥ 0"))
    @assert L >= 0 "degree must be non-negative"
    phasempos, phasemneg = phase(SHType, FullRange, m, ϕ, _invsqrt2, CSphase)

    PS = _wrapSHArray(P, L, ZeroTo)
    Ylm = _wrapSHArray(Y, L, FullRange)

    for l in m:Int(L)
        Plm = PS[(l, m)]
        Ylm[(l, -m)] = phasemneg * Plm
        Ylm[(l,  m)] = phasempos * Plm
    end
    return Y
end

@propagate_inbounds function fill_m_maybenegm!(Y, P, L, m, ϕ, CSphase, ::Type{ZeroTo}, SHType)
    m >= 0 || throw(ArgumentError("m must be ≥ 0"))
    @assert L >= 0 "degree must be non-negative"
    phasem = phase(SHType, ZeroTo, m, ϕ, _invsqrt2, CSphase)

    PS = _wrapSHArray(P, L, ZeroTo)
    Ylm = _wrapSHArray(Y, L, ZeroTo)

    for l in abs(Int(m)):Int(L)
        Ylm[(l,m)] = phasem * PS[(l,m)]
    end
    return Y
end

@propagate_inbounds function fill_m!(Y, P, L, m, ϕ, CSphase, ::Type{FullRange}, SHType::ComplexHarmonics)
    @assert L >= 0 "degree must be non-negative"
    phasempos, phasemneg = phase(SHType, FullRange, m, ϕ, _invsqrt2, CSphase)

    PS = _wrapSHArray(P, L, ZeroTo)
    Ylm = _wrapSHArray(Y, L, FullRange)

    for l in abs(Int(m)):Int(L)
        Plm = PS[(l, abs(m))]
        if m >= 0
            Ylm[(l, m)] = phasempos * Plm
        else
            Ylm[(l, m)] = (-1)^m * phasempos * Plm
        end
    end
    return Y
end

@propagate_inbounds function fill_m!(Y, P, L, m, ϕ, CSphase, ::Type{FullRange}, SHType::RealHarmonics)
    @assert L >= 0 "degree must be non-negative"
    C, S = phase(SHType, FullRange, m, ϕ, _invsqrt2, CSphase)

    PS = _wrapSHArray(P, L, ZeroTo)
    Ylm = _wrapSHArray(Y, L, FullRange)

    for l in abs(Int(m)):Int(L)
        Plm = PS[(l, abs(m))]
        if m >= 0
            Ylm[(l, m)] = C * Plm
        else
            Ylm[(l, m)] = S * Plm
        end
    end
    return Y
end

@propagate_inbounds function fill_m!(Y, P, L, m, ϕ, CSphase, ::Type{ZeroTo}, SHType)
    m >= 0 || throw(ArgumentError("m must be ≥ 0"))
    @assert L >= 0 "degree must be non-negative"

    phasem = phase(SHType, ZeroTo, m, ϕ, _invsqrt2, CSphase)
    PS = _wrapSHArray(P, L, ZeroTo)
    Ylm = _wrapSHArray(Y, L, ZeroTo)

    for l in Int(m):Int(L)
        Ylm[(l, m)] = phasem * P[(l, m)]
    end
    return Y
end

@propagate_inbounds function computeYlm_maybeallm!(Y, P, ϕ, L, ::Nothing, m_range, SHType)
    @assert L >= 0 "degree must be non-negative"
    PS = _wrapSHArray(P, L, ZeroTo)
    Ylm = _wrapSHArray(Y, L, m_range)
    for l in 0:Int(L)
        Ylm[(l, 0)] = PS[(l, 0)] * _invsqrt2
    end

    CSphase = 1
    for m in 1:Int(L)
        CSphase *= -1
        fill_m_maybenegm!(Y, P, L, m, ϕ, CSphase, m_range, SHType)
    end
    return Y
end
@propagate_inbounds function computeYlm_maybeallm!(Y, P, ϕ, L, m::Integer, m_range, SHType)
    @assert L >= 0 "degree must be non-negative"
    -L <= m <= L || throw(ArgumentError("m = $m does not satisfy 0 ≤ |m| ≤ lmax = $L"))

    PS = _wrapSHArray(P, L, ZeroTo)
    Ylm = _wrapSHArray(Y, L, m_range)
    if iszero(m)
        for l in 0:Int(L)
            Ylm[(l, 0)] = PS[(l, 0)] * _invsqrt2
        end
    else
        fill_m!(Y, P, L, m, ϕ, (-1)^m, m_range, SHType)
    end
    return Y
end

_maybeabs(::Nothing) = nothing
_maybeabs(m::Integer) = abs(m)

"""
    computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Real, ϕ::Real; lmax::Integer, [m::Integer], [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])
    computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Real, ϕ::Real, lmax::Integer, [m::Integer, [m_range = SphericalHarmonics.FullRange, [SHType = SphericalHarmonics.ComplexHarmonics()]]])

Compute an entire set of spherical harmonics ``Y_{l,m}(θ,ϕ)``
using the precomputed associated Legendre Polynomials ``\\bar{P}_l^m(\\cos θ)``,
and store in the array `Y`. The array `P` may be computed using [`computePlmcostheta`](@ref).

The optional argument `m_range` decides if the spherical harmonics for negative `m` values are computed.
By default the functions for all values of `m` are evaluated. Setting `m_range` to `SphericalHarmonics.ZeroTo` would result
in only functions for `m ≥ 0` being evaluated. Providing ``m`` would override this, in which case only the polynomials
corresponding to the azimuthal order ``m`` would be computed.

The optional argument `SHType` may be used to choose between real and complex harmonics.
To compute real spherical harmonics, set this to `RealHarmonics()`.

!!! note
    This function assumes that the associated Legendre Polynomials have been pre-computed, and does not perform any
    check on the values of `P`.
"""
function computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Real,
    ϕ::Real, L::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())

    checksize(length(P), sizeP(Int(L)))
    checksize(length(Y), sizeY(Int(L), m_range))

    fill!(Y, zero(eltype(Y)))
    @inbounds computeYlm_maybeallm!(Y, P, ϕ, L, m, m_range, SHType)

    return Y
end
function computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Real,
    ϕ::Real, L::Integer, m_range::SHMRange, SHType::HarmonicType = ComplexHarmonics())

    computeYlm!(Y, P, θ, ϕ, L, nothing, m_range, SHType)
end

function computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Real,
    ϕ::Real; lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())

    computeYlm!(Y, P, θ, ϕ, lmax, m, m_range, SHType)
end

function computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Pole,
    ϕ::Real, L::Integer, m_range::SHMRange = FullRange,
    SHType::HarmonicType = ComplexHarmonics())

    checksize(length(P), sizeP(Int(L)))
    checksize(length(Y), sizeY(L, m_range))

    fill!(Y, zero(eltype(Y)))

    PS = _wrapSHArray(P, L, ZeroTo)
    Ylm = _wrapSHArray(Y, L, m_range)

    @inbounds for l in 0:L
        Ylm[(l, 0)] = PS[(l, 0)] * _invsqrt2
    end

    return Y
end
function computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Pole,
    ϕ::Real, L::Integer, m::Integer, m_range::SHMRange = FullRange,
    SHType::HarmonicType = ComplexHarmonics())

    fill!(Y, zero(eltype(Y)))

    !iszero(m) && return Y

    computeYlm!(Y, P, θ, ϕ, L, m_range, SHType)

    return Y
end

function computeYlm!(S::SphericalHarmonicsCache{<:Any,M,SHT}, θ::Real, ϕ::Real, lmax::Integer, m::Union{Integer,Nothing} = nothing) where {M,SHT}
    @assert lmax <= S.lmax "Plm for lmax = $lmax is not available, please run computePlmcostheta! first"
    computeYlm!(S.Y, S.P, θ, ϕ, lmax, m, M, SHT())
    return S.Y
end

"""
    computeYlm!(Y::AbstractVector, θ::Real, ϕ::Real; lmax::Integer, [m::Integer] [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])

Compute an entire set of spherical harmonics ``Y_{l,m}(θ,ϕ)`` for ``0 ≤ l ≤ l_\\mathrm{max}``,
and store them in the array `Y`.

The optional argument `m_range` decides if the spherical harmonics for negative `m` values are computed.
By default the functions for all values of `m` are evaluated. Setting `m_range` to `SphericalHarmonics.ZeroTo` would result
in only functions for `m ≥ 0` being evaluated. Providing ``m`` would override this, in which case only the polynomials
corresponding to the azimuthal order ``m`` would be computed.

The optional argument `SHType` may be used to choose between real and complex harmonics.
To compute real spherical harmonics, set this to `SphericalHarmonics.RealHarmonics()`.
"""
function computeYlm!(Y::AbstractVector, θ::Real, ϕ::Real, L::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())
    P = computePlmcostheta(θ, L, _maybeabs(m))
    computeYlm!(Y, P, θ, ϕ, L, m, m_range, SHType)
    return Y
end
function computeYlm!(Y::AbstractVector, θ::Real, ϕ::Real, L::Integer, m_range::SHMRange,
    SHType::HarmonicType = ComplexHarmonics())
    computeYlm!(Y, θ, ϕ, L, nothing, m_range, SHType)
    return Y
end

function computeYlm!(Y::AbstractVector, θ::Real, ϕ::Real; lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())
    computeYlm!(Y, θ, ϕ, lmax, m, m_range, SHType)
    return Y
end

function _computeYlm(P, θ, ϕ, L, m, m_range, SHType)
    T = eltypeY(promote_type(eltype(P), typeof(ϕ)), SHType)
    Y = allocate_y(T, L, m_range)
    computeYlm!(Y, P, θ, ϕ, L, m, m_range, SHType)
    return Y
end

@doc raw"""
    computeYlm(θ::Real, ϕ::Real; lmax::Integer, [m::Integer], [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])
    computeYlm(θ::SphericalHarmonics.Pole; lmax::Integer, [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])

Compute an entire set of spherical harmonics ``Y_{l,m}(θ,ϕ)`` for
``0 ≤ l ≤ l_\mathrm{max}`` and ``-l ≤ m ≤ l``, for colatitude ``\theta`` and
azimuth ``\phi``. If ``m`` is provided then only the polynomials for the specified ``m`` are computed.

Returns an `SHVector` that may be
indexed using `(l,m)` pairs aside from the canonical indexing with `Int`s.

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

This definition corresponds to [Laplace sphercial harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics#Conventions),
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
function computeYlm(θ::Real, ϕ::Real, L::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())

    T = promote_type(promote_type(typeof(θ), typeof(ϕ)), float(typeof(L)))
    P = computePlmcostheta(T(θ), L, _maybeabs(m))
    Y = _computeYlm(P, T(θ), T(ϕ), L, m, m_range, SHType)
    return Y
end
function computeYlm(θ::Real, ϕ::Real, L::Integer,
    m_range::SHMRange, SHType::HarmonicType = ComplexHarmonics())

    computeYlm(θ, ϕ, L, nothing, m_range, SHType)
end

function computeYlm(θ::Pole, ϕ::Real, L::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange,
    SHType::HarmonicType = ComplexHarmonics())

    P = computePlmcostheta(θ, L, _maybeabs(m))
    Y = _computeYlm(P, θ, ϕ, L, m, m_range, SHType)
    return Y
end
function computeYlm(θ::Pole, ϕ::Real, L::Integer, m_range::SHMRange,
    SHType::HarmonicType = ComplexHarmonics())

    computeYlm(θ, ϕ, L, nothing, m_range, SHType)
end

function computeYlm(θ::Pole, L::Integer, m_range::SHMRange = FullRange,
    SHType::HarmonicType = ComplexHarmonics())
    computeYlm(θ, 0, L, nothing, m_range, SHType)
end

function computeYlm(θ::Real, ϕ::Real; lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())

    computeYlm(θ, ϕ, lmax, m, m_range, SHType)
end
function computeYlm(θ::Pole; lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())
    computeYlm(θ, 0, lmax, m, m_range, SHType)
end

"""
    SphericalHarmonics.sphericalharmonic(θ, ϕ; l, m, [SHType = ComplexHarmonics()], [coeff = nothing])

Evaluate the spherical harmonic ``Y_{l m}(θ, ϕ)``. The flag `SHType` sets the type of the harmonic computed,
and setting this to `RealHarmonics()` would evaluate real spherical harmonics. Optionally a precomputed
`Matrix` of coefficients returned by [`compute_coefficients`](@ref) may be provided.

# Example
```jldoctest
julia> SphericalHarmonics.sphericalharmonic(π/3, π/3, l = 500, m = 250)
-0.18910100312194328 - 0.32753254516944075im

julia> SphericalHarmonics.sphericalharmonic(π/3, π/3, l = 500, m = 250, SHType = SphericalHarmonics.RealHarmonics())
-0.26742920327340913
```
"""
function sphericalharmonic(θ::Real, ϕ::Real; l::Integer, m::Integer,
    SHType::HarmonicType = ComplexHarmonics(), coeff = nothing)

    sphericalharmonic(θ, ϕ, l, m, SHType, coeff)
end
function sphericalharmonic(θ::Real, ϕ::Real, l::Integer, m::Integer, SHType::HarmonicType = ComplexHarmonics(),
    coeff = nothing)

    TΩ = promote_type(float(typeof(θ)), float(typeof(ϕ)))
    Tlm = promote_type(float(typeof(l)), float(typeof(m)))
    T = promote_type(TΩ, Tlm)

    P = associatedLegendre(θ, l, abs(m), coeff)
    if m == 0
        return _invsqrt2 * P
    end
    phasepos, phaseneg = phase(SHType, FullRange, abs(m), ϕ, _invsqrt2, (-1)^Int(m))
    norm = m >= 0 ? phasepos : phaseneg
    norm * P
end

end
