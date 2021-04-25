module SphericalHarmonics

using SphericalHarmonicModes
using SphericalHarmonicArrays
using ElasticArrays
using Printf

export computeYlm, computeYlm!
export computePlmcostheta, computePlmcostheta!
export computePlmx, computePlmx!

import Base: @propagate_inbounds

include("irrationals.jl")

abstract type HarmonicType end
struct RealHarmonics <: HarmonicType end
struct ComplexHarmonics <: HarmonicType end

"""
    SphericalHarmonicsCache

Preallocate arrays of associated Legendre polynomials and spherical harmonics.
Such an object may be allocated using [`cache`](@ref).
"""
mutable struct SphericalHarmonicsCache{T, M, SHT, C<:AbstractMatrix{<:Real}, PLM<:AbstractVector{<:Real}, YLM<:AbstractVector}
    lmax :: Int
    C :: C
    P :: PLM
    Y :: YLM
end

mutable struct AssociatedLegendrePolynomials{T, PLM<:AbstractVector{T}} <: AbstractVector{T}
    cosθ :: T
    lmax :: Int
    P :: PLM
    initialized :: Bool
end
Base.parent(P::AssociatedLegendrePolynomials) = P.P
Base.size(P::AssociatedLegendrePolynomials) = size(parent(P))
Base.axes(P::AssociatedLegendrePolynomials) = axes(parent(P))
Base.IndexStyle(::Type{<:AssociatedLegendrePolynomials{<:Any, PLM}}) where {PLM} = IndexStyle(PLM)
Base.@propagate_inbounds Base.getindex(P::AssociatedLegendrePolynomials, I...) = getindex(parent(P), I...)
function Base.summary(io::IO, P::AssociatedLegendrePolynomials)
    print(io, "$(length(P))-element AssociatedLegendrePolynomials{$(eltype(P))} for lmax = $(Int(P.lmax))")
    if P.initialized
        print(io, " and cosθ = ")
        @printf io "%.4g" P.cosθ
    else
        print(io, " (uninitialized)")
    end
end

function SphericalHarmonicsCache(T::Type, lmax::Int; m_range = FullRange, SHType::HarmonicType = ComplexHarmonics())
    C = compute_coefficients(T, lmax)
    P = AssociatedLegendrePolynomials(T(0), lmax, allocate_p(T, lmax), false)
    Y = allocate_y(eltypeY(T, SHType), lmax, m_range)
    SphericalHarmonicsCache{T,m_range,typeof(SHType),typeof(C),typeof(P),typeof(Y)}(lmax, C, P, Y)
end
SphericalHarmonicsCache(lmax; kw...) = SphericalHarmonicsCache(Float64, lmax; kw...)

function Base.show(io::IO, S::SphericalHarmonicsCache{T,M,SHT}) where {T,M,SHT}
    print(io, "$SphericalHarmonicsCache($T, $(Int(S.lmax)), m_range = $M, SHType = $SHT())")
end

"""
    cache([T::Type = Float64], lmax; [m_range = SphericalHarmonicModes.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])

Allocate arrays to evaluate associated Legendre polynomials and spherical harmonics.
The returned object may be passed to [`computePlmcostheta!`](@ref) and [`computeYlm!`](@ref).
The coefficients are cached and need not be recomputed.

# Examples
```jldoctest cache
julia> S = SphericalHarmonics.cache(1);

julia> computePlmcostheta!(S, pi/3, 1)
3-element AssociatedLegendrePolynomials{Float64} for lmax = 1 and cosθ = 0.5:
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
6-element AssociatedLegendrePolynomials{Float64} for lmax = 2 and cosθ = 0.5:
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

eltypeY(::Type{R}, ::ComplexHarmonics) where {R} = Complex{R}
eltypeY(::Type{R}, ::RealHarmonics) where {R} = R

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
    A = ElasticArray(zeros(T, 2, length(shmodes)))
    coeff = SHArray(A, (2, shmodes))

    @inbounds _compute_coefficients!(coeff, 2, Int(lmax))
    return coeff
end
compute_coefficients(T::Type, lmax::Integer, ::Nothing) = compute_coefficients(T, lmax)
function compute_coefficients(T::Type, lmax::Integer, m::Integer)
    @assert lmax >= 0 "degree must be non-negative"
    @assert m >= 0 "m must be non-negative"

    shmodes = ML(ZeroTo(Int(lmax)), SingleValuedRange(m))
    coeff = zeros(T, 2, shmodes)

    @inbounds _compute_coefficients!(coeff, abs(Int(m)) + 2, Int(lmax), abs(Int(m)), abs(Int(m)))
    return coeff
end
function compute_coefficients!(S::SphericalHarmonicsCache{T,M}, lmax::Integer) where {T,M}
    @assert lmax >= 0 "degree must be >= 0"
    if lmax > S.lmax
        shmodesP = ML(ZeroTo(Int(lmax)), ZeroTo)
        shmodesY = ML(ZeroTo(Int(lmax)), M)
        A = parent(S.C)
        resize!(A, 2, length(shmodesP))
        coeff = SHArray(ElasticArray(A), (2, shmodesP))

        @inbounds _compute_coefficients!(coeff, Int(S.lmax)+1, Int(lmax))

        S.lmax = lmax
        S.C = coeff
        P_len = length(S.P)
        P_new = resize!(parent(parent(S.P)), length(shmodesP))
        P_new[P_len+1:end] .= zero(eltype(P_new))
        P_new_SA = SHArray(P_new, (shmodesP,))
        S.P.P = P_new_SA

        Y_len = length(S.Y)
        Y_new = resize!(parent(S.Y), length(shmodesY))
        Y_new[Y_len+1:end] .= zero(eltype(Y_new))
        S.Y = SHArray(Y_new, (shmodesY,))
    end
    return S.C
end

@propagate_inbounds function _compute_coefficients!(coeff::AbstractArray{T}, lmin, lmax, m_min = 0, m_max = lmax - 2) where {T}
    for l in lmin:lmax
        # pre-compute certain terms to improve performance
        invlm1 = 1/T(l-1)
        invl = 1/T(l)
        Anum = 4 - invl^2
        Bden = 1/√(4 - invlm1^2)
        for m in max(0, m_min):min(l-2, m_max)
            coeff[1, (l, m)] = √(Anum / (1 - (m * invl)^2))
            coeff[2, (l, m)] = -√(1 - (m * invlm1)^2) * Bden
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

@propagate_inbounds function readcoeffs(coeff::SHArray, T::Type, l, m)
    alm = coeff[1, (l,m)]
    blm = coeff[2, (l,m)]
    return alm, blm
end

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
_promotetype(costheta, sintheta, coeff::AbstractArray) = float(reduce(promote_type, (typeof(costheta), typeof(sintheta), eltype(coeff))))

_viewmodes(P, modes) = @view P[(firstindex(P) - 1) .+ (1:length(modes))]

function _wrapSHArray(P, modes)
    SHArray(_viewmodes(P, modes), (modes,))
end
function _wrapSHArray(P, lmax, m_range)
    modes = ML(ZeroTo(Int(lmax)), m_range)
    _wrapSHArray(P, modes)
end

@propagate_inbounds function _computePlmcostheta!(P, costheta, sintheta, l_range,
    coeff::Union{AbstractArray,Nothing} = nothing)

    lmin, lmax = map(Int, extrema(l_range))
    @assert lmin >= 0 "degree must be non-negative"

    T = _promotetype(costheta, sintheta, coeff)
    Plm = _wrapSHArray(P, lmax, ZeroTo)

    if lmin == 0
        Plm[(0, 0)] = _invsqrt2pi
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
            temp = -√(T(2l + 1)/2l) * sintheta * temp
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

    return P
end

@propagate_inbounds function _computePlmcostheta!(P, costheta, sintheta, l_range, m::Integer,
    coeff::Union{AbstractArray,Nothing} = nothing)

    lmin, lmax = map(Int, extrema(l_range))
    @assert lmin == 0 "minimum degree must be zero"
    # @assert lmin >= 0 "degree must be non-negative"
    0 <= m <= lmax || throw(ArgumentError("m = $m does not satisfy 0 ≤ m ≤ lmax = $lmax"))

    T = _promotetype(costheta, sintheta, coeff)

    Plm = _wrapSHArray(P, lmax, ZeroTo)

    if m == 0 && lmin == 0
        Plm[(0, 0)] = _invsqrt2pi
    end
    temp = T(_invsqrt2pi)

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
            return P
        end

        Plm[(m + 1, m)] = √(T(2m + 3)) * costheta * temp

        # Compute Plm using recursion over l at a fixed m
        for l in max(lmin, Int(m) + 2):lmax
            alm, blm = readcoeffs(coeff, T, l, m)
            Plm[(l,m)] = alm * (costheta * Plm[(l - 1, m)] + blm * Plm[(l - 2, m)])
        end
    end

    return P
end

@propagate_inbounds function _computePlmcostheta!(P, costheta, sintheta, l_range, m::Nothing, coeff::Union{AbstractArray,Nothing})
    _computePlmcostheta!(P, costheta, sintheta, l_range, coeff)
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

@propagate_inbounds function _computePlmcostheta(costheta, sintheta, l, m, coeff)
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
    Plm = Plmrecursion(l, m, costheta, Pmm, Pmp1m, coeff, T)
end

"""
    computePlmx!(P::AbstractVector{<:Real}, x::Real, lmax::Integer, coeff::AbstractMatrix)

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_ℓ^m(x)``
using the given coefficients, and store in the array `P`.
The matrix `coeff` may be computed using [`compute_coefficients`](@ref).

The argument `x` needs to lie in ``-1 ≤ x ≤ 1``. The function implicitly assumes that
``x = \\cos(\\theta)`` where ``0 ≤ \\theta ≤ π``.

See [`computePlmcostheta`](@ref) for the normalization used.

    computePlmx!(P::AbstractVector{<:Real}, x::Real, lmax::Integer, m::Integer, coeff::AbstractMatrix)

Compute the set of normalized Associated Legendre Polynomials ``\\bar{P}_ℓ^m(x)`` for for the specified ``m``
and all ``ℓ`` lying in ``|m| ≤ ℓ ≤ ℓ_\\mathrm{max}`` .
"""
function computePlmx!(P::AbstractVector{<:Real}, x::Real, lmax::Integer, args...)
    @assert lmax >= 0 "degree must be non-negative"
    -1 <= x <= 1 || throw(DomainError("x", "The argument to associated Legendre polynomials must satisfy -1 <= x <= 1"))
    fill!(P, zero(eltype(P)))
    _computePlmx_range!(P, x, 0:lmax, args...)
    return P
end

"""
    computePlmx!(S::SphericalHarmonicsCache, x::Real, [lmax::Integer])

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_ℓ^m(x)``
using the pre-computed coefficients in `S`, and store the result in `S`. If `lmax` is not provided,
the value of `lmax` for which coefficients have been computed in `S` is used.
"""
function computePlmx!(S::SphericalHarmonicsCache, x::Real, lmax::Integer = S.lmax)
    compute_coefficients!(S, lmax)
    computePlmx!(S.P, x, lmax, S.C)
    return S.P
end
function computePlmx!(P::AssociatedLegendrePolynomials, x::Real, lmax::Integer, coeff::Union{AbstractMatrix,Nothing} = nothing)
    @assert lmax >= 0 "degree must be non-negative"
    -1 <= x <= 1 || throw(DomainError("x", "The argument to associated Legendre polynomials must satisfy -1 <= x <= 1"))
    if P.initialized && P.cosθ == x && P.lmax >= lmax
        return P
    elseif P.initialized && P.cosθ == x && P.lmax < lmax
        _computePlmx_range!(parent(P), x, P.lmax+1:lmax, coeff)
    else
        computePlmx!(parent(P), x, lmax, coeff)
        P.cosθ = x
    end
    P.lmax = lmax
    P.initialized = true
    return P
end
function _computePlmx_range!(P::AbstractVector{<:Real}, x::Real, l_range::AbstractUnitRange{<:Integer}, args...)
    lmin, lmax = map(Int, extrema(l_range))
    checksizesP(P, lmax)
    cosθ, sinθ  = promote(x, √(1-x^2))
    @inbounds _computePlmcostheta!(P, cosθ, sinθ, lmin:lmax, args...)
    return P
end

"""
    computePlmx(x::Real; lmax::Integer, [m::Integer])
    computePlmx(x::Real, lmax::Integer, [m::Integer])

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_ℓ^m(x)`` where
``0 ≤ ℓ ≤ ℓ_\\mathrm{max}`` and ``0 ≤ m ≤ ℓ``. If `m` is provided then only the polynomials for that
azimuthal order are computed.

The argument `x` needs to lie in ``-1 ≤ x ≤ 1``. The function implicitly assumes that
``x = \\cos(\\theta)`` where ``0 ≤ \\theta ≤ π``.
"""
computePlmx(x::Real; lmax::Integer, m::Union{Integer, Nothing} = nothing) = computePlmx(x, lmax, m)
function computePlmx(x::Real, lmax::Integer, m::Union{Integer, Nothing} = nothing)
    P = allocate_p(float(typeof(x)), lmax)
    coeff = compute_coefficients(lmax, m)
    _applymaybeallm(computePlmx!, P, x, lmax, m, coeff)
    return AssociatedLegendrePolynomials(x, Int(lmax), P, true)
end

"""
    computePlmcostheta!(P::AbstractVector{<:Real}, θ::Real, lmax::Integer, coeff)
    computePlmcostheta!(P::AbstractVector{<:Real}, θ::SphericalHarmonics.Pole, lmax::Integer)

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_ℓ^m(\\cos θ)``
using the given coefficients, and store in the array `P`. The matrix `coeff` may be computed
using [`compute_coefficients`](@ref).

See [`computePlmcostheta`](@ref) for the normalization used.

    computePlmcostheta!(P::AbstractVector{<:Real}, θ::Real, lmax::Integer, m::Integer, coeff)

Compute the Associated Legendre Polynomials ``\\bar{P}_ℓ^m(\\cos θ)`` for for the specified ``m``
and all ``ℓ`` lying in ``|m| ≤ ℓ ≤ ℓ_\\mathrm{max}``. The array `P` needs to be large enough to hold all the polynomials
for ``0 ≤ ℓ ≤ ℓ_\\mathrm{max}`` and ``0 ≤ m ≤ ℓ``, as the recursive evaluation requires the computation of extra elements.
Pre-existing values in `P` may be overwritten, even for azimuthal orders not equal to ``m``.
"""
function computePlmcostheta!(P::AbstractVector{<:Real}, θ::Real, lmax, args...)
    fill!(P, zero(eltype(P)))
    _computePlmcostheta_range!(P, θ, 0:lmax, args...)
    return P
end
function _computePlmcostheta_range!(P::AbstractVector{<:Real}, θ::Real, l_range::AbstractUnitRange{<:Integer}, args...)
    lmin, lmax = map(Int, extrema(l_range))
    checksizesP(P, lmax)
    cosθ, sinθ  = promote(cos(θ), sin(θ))
    @inbounds _computePlmcostheta!(P, cosθ, sinθ, lmin:lmax, args...)
    return P
end

function computePlmcostheta!(P::AbstractVector{<:Real}, θ::Pole, lmax, args...)
    fill!(P, zero(eltype(P)))
    _computePlmcostheta_range!(P, θ, 0:lmax, args...)
    return P
end
function _computePlmcostheta_range!(P::AbstractVector{<:Real}, θ::Pole, l_range::AbstractUnitRange{<:Integer}, m::Integer, args...)
    lmin, lmax = map(Int, extrema(l_range))
    checksizesP(P, lmax)
    if !iszero(m)
        return P
    end
    _computePlmcostheta_m0_range!(P, θ, lmin:lmax)
    return P
end
function _computePlmcostheta_range!(P::AbstractVector{<:Real}, θ::Pole, l_range::AbstractUnitRange{<:Integer}, args...)
    _computePlmcostheta_m0_range!(P, θ, l_range)
    return P
end

function _computePlmcostheta_m0_range!(P::AbstractVector{T}, ::NorthPole, l_range::AbstractUnitRange{<:Integer}) where {T<:Real}
    lmin, lmax = map(Int, extrema(l_range))
    checksizesP(P, lmax)
    Plm = _wrapSHArray(P, lmax, ZeroTo)
    @inbounds for l in lmin:lmax
        Plm[(l, 0)] = _invsqrt2pi * √(T(2l + 1))
    end
    return P
end

function _computePlmcostheta_m0_range!(P::AbstractVector{T}, ::SouthPole, l_range::AbstractUnitRange{<:Integer}) where {T<:Real}
    lmin, lmax = map(Int, extrema(l_range))
    checksizesP(P, lmax)
    fill!(P, zero(eltype(P)))
    Plm = _wrapSHArray(P, lmax, ZeroTo)
    phase = 1
    @inbounds for l in lmin:lmax
        Plm[(l, 0)] = phase * _invsqrt2pi * √(T(2l + 1))
        phase *= -1
    end
    return P
end

function computePlmcostheta!(P::AssociatedLegendrePolynomials, θ::Pole, lmax, args...)
    _computePlmcostheta_alp!(P, θ, lmax, args...)
    return P
end
function computePlmcostheta!(P::AssociatedLegendrePolynomials, θ::Real, lmax, args...)
    _computePlmcostheta_alp!(P, θ, lmax, args...)
    return P
end
function _computePlmcostheta_alp!(P::AssociatedLegendrePolynomials, θ::Real, lmax, args...)
    if P.initialized && P.cosθ == cos(θ) && P.lmax >= lmax
        return P
    elseif P.initialized && P.cosθ == cos(θ) && P.lmax < lmax
        _computePlmcostheta_range!(parent(P), θ, P.lmax+1:lmax, args...)
    else
        computePlmcostheta!(parent(P), θ, lmax, args...)
        P.cosθ = cos(θ)
    end
    P.lmax = lmax
    P.initialized = true
    return P
end

"""
    computePlmcostheta!(S::SphericalHarmonicsCache, θ::Real, [lmax::Integer])

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_ℓ^m(\\cos θ)``
using the pre-computed coefficients in `S`, and store the result in `S`. If `lmax` is not provided,
the value of `lmax` for which coefficients have been computed in `S` is used.
"""
function computePlmcostheta!(S::SphericalHarmonicsCache, θ::Real, lmax::Integer = S.lmax)
    compute_coefficients!(S, lmax)
    computePlmcostheta!(S.P, θ, lmax, S.C)
    return S.P
end

"""
    computePlmcostheta(θ::Real; lmax::Integer, [m::Integer])
    computePlmcostheta(θ::Real, lmax::Integer, [m::Integer])

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
P_{\\ell m}\\left(x\\right)=\\left(1-x^{2}\\right)^{m/2}\\frac{d^{m}}{dx^{m}}P_{\\ell}\\left(x\\right).
```

The normalized polynomials ``\\bar{P}_{\\ell}^m`` satisfy

```math
\\int_{0}^{\\pi} \\sin θ d\\theta\\,\\left| \\bar{P}_{\\ell}^m(\\cos θ) \\right|^2 = \\frac{1}{\\pi}
```

!!! info
    The Condon-Shortley phase factor ``(-1)^m`` is not included in the definition of the polynomials.

Returns an `AbstractVector` that may be indexed using `(ℓ,m)` pairs aside from the
canonical indexing with `Int`s.

The precision of the result may be increased by using arbitrary-precision arguments.

# Examples
```jldoctest
julia> P = computePlmcostheta(pi/2, lmax = 1)
3-element AssociatedLegendrePolynomials{Float64} for lmax = 1 and cosθ = 6.123e-17:
  0.3989422804014327
  4.231083042742082e-17
 -0.4886025119029199

julia> P[(0,0)]
0.3989422804014327

julia> P = computePlmcostheta(big(pi)/2, lmax = 1)
3-element AssociatedLegendrePolynomials{BigFloat} for lmax = 1 and cosθ = 5.485e-78:
  0.3989422804014326779399460599343818684758586311649346576659258296706579258993008
  3.789785583114350800838137317730900078444216599640987847808409161681770236721676e-78
 -0.4886025119029199215863846228383470045758856081942277021382431574458410003616367
```
"""
computePlmcostheta(θ::Real; lmax::Integer, m::Union{Integer, Nothing} = nothing) = computePlmcostheta(θ, lmax, m)

_applymaybeallm(f, P, θ, lmax, m::Nothing, coeff...) = f(P, θ, lmax, coeff...)
_applymaybeallm(f, P, θ, lmax, m, coeff...) = f(P, θ, lmax, m, coeff...)

_maybecomputecoeff(lmax, θ::Pole, m) = nothing
_maybecomputecoeff(lmax, θ, m) = compute_coefficients(promote_type(Float64, float(typeof(θ))), Int(lmax), m)

function computePlmcostheta(θ::Real, lmax::Integer, m::Union{Integer, Nothing} = nothing)
    coeff = _maybecomputecoeff(lmax, θ, m)
    P = allocate_p(float(typeof(θ)), lmax)
    _applymaybeallm(computePlmcostheta!, P, θ, lmax, m, coeff)
    return AssociatedLegendrePolynomials(cos(θ), Int(lmax), P, true)
end

@doc raw"""
    SphericalHarmonics.associatedLegendre(θ::Real; l::Integer, m::Integer, [coeff = nothing])

Evaluate the normalized associated Legendre polynomial ``\bar{P}_ℓ^m(\cos \theta)``.
Optionally a matrix of coefficients returned by [`compute_coefficients`](@ref) may be provided.

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
    phasempos, phasemneg = phase(SHType, FullRange, m, ϕ, _invsqrt2, CSphase)

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
    phasem = phase(SHType, ZeroTo, m, ϕ, _invsqrt2, CSphase)

    PS = _wrapSHArray(P, lmax, ZeroTo)
    Ylm = _wrapSHArray(Y, lmax, ZeroTo)

    for l in abs(Int(m)):Int(lmax)
        Ylm[(l,m)] = phasem * PS[(l,m)]
    end
    return Y
end

@propagate_inbounds function fill_m!(Y, P, lmax, m, ϕ, CSphase, ::Type{FullRange}, SHType::ComplexHarmonics)
    @assert lmax >= 0 "degree must be non-negative"
    phasempos, phasemneg = phase(SHType, FullRange, m, ϕ, _invsqrt2, CSphase)

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
    C, S = phase(SHType, FullRange, m, ϕ, _invsqrt2, CSphase)

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

    phasem = phase(SHType, ZeroTo, m, ϕ, _invsqrt2, CSphase)
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
        Ylm[(l, 0)] = PS[(l, 0)] * _invsqrt2
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
            Ylm[(l, 0)] = PS[(l, 0)] * _invsqrt2
        end
    else
        fill_m!(Y, P, lmax, m, ϕ, (-1)^m, m_range, SHType)
    end
    return Y
end

_maybeabs(::Nothing) = nothing
_maybeabs(m::Integer) = abs(m)

"""
    computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Real, ϕ::Real; lmax::Integer, [m::Integer], [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])
    computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Real, ϕ::Real, lmax::Integer, [m::Integer, [m_range = SphericalHarmonics.FullRange, [SHType = SphericalHarmonics.ComplexHarmonics()]]])

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
function computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Real,
    ϕ::Real, lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())

    checksize(length(P), sizeP(Int(lmax)))
    checksize(length(Y), sizeY(Int(lmax), m_range))

    fill!(Y, zero(eltype(Y)))
    @inbounds computeYlm_maybeallm!(Y, P, ϕ, lmax, m, m_range, SHType)

    return Y
end
function computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Real,
    ϕ::Real, lmax::Integer, m_range::SHMRange, SHType::HarmonicType = ComplexHarmonics())

    computeYlm!(Y, P, θ, ϕ, lmax, nothing, m_range, SHType)
end

function computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Real,
    ϕ::Real; lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())

    computeYlm!(Y, P, θ, ϕ, lmax, m, m_range, SHType)
end

function computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Pole,
    ϕ::Real, lmax::Integer, m_range::SHMRange = FullRange,
    SHType::HarmonicType = ComplexHarmonics())

    checksize(length(P), sizeP(Int(lmax)))
    checksize(length(Y), sizeY(lmax, m_range))

    fill!(Y, zero(eltype(Y)))

    PS = _wrapSHArray(P, lmax, ZeroTo)
    Ylm = _wrapSHArray(Y, lmax, m_range)

    @inbounds for l in 0:lmax
        Ylm[(l, 0)] = PS[(l, 0)] * _invsqrt2
    end

    return Y
end
function computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Pole,
    ϕ::Real, lmax::Integer, m::Integer, m_range::SHMRange = FullRange,
    SHType::HarmonicType = ComplexHarmonics())

    fill!(Y, zero(eltype(Y)))

    !iszero(m) && return Y

    computeYlm!(Y, P, θ, ϕ, lmax, m_range, SHType)

    return Y
end

"""
    computeYlm!(S::SphericalHarmonicsCache, θ::Real, ϕ::Real, [lmax::Integer])

Compute an entire set of spherical harmonics ``Y_{ℓ,m}(θ,ϕ)`` for ``0 ≤ ℓ ≤ ℓ_\\mathrm{max}`` using the
pre-computed associated Legendre polynomials saved in `S`, and store the result in `S`. If `lmax` is not provided,
the value of `lmax` for which associated Legendre polynomials have been computed in `S` is used.

!!! note
    This function assumes that the associated Legendre polynomials have been pre-computed, and does not perform
    any check on their values. In general `computeYlm!(S::SphericalHarmonicsCache, θ, ϕ, lmax)` should only be
    called after a preceeding call to `computePlmcostheta!(S, θ, lmax)` in order to obtain meaningful results.
"""
function computeYlm!(S::SphericalHarmonicsCache{<:Any,M,SHT}, θ::Real, ϕ::Real, lmax::Integer = S.lmax) where {M,SHT}
    @assert lmax <= S.lmax "Plm for lmax = $lmax is not available, please run computePlmcostheta!(S, θ, lmax) first"
    !S.P.initialized && throw(ArgumentError("please run computePlmcostheta!(S, θ, lmax) first"))
    computeYlm!(S.Y, S.P, θ, ϕ, lmax, nothing, M, SHT())
    return S.Y
end

"""
    computeYlm!(Y::AbstractVector, θ::Real, ϕ::Real; lmax::Integer, [m::Integer] [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])

Compute an entire set of spherical harmonics ``Y_{ℓ,m}(θ,ϕ)`` for ``0 ≤ ℓ ≤ ℓ_\\mathrm{max}``,
and store them in the array `Y`.

The optional argument `m_range` decides if the spherical harmonics for negative `m` values are computed.
By default the functions for all values of `m` are evaluated. Setting `m_range` to `SphericalHarmonics.ZeroTo` would result
in only functions for `m ≥ 0` being evaluated. Providing `m` would override this, in which case only the polynomials
corresponding to the azimuthal order `m` would be computed.

The optional argument `SHType` may be used to choose between real and complex harmonics.
To compute real spherical harmonics, set this to `SphericalHarmonics.RealHarmonics()`.
"""
function computeYlm!(Y::AbstractVector, θ::Real, ϕ::Real, lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())
    P = computePlmcostheta(θ, lmax, _maybeabs(m))
    computeYlm!(Y, P, θ, ϕ, lmax, m, m_range, SHType)
    return Y
end
function computeYlm!(Y::AbstractVector, θ::Real, ϕ::Real, lmax::Integer, m_range::SHMRange,
    SHType::HarmonicType = ComplexHarmonics())
    computeYlm!(Y, θ, ϕ, lmax, nothing, m_range, SHType)
    return Y
end

function computeYlm!(Y::AbstractVector, θ::Real, ϕ::Real; lmax::Integer, m::Union{Integer,Nothing} = nothing,
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
    computeYlm(θ::Real, ϕ::Real; lmax::Integer, [m::Integer], [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])
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
function computeYlm(θ::Real, ϕ::Real, lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())

    T = float(promote_type(typeof(θ), typeof(ϕ)))
    P = computePlmcostheta(T(θ), lmax, _maybeabs(m))
    Y = _computeYlm(P, T(θ), T(ϕ), lmax, m, m_range, SHType)
    return Y
end
function computeYlm(θ::Real, ϕ::Real, lmax::Integer,
    m_range::SHMRange, SHType::HarmonicType = ComplexHarmonics())

    computeYlm(θ, ϕ, lmax, nothing, m_range, SHType)
end

function computeYlm(θ::Pole, ϕ::Real, lmax::Integer, m::Union{Integer,Nothing} = nothing,
    m_range::SHMRange = FullRange,
    SHType::HarmonicType = ComplexHarmonics())

    P = computePlmcostheta(θ, lmax, _maybeabs(m))
    Y = _computeYlm(P, θ, ϕ, lmax, m, m_range, SHType)
    return Y
end
function computeYlm(θ::Pole, ϕ::Real, lmax::Integer, m_range::SHMRange,
    SHType::HarmonicType = ComplexHarmonics())

    computeYlm(θ, ϕ, lmax, nothing, m_range, SHType)
end

function computeYlm(θ::Pole, lmax::Integer, m_range::SHMRange = FullRange,
    SHType::HarmonicType = ComplexHarmonics())
    computeYlm(θ, 0, lmax, nothing, m_range, SHType)
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
function sphericalharmonic(θ::Real, ϕ::Real; l::Integer, m::Integer,
    SHType::HarmonicType = ComplexHarmonics(), coeff = nothing)

    sphericalharmonic(θ, ϕ, l, m, SHType, coeff)
end
function sphericalharmonic(θ::Real, ϕ::Real, l::Integer, m::Integer, SHType::HarmonicType = ComplexHarmonics(),
    coeff = nothing)

    P = associatedLegendre(θ, l, abs(m), coeff)
    if m == 0
        return _invsqrt2 * P
    end
    phasepos, phaseneg = phase(SHType, FullRange, abs(m), ϕ, _invsqrt2, (-1)^Int(m))
    norm = m >= 0 ? phasepos : phaseneg
    norm * P
end

end
