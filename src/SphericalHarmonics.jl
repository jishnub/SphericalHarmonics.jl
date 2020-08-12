module SphericalHarmonics

using SphericalHarmonicModes
using SphericalHarmonicArrays

export computeYlm, computeYlm!, computePlmcostheta, computePlmcostheta!

Base.@irrational invsqrt2 0.7071067811865476 1/√(big(2))
Base.@irrational invsqrt2pi 0.3989422804014327 1/√(2*big(pi))
Base.@irrational sqrt3by4pi 0.4886025119029199 √(3/(4big(pi)))
Base.@irrational sqrt3by2pi 0.690988298942671 √(3/(2big(pi)))

"""
	SphericalHarmonics.Pole <: Real

Supertype of [`NorthPole`](@ref) and [`SouthPole`](@ref).
"""
abstract type Pole <: Real end
"""
	SphericalHarmonics.NorthPole <: SphericalHarmonics.Pole

The angle ``\\theta = 0`` in spherical polar coordinates.
"""
struct NorthPole <: Pole end
"""
	SphericalHarmonics.SouthPole <: SphericalHarmonics.Pole

The angle ``\\theta = π`` in spherical polar coordinates.
"""
struct SouthPole <: Pole end

Base.cos(::NorthPole) = one(Float64)
Base.cos(::SouthPole) = -one(Float64)
Base.sin(::Pole) = zero(Float64)

for DT in [:zero, :one]
	@eval Base.$DT(::Type{<:Pole}) = $DT(Float64)
	@eval Base.$DT(x::Pole) = $DT(typeof(x))
end

Base.promote_rule(::Type{<:Pole}, T::Type{<:Real}) = promote_type(Float64, T)

# Return the value of θ corresponding to the poles
Base.Float64(::SouthPole) = Float64(pi)
Base.Float64(::NorthPole) = zero(Float64)

Base.float(p::Pole) = Float64(p)

const SHMRange = Union{Type{FullRange}, Type{ZeroTo}}

sizeP(maxDegree::Int) = div((maxDegree + 1) * (maxDegree + 2), 2)
sizeY(maxDegree::Int, m_range::Type{FullRange} = FullRange) = (maxDegree + 1) * (maxDegree + 1)
sizeY(maxDegree::Int, ::Type{ZeroTo}) = sizeP(maxDegree)

index_p(l::Int, m::Int) = m + div(l*(l+1), 2) + 1
function index_p(l::Integer, m::AbstractUnitRange{<:Integer})
	index_p(Int(l),Int(first(m))):index_p(Int(l),Int(last(m)))
end
index_p(l::Integer) = index_p(l, ZeroTo(l))

index_y(l::Int, m::Int, m_range::Type{FullRange} = FullRange) = m + l + l^2 + 1
index_y(l::Int, m::Int, ::Type{ZeroTo}) = index_p(l, m)
function index_y(l::Integer, m::AbstractUnitRange{<:Integer}, m_range)
	index_y(Int(l),Int(first(m)),m_range):index_y(Int(l),Int(last(m)),m_range)
end
index_y(l::Integer) = index_y(l, FullRange(l), FullRange)

"""
	SphericalHarmonics.compute_coefficients(L::Integer)

Precompute coefficients ``a_l^m`` and ``b_l^m`` for all ``l ≤ L`` and ``0 ≤ m ≤ l``.
"""
function compute_coefficients(L::Integer)
	T = typeof(sqrt(L^2))
	coeff = zeros(T, 2, sizeP(Int(L)))

	@inbounds for l in 2:L

		Bden = 1/√(4 * (l-1)^2 - 1)
		
		for m in 0:(l-2)
			lmind = index_p(Int(l), Int(m))
			coeff[1, lmind] = √((4l^2 - 1) / (l^2 - m^2))
			coeff[2, lmind] = -√((l-1)^2 - m^2) * Bden
		end
	end
	return coeff
end

"""
	SphericalHarmonics.allocate_p([T::Type = Float64], L::Integer)

Allocate an array large enough to store an entire set of Associated Legendre
Polynomials ``\\bar{P}_l^m(x)`` of maximum degree ``L``.
"""
allocate_p(T::Type, L::Integer) = SHVector{T}(undef, ML(ZeroTo(L), ZeroTo))
allocate_p(L::Integer) = allocate_p(Float64, L)

"""
	SphericalHarmonics.allocate_y([T::Type = ComplexF64], L::Integer)

Allocate an array large enough to store an entire set of spherical harmonics
``Y_{l,m}(θ,ϕ)`` of maximum degree ``L``.
"""
allocate_y(T::Type, L::Integer, m_range = FullRange) = SHVector{T}(undef, ML(ZeroTo(L), m_range))
allocate_y(L::Integer, m_range = FullRange) = allocate_y(ComplexF64, L, m_range)

function checksize(sz, minsize)
	@assert sz >= minsize "array needs to have a minimum size of $minsize, received size $sz"
end

function _computePlmcostheta!(P, costheta, sintheta, L, coeff, ::Type{T}) where {T<:Real}
	@inbounds P[index_p(0, 0)] = invsqrt2pi

	@inbounds if (L > 0)
		P[index_p(1, 0)] = sqrt3by2pi * costheta
		P11 = -(sqrt3by4pi * sintheta)
		P[index_p(1, 1)] = P11
		temp = T(P11)

		for l in 2:Int(L)
			for m in 0:(l-2)
				lmind = index_p(l, m)
				P[lmind] = coeff[1,lmind] *(costheta * P[index_p(l - 1, m)]
						     + coeff[2,lmind] * P[index_p(l - 2, m)])
			end
			P[index_p(l, l - 1)] = costheta * √(T(2l + 1)) * temp
			temp = -√(1 + 1 / 2T(l)) * sintheta * temp
			P[index_p(l, l)] = temp
		end
	end
end

"""
	computePlmx!(P::AbstractVector{<:Real}, x::Real, lmax::Integer, coeff::AbstractMatrix)

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_l^m(x)``
using the given coefficients, and store in the array `P`. The matrix `coeff` may be computed 
using [`compute_coefficients`](@ref).

The argument `x` needs to lie in ``-1 ≤ x ≤ 1``. The function implicitly assumes that 
``x = \\cos(\\theta)`` where ``0 ≤ \\theta ≤ π``.
"""
function computePlmx!(P::AbstractVector{<:Real}, x::THETA, L::Integer, coeff::AbstractMatrix) where {THETA<:Real}

	checksize(size(coeff, 2), sizeP(Int(L)))
	checksize(length(P), sizeP(Int(L)))

	-1 <= x <= 1 || throw(ArgumentError("x needs to lie in [-1,1]"))

	T = promote_type(Float64, promote_type(THETA, eltype(coeff)))

	_computePlmcostheta!(P, x, √(1-x^2), L, coeff, T)
	
	return P
end

"""
	computePlmcostheta!(P::AbstractVector{<:Real}, θ::Real, lmax::Integer, coeff::AbstractMatrix)
	computePlmcostheta!(P::AbstractVector{<:Real}, θ::SphericalHarmonics.Pole, lmax::Integer)

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_l^m(\\cos(θ))``
using the given coefficients, and store in the array `P`. The matrix `coeff` may be computed 
using [`compute_coefficients`](@ref).
"""
function computePlmcostheta!(P::AbstractVector{<:Real}, θ::THETA, L::Integer, coeff::AbstractMatrix) where {THETA<:Real}

	checksize(size(coeff, 2), sizeP(Int(L)))
	checksize(length(P), sizeP(Int(L)))

	T = promote_type(Float64, promote_type(THETA, eltype(coeff)))

	_computePlmcostheta!(P, cos(θ), sin(θ), L, coeff, T)
	
	return P
end

# This method is needed for ambiguity resolution
function computePlmcostheta!(P::AbstractVector{<:Real}, θ::Pole, L::Integer, coeff::AbstractMatrix)
	computePlmcostheta!(P, θ, L)
	return P
end

function computePlmcostheta!(P::AbstractVector{R}, ::NorthPole, L::Integer) where {R<:Real}
	checksize(length(P), sizeP(Int(L)))
	
	fill!(P, zero(R))

	T = promote_type(promote_type(Float64, R), float(typeof(L)))

	norm = T(invsqrt2pi)

	for l in ZeroTo(L)
		P[index_p(l, 0)] = norm * √(T(2l + 1))
	end
	return P
end

function computePlmcostheta!(P::AbstractVector{R}, ::SouthPole, L::Integer) where {R<:Real}
	checksize(length(P), sizeP(Int(L)))
	
	fill!(P, zero(R))

	T = promote_type(promote_type(Float64, R), float(typeof(L)))

	norm = -T(invsqrt2pi)
	
	for l in ZeroTo(L)
		norm *= -1
		P[index_p(l, 0)] = norm * √(T(2l + 1))
	end
	return P
end

function computePlmcostheta(θ::THETA, L::Integer) where {THETA<:Real}
	T = promote_type(float(typeof(L)), float(THETA))
	P = allocate_p(T, L)
	coeff = compute_coefficients(L)
	computePlmcostheta!(P, θ, L, coeff)
	return P
end

function computePlmcostheta(θ::Pole, L::Integer)
	T = promote_type(float(typeof(L)), Float64)
	P = allocate_p(T, L)
	computePlmcostheta!(P, θ, L)
	return P
end

function computePlmx(x::X, L::Integer) where {X<:Real}
	T = promote_type(float(typeof(L)), float(X))
	P = allocate_p(T, L)
	coeff = compute_coefficients(L)
	computePlmx!(P, x, L, coeff)
	return P
end

"""
	computePlmcostheta(θ::Real; lmax::Integer)
	computePlmcostheta(θ::Real, lmax::Integer)

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_l^m(\\cos(θ))`` where
``0 ≤ l ≤ l_\\mathrm{max}`` and ``0 ≤ m ≤ l`` for colatitude ``\\theta``. 

The polynomials are normalized as 

```math
\\bar{P}_{\\ell m} = \\sqrt{\\frac{(2\\ell + 1)(\\ell-m)!}{2\\pi (\\ell+m)!}} P_{\\ell m}.
```

Returns an `SHVector` that may be 
indexed using `(l,m)` pairs aside from the canonical indexing with `Int`s.

The precision of the result may be increased by using arbitrary-precision arguments.

# Examples
```jldoctest
julia> P = computePlmcostheta(pi/2, lmax = 1)
3-element SHArray(::Array{Float64,1}, (ML(0:1, 0:1),)):
  0.3989422804014327
  4.231083042742082e-17
 -0.4886025119029199

julia> P[(0,0)]
0.3989422804014327

julia> P = computePlmcostheta(big(pi)/2, lmax = big(1)) # Arbitrary precision
3-element SHArray(::Array{BigFloat,1}, (ML(0:1, 0:1),)):
  0.3989422804014326779399460599343818684758586311649346576659258296706579258993008
  3.789785583114350800838137317730900078444216599640987847808409161681770236721676e-78
 -0.4886025119029199215863846228383470045758856081942277021382431574458410003616367
```
"""
computePlmcostheta(θ::Real; lmax::Integer) = computePlmcostheta(θ, lmax)

"""
	computePlmx(x::Real; lmax::Integer)

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_l^m(x)`` where
``0 ≤ l ≤ l_\\mathrm{max}`` and ``0 ≤ m ≤ l``.

The argument `x` needs to lie in ``-1 ≤ x ≤ 1``. The function implicitly assumes that 
``x = \\cos(\\theta)`` where ``0 ≤ \\theta ≤ π``.
"""
computePlmx(x::Real; lmax::Integer) = computePlmx(x, lmax)

abstract type HarmonicType end
struct RealHarmonics <: HarmonicType end
struct ComplexHarmonics <: HarmonicType end

function computeYlm!(Y::AbstractVector, P::AbstractVector{R}, θ::Pole, 
	ϕ::Real, L::Integer, m_range::SHMRange = FullRange, 
	SHType::HarmonicType = ComplexHarmonics()) where {R<:Real}

	checksize(length(P), sizeP(Int(L)))
	checksize(length(Y), sizeY(L, m_range))

	fill!(Y, zero(eltype(Y)))

	for l in ZeroTo(L)
		Y[index_y(l, 0, m_range)] = P[index_p(l, 0)] * invsqrt2
	end

	return Y
end

function phase(::RealHarmonics, ::Type{FullRange}, m, ϕ, ::Any, ::Any)
	S, C = sincos(m*ϕ)
	ep = C
	em = S
	ep, em
end

function phase(::ComplexHarmonics, ::Type{FullRange}, m, ϕ, norm, CSphase)
	ep = cis(m*ϕ) * norm
	em = CSphase * conj(ep)
	ep, em
end

function phase(::RealHarmonics, ::Type{ZeroTo}, m, ϕ, ::Any, ::Any)
	cos(m*ϕ)
end

function phase(::ComplexHarmonics, ::Type{ZeroTo}, m, ϕ, norm, CSphase)
	cis(m*ϕ) * norm
end

function fill_m_maybenegm!(Y, P, L, m, ϕ, CSphase, ::Type{FullRange}, SHType)
	
	phasempos, phasemneg = phase(SHType, FullRange, m, ϕ, invsqrt2, CSphase)

	for l in m:Int(L)
		Plm = P[index_p(l, m)]
		Y[index_y(l, -m)] = phasemneg * Plm
		Y[index_y(l,  m)] = phasempos * Plm
	end
end

function fill_m_maybenegm!(Y, P, L, m, ϕ, CSphase, ::Type{ZeroTo}, SHType)
	
	phasem = phase(SHType, ZeroTo, m, ϕ, invsqrt2, CSphase)

	for l in m:Int(L)
		ind = index_p(l, m)
		Y[ind] = phasem * P[ind]
	end
end

"""
	computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Real, ϕ::Real, L::Integer, [m_range = SphericalHarmonics.FullRange, [SHType = SphericalHarmonics.ComplexHarmonics()]])

Compute an entire set of spherical harmonics ``Y_{l,m}(θ,ϕ)``
using the precomputed associated Legendre Polynomials ``\\bar{P}_l^m(x = \\cos(θ))``,
and store in the array `Y`. The array `P` may be computed using [`computePlmcostheta`](@ref).

The optional argument `m_range` decides if the spherical harmonics for negative `m` values are computed. 
By default the functions for all values of `m` are evaluated. Setting `m_range` to `SphericalHarmonics.ZeroTo` would result 
in only functions for `m ≥ 0` being evaluated.

The optional argument `SHType` may be used to choose between real and complex harmonics.
To compute real spherical harmonics, set this to `RealHarmonics()`.
"""
function computeYlm!(Y::AbstractVector, P::AbstractVector{R}, θ::Real,
	ϕ::Real, L::Integer, m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics()) where {R<:Real}

	checksize(length(P), sizeP(Int(L)))
	checksize(length(Y), sizeY(Int(L), m_range))

	for l in ZeroTo(L)
		Y[index_y(l, 0, m_range)] = P[index_p(l, 0)] * invsqrt2
	end

	CSphase = 1
	@inbounds for m in 1:Int(L)
		CSphase *= -1
		fill_m_maybenegm!(Y, P, L, m, ϕ, CSphase, m_range, SHType)
	end

	return Y
end

function computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Real, 
	ϕ::Real; lmax::Integer, m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())

	computeYlm!(Y, P, θ, ϕ, lmax, m_range, SHType)
end

function computeYlm!(Y::AbstractVector, θ::Real, ϕ::Real, L::Integer, 
	m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())
	
	P = computePlmcostheta(θ, L)
	computeYlm!(Y, P, θ, ϕ, L, m_range, SHType)
end

"""
	computeYlm!(Y::AbstractVector, θ::Real, ϕ::Real; lmax::Integer, [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])

Compute an entire set of spherical harmonics ``Y_{l,m}(θ,ϕ)`` for ``0 ≤ l ≤ l_\\mathrm{max}``,
and store them in the array `Y`.

The optional argument `m_range` decides if the spherical harmonics for negative `m` values are computed. 
By default the functions for all values of `m` are evaluated. Setting `m_range` to `SphericalHarmonics.ZeroTo` would result 
in only functions for `m ≥ 0` being evaluated.

The optional argument `SHType` may be used to choose between real and complex harmonics.
To compute real spherical harmonics, set this to `SphericalHarmonics.RealHarmonics()`.
"""
function computeYlm!(Y::AbstractVector, θ::Real, ϕ::Real; lmax::Integer, 
	m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())
	computeYlm!(Y, θ, ϕ, lmax, m_range, SHType)
end

eltypeY(::Type{R}, ::ComplexHarmonics) where {R} = Complex{R}
eltypeY(::Type{R}, ::RealHarmonics) where {R} = R

function _computeYlm(P, θ, ϕ, L, m_range, SHType)
	T = eltypeY(promote_type(eltype(P), typeof(ϕ)), SHType)
	Y = allocate_y(T, L, m_range)
	computeYlm!(Y, P, θ, ϕ, L, m_range, SHType)
	return Y
end

"""
	computeYlm(θ::Real, ϕ::Real; lmax::Integer, [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])
	computeYlm(θ::SphericalHarmonics.Pole; lmax::Integer, [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])

Compute an entire set of spherical harmonics ``Y_{l,m}(θ,ϕ)`` for 
``0 ≤ l ≤ l_\\mathrm{max}`` and ``-l ≤ m ≤ l``, for colatitude ``\\theta`` and 
azimuth ``\\phi``.

Returns an `SHVector` that may be 
indexed using `(l,m)` pairs aside from the canonical indexing with `Int`s.

The optional argument `m_range` decides if the spherical harmonics for negative `m` values are computed. 
By default the functions for all values of `m` are evaluated. Setting `m_range` to `SphericalHarmonics.ZeroTo` would result 
in only functions for `m ≥ 0` being evaluated.

The optional argument `SHType` may be used to choose between real and complex harmonics.
To compute real spherical harmonics, set this to `SphericalHarmonics.RealHarmonics()`.

The precision of the result may be increased by using arbitrary-precision arguments.

# Examples
```jldoctest
julia> Y = computeYlm(pi/2, 0, lmax = 1)
4-element SHArray(::Array{Complex{Float64},1}, (ML(0:1, -1:1),)):
     0.2820947917738782 + 0.0im
     0.3454941494713355 - 0.0im
 2.9918275112863375e-17 + 0.0im
    -0.3454941494713355 - 0.0im

julia> Y[(1,-1)] # index using (l,m)
0.3454941494713355 - 0.0im

julia> Y = computeYlm(big(pi)/2, big(0), lmax = big(1)) # Arbitrary precision
4-element SHArray(::Array{Complex{BigFloat},1}, (ML(0:1, -1:1),)):
    0.2820947917738781434740397257803862929220253146644994284220428608553212342207478 + 0.0im
    0.3454941494713354792652446460318896831393773703262433134867073548945156550201567 - 0.0im
 2.679783085063171668225419916118067917387251852939708540164955895366691604430101e-78 + 0.0im
   -0.3454941494713354792652446460318896831393773703262433134867073548945156550201567 - 0.0im

julia> computeYlm(SphericalHarmonics.NorthPole(), 0, lmax = 1)
4-element SHArray(::Array{Complex{Float64},1}, (ML(0:1, -1:1),)):
 0.2820947917738782 + 0.0im
                0.0 + 0.0im
   0.48860251190292 + 0.0im
                0.0 + 0.0im

julia> Y = computeYlm(pi/2, pi/3, lmax = 1, m_range = SphericalHarmonics.ZeroTo, SHType = SphericalHarmonics.RealHarmonics())
3-element SHArray(::Array{Float64,1}, (ML(0:1, 0:1),)):
  0.2820947917738782
  2.9918275112863375e-17
 -0.24430125595146002
```
"""
function computeYlm(θ::THETA, ϕ::PHI, L::I, 
	m_range::SHMRange = FullRange,
	SHType::HarmonicType = ComplexHarmonics()) where {THETA<:Real,PHI<:Real,I<:Integer}
	
	T = promote_type(promote_type(THETA, PHI), float(I))
	P = computePlmcostheta(T(θ), L)
	Y = _computeYlm(P, T(θ), T(ϕ), L, m_range, SHType)
	return Y
end

function computeYlm(θ::Pole, ϕ::Real, L::Integer,
	m_range::SHMRange = FullRange,
	SHType::HarmonicType = ComplexHarmonics())

	P = computePlmcostheta(θ, L)
	Y = _computeYlm(P, θ, ϕ, L, m_range, SHType)
	return Y
end

function computeYlm(θ::Pole, L::Integer, m_range::SHMRange = FullRange,
	SHType::HarmonicType = ComplexHarmonics())
	computeYlm(θ, 0, L, m_range, SHType)
end

function computeYlm(θ::Real, ϕ::Real; lmax::Integer, m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())
	computeYlm(θ, ϕ, lmax, m_range, SHType)
end
function computeYlm(θ::Pole; lmax::Integer, m_range::SHMRange = FullRange, SHType::HarmonicType = ComplexHarmonics())
	computeYlm(θ, 0, lmax, m_range, SHType)
end

end
