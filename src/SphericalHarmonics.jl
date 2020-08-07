module SphericalHarmonics

using SphericalHarmonicModes
using SphericalHarmonicArrays

export computeYlm, computeYlm!, computePlmcostheta, computePlmcostheta!

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

Base.one(::Type{<:Pole}) = one(Float64)
Base.zero(::Type{<:Pole}) = zero(Float64)

Base.promote_rule(::Type{<:Pole},::Type{Float64}) = Float64
Base.promote_rule(::Type{<:Pole},T::Type{<:Real}) = promote_rule(Float64,T)

# Return the value of θ corresponding to the poles
Base.Float64(::SouthPole) = Float64(pi)
Base.Float64(::NorthPole) = zero(Float64)

Base.float(p::Pole) = Float64(p)

sizeP(maxDegree::Integer) = sizeP(Int(maxDegree))
sizeP(maxDegree::Int) = div((maxDegree + 1) * (maxDegree + 2), 2)

sizeY(maxDegree::Integer) = sizeY(Int(maxDegree))
sizeY(maxDegree::Int) = (maxDegree + 1) * (maxDegree + 1)

index_p(l::Int, m::Int) = m + div(l*(l+1), 2) + 1
index_p(l::Integer, m::Integer) = index_p(Int(l), Int(m))
index_p(l::Integer, m::AbstractUnitRange) = index_p(l,first(m)):index_p(l,last(m))
index_p(l::Integer) = index_p(l,-l:l)

index_y(l::Int, m::Int) = m + l + l^2 + 1
index_y(l::Integer, m::Integer) = index_y(Int(l), Int(m))
index_y(l::Integer, m::AbstractUnitRange) = index_y(l,first(m)):index_y(l,last(m))
index_y(l::Integer) = index_y(l,-l:l)

"""
	SphericalHarmonics.compute_coefficients(L::Integer)

Precompute coefficients ``a_l^m`` and ``b_l^m`` for all ``l ≤ L`` and ``0 ≤ m ≤ l``.
"""
function compute_coefficients(L::Integer)
	TL2 = typeof(L^2)
	T = typeof(sqrt(L^2))
	modesPlm = ML(ZeroTo(L), ZeroTo)
	coeff = zeros(T, 2, length(modesPlm))

	@inbounds for l in 2:L

		Bden = 1/√(4 * (l-1)^2 - 1)
		
		for m in 0:(l-2)
			lmind = index_p(l, m)
			coeff[1, lmind] = √((4l^2 - 1) / (l^2 - m^2))
			coeff[2, lmind] = -√((l-1)^2 - m^2) * Bden
		end
	end
	return coeff
end

"""
	SphericalHarmonics.allocate_p(T::Type, L::Integer)

Allocate an array large enough to store an entire set of Associated Legendre
Polynomials ``\\bar{P}_l^m(x)`` of maximum degree ``L``.
"""
allocate_p(T::Type, L::Integer) = SHVector{T}(undef, ML(ZeroTo(L), ZeroTo))

"""
	SphericalHarmonics.allocate_y(T::Type, L::Integer)

Allocate an array large enough to store an entire set of spherical harmonics
``Y_{l,m}(θ,φ)`` of maximum degree ``L``.
"""
allocate_y(T::Type, L::Integer) = SHVector{T}(undef, ML(ZeroTo(L)))

function checksize(sz, minsize)
	@assert sz >= minsize "array needs to have a minimum size of $minsize, received size $sz"
end

function _computePlmcostheta!(P, costheta, sintheta, L, coeff, ::Type{T}) where {T<:Real}
	temp = 1/√(2 * T(π))
	@inbounds P[index_p(0, 0)] = temp

	@inbounds if (L > 0)
		P[index_p(1, 0)] = costheta * √(T(3)) * temp
		temp = -√(T(3)/2) * sintheta * temp
		P[index_p(1, 1)] = temp

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

	checksize(size(coeff, 2), sizeP(L))
	checksize(length(P), sizeP(L))

	-1 <= x <= 1 || throw(ArgumentError("x needs to lie in [-1,1]"))

	T = promote_type(Float64, promote_type(THETA, eltype(coeff)))

	_computePlmcostheta!(P, x, √(1-x^2), L, coeff, T)
	
	return P
end

"""
	computePlmcostheta!(P::AbstractVector{<:Real}, θ::Real, lmax::Integer, coeff::AbstractMatrix)
	computePlmcostheta!(P::AbstractVector{<:Real}, θ::SphericalHarmonics.Pole, lmax::Integer)

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_l^m(cos(θ))``
using the given coefficients, and store in the array `P`. The matrix `coeff` may be computed 
using [`compute_coefficients`](@ref).
"""
function computePlmcostheta!(P::AbstractVector{<:Real}, θ::THETA, L::Integer, coeff::AbstractMatrix) where {THETA<:Real}

	checksize(size(coeff, 2), sizeP(L))
	checksize(length(P), sizeP(L))

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
	checksize(length(P), sizeP(L))
	
	fill!(P, zero(R))

	T = promote_type(promote_type(Float64, R), float(typeof(L)))

	norm = 1/√(2 * T(π))

	for l in ZeroTo(L)
		P[index_p(l, 0)] = norm * √(T(2l + 1))
	end
	return P
end

function computePlmcostheta!(P::AbstractVector{R}, ::SouthPole, L::Integer) where {R<:Real}
	checksize(length(P), sizeP(L))
	
	fill!(P, zero(R))

	T = promote_type(promote_type(Float64, R), float(typeof(L)))

	norm = -1/√(2 * T(π))
	
	for l in ZeroTo(L)
		norm *= -1
		P[index_p(l, 0)] = norm * √(T(2l + 1))
	end
	return P
end

"""
	computePlmcostheta(θ::Real; lmax::Integer)

Compute an entire set of normalized Associated Legendre Polynomials ``\\bar{P}_l^m(cos(θ))`` where
``0 ≤ l ≤ l_\\mathrm{max}`` and ``0 ≤ m ≤ l``. The polynomials are normalized as 

```math
\\bar{P}_{\\ell m} = \\sqrt{\\frac{(2\\ell + 1)(\\ell-m)!}{2\\pi (\\ell+m)!}} P_{\\ell m}.
```

Returns an `SHVector` that may be 
indexed using `(l,m)` pairs aside from the canonical indexing with `Int`s.

# Examples
```jldoctest
julia> computePlmcostheta(pi/2, 1)
3-element SHArray(::Array{Float64,1}, (ML(0:1, 0:1),)):
  0.3989422804014327
  4.231083042742082e-17
 -0.4886025119029199
```
"""
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

computePlmcostheta(x::Real; lmax::Integer) = computePlmcostheta(x, lmax)

"""
	computeYlm!(Y::AbstractVector{<:Complex}, P::AbstractVector{<:Real}, θ::Real, φ::Real, L::Integer)

Compute an entire set of spherical harmonics ``Y_{l,m}(θ,φ)``
using the precomputed associated Legendre Polynomials ``\\bar{P}_l^m(x = \\cos(θ))``,
and store in the array `Y`. The array `P` may be computed using [`computePlmcostheta`](@ref).
"""
function computeYlm!(Y::AbstractVector{<:Complex}, P::AbstractVector{<:Real}, θ::Pole, 
	ϕ::Real, L::Integer)

	checksize(length(P), sizeP(L))
	checksize(length(Y), sizeY(L))

	fill!(Y, zero(eltype(Y)))

	norm = 1/√2

	@inbounds for l in ZeroTo(L)
		Y[index_y(l, 0)] = P[index_p(l, 0)] * norm
	end

	return Y
end

function computeYlm!(Y::AbstractVector{<:Complex}, P::AbstractVector{<:Real}, θ::Real,
	ϕ::Real, L::Integer)

	checksize(length(P), sizeP(L))
	checksize(length(Y), sizeY(L))

	norm = 1/√2
	for l in 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * norm
	end

	sig = 1
	@inbounds for m in 1:L
		sig *= -1
		ep = cis(m*ϕ) * norm
		em = sig * conj(ep)
		for l in m:L
			p = P[index_p(l, m)]
			Y[index_y(l, -m)] = em * p
			Y[index_y(l,  m)] = ep * p
		end
	end

	return Y
end

function computeYlm!(Y::AbstractVector{<:Complex}, P::AbstractVector{<:Real}, θ::Pole, 
	ϕ::Real; lmax::Integer)

	computeYlm!(Y, P, θ, ϕ, lmax)
end

"""
	computeYlm!(Y::AbstractVector{<:Complex}, θ::Real, φ::Real; lmax::Integer)

Compute an entire set of spherical harmonics ``Y_{l,m}(θ,φ)`` for ``0 ≤ l ≤ l_\\mathrm{max}``,
and store them in the array `Y`.
"""
function computeYlm!(Y::AbstractVector{<:Complex}, θ::Real, ϕ::Real, L::Integer)
	P = computePlmcostheta(L, θ)
	computeYlm!(Y, P, θ, ϕ, L)
end

function computeYlm!(Y::SHVector{<:Complex}, θ::Real, ϕ::Real)
	L = last(l_range(first(SphericalHarmonicArrays.shmodes(Y))))
	@inbounds computeYlm!(Y, θ, ϕ, L)
end

function computeYlm!(Y::AbstractVector{<:Complex}, θ::Real, φ::Real; lmax::Integer)
	computeYlm!(Y, θ, ϕ, lmax)
end

function _computeYlm(P, θ, ϕ, L)
	Y = allocate_y(Complex{promote_type(eltype(P), typeof(ϕ))}, L)
	computeYlm!(Y, P, θ, ϕ, L)
	return Y
end

"""
	computeYlm(θ::Real, ϕ::Real; lmax::Integer)

Compute an entire set of spherical harmonics ``Y_{l,m}(θ,φ)`` for 
``0 ≤ l ≤ l_\\mathrm{max}`` and ``-l ≤ m ≤ l``.

Returns an `SHVector` that may be 
indexed using `(l,m)` pairs aside from the canonical indexing with `Int`s.

The precision of the result is set as a combination of the types of all the arguments.

# Examples
```jldoctest
julia> Y = computeYlm(pi/2, 0, 1)
4-element SHArray(::Array{Complex{Float64},1}, (ML(0:1, -1:1),)):
   0.28209479177387814 + 0.0im
   0.34549414947133544 - 0.0im
 2.991827511286337e-17 + 0.0im
  -0.34549414947133544 - 0.0im

julia> Y[(1,-1)] # l = 1, m = -1
0.34549414947133544 - 0.0im

julia> Y = computeYlm(big(pi)/2, 0, 1) # Arbitrary precision
4-element SHArray(::Array{Complex{BigFloat},1}, (ML(0:1, -1:1),)):
    0.2820947917738781184660094048589327597216759680130934754730145983228317949194716 + 0.0im
    0.3454941494713354486367877668779846612877814388031855602538601251440676669552971 - 0.0im
 2.679783085063171430659542774823417668417011548855617520042112930055124837543658e-78 + 0.0im
   -0.3454941494713354486367877668779846612877814388031855602538601251440676669552971 - 0.0im

julia> computeYlm(SphericalHarmonics.NorthPole(), 0, 1)
4-element SHArray(::Array{Complex{Float64},1}, (ML(0:1, -1:1),)):
 0.28209479177387814 + 0.0im
                 0.0 + 0.0im
  0.4886025119029199 + 0.0im
                 0.0 + 0.0im 
```
"""
function computeYlm(θ::THETA, ϕ::PHI, L::I) where {THETA<:Real,PHI<:Real,I<:Integer}
	YT = promote_type(promote_type(THETA, PHI), float(I))
	P = computePlmcostheta(YT(θ), L)
	Y = _computeYlm(P, θ, ϕ, L)
	return Y
end
function computeYlm(θ::Pole, ϕ::Real, L::Integer)
	P = computePlmcostheta(θ, L)
	Y = _computeYlm(P, θ, ϕ, L)
	return Y
end
computeYlm(θ::Pole, L::Integer) = computeYlm(θ, 0, L)

computeYlm(θ::Real, ϕ::Real; lmax::Integer) = computeYlm(θ, ϕ, lmax)

end
