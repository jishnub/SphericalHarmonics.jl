using StaticArrays, LinearAlgebra

const SVec3 = SVector{3}

export compute_y, compute_y!, index_y, compute_p, compute_p!

const ContiguousArrayLike{T,N} = 
	Union{Array{T,N}, Base.FastContiguousSubArray{T,N,<:Array{T}}}

const VectorLike{T} = ContiguousArrayLike{T,1}

"""
	sizeP(maxDegree)

Return the size of the set of Associated Legendre Polynomials ``P_l^m(x)`` of
degree less than or equal to the given maximum degree
"""
sizeP(maxDegree) = div((maxDegree + 1) * (maxDegree + 2), 2)

"""
	sizeY(maxDegree)

Return the size of the set of spherical harmonics ``Y_{l,m}(θ,φ)`` of
degree less than or equal to the given maximum degree
"""
sizeY(maxDegree) = (maxDegree + 1) * (maxDegree + 1)

"""
	index_p(l,m)

Return the index into a flat array of Associated Legendre Polynomials ``P_l^m``
for the given indices ``(l,m)``.
``P_l^m`` are stored in l-major order i.e. [P(0,0), [P(1,0), P(1,1), P(2,0), ...]
"""
index_p(l,m) = m + div(l*(l+1), 2) + 1

"""
	index_y(l,m)

Return the index into a flat array of spherical harmonics ``Y_{l,m}``
for the given indices ``(l,m)``.
``Y_{l,m}`` are stored in l-major order i.e.
[Y(0,0), [Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), ...]
"""
index_y(l,m) = m + l + (l*l) + 1

"""
TODO: documentation
"""
struct ALPCoefficients
	A::Array{Float64}
	B::Array{Float64}
end

ALPCoefficients(maxDegree::Int) =
	ALPCoefficients( zeros(sizeP(maxDegree)),
						  zeros(sizeP(maxDegree)) )

"""
	compute_coefficients(L)

Precompute coefficients ``a_l^m`` and ``b_l^m`` for all l <= L, m <= l
"""
function compute_coefficients(L::Integer)
	coeff = ALPCoefficients(L)
	@inbounds for l in 2:L, m in 0:(l-2)

		coeff.A[index_p(l, m)] = sqrt((4l^2 - 1.0) / (l^2 - m^2))
		coeff.B[index_p(l, m)] = -sqrt(((l-1)^2 - m^2) / (4 * (l-1)^2 - 1))
	end
	return coeff
end

"""
	allocate_p(L)

Create an array large enough to store an entire set of Associated Legendre
Polynomials ``P_l^m(x)`` of maximum degree L.
"""
allocate_p(L::Integer) = zeros(sizeP(L))

"""
	allocate_y(L)

Create an array large enough to store an entire set of spherical harmonics
``Y_{l,m}(θ,φ)`` of maximum degree L.
"""
allocate_y(L::Integer) = zeros(ComplexF64,sizeY(L))

"""
	compute_p!(L, x, coeff, P)

Compute an entire set of Associated Legendre Polynomials ``P_l^m(x)``
using the given coefficients, and store in the array P.
"""
function compute_p!(L::Integer, x::Real, 
	coeff::ALPCoefficients,P::VectorLike{<:Real})

	@assert length(coeff.A) >= sizeP(L)
	@assert length(coeff.B) >= sizeP(L)
	@assert length(P) >= sizeP(L)

	sintheta = sqrt(1.0 - x^2)
	temp = √(1/2π)
	P[index_p(0, 0)] = temp

	if (L > 0)
		P[index_p(1, 0)] = x * √3 * temp
		temp = -√(3/2) * sintheta * temp
		P[index_p(1, 1)] = temp

		@inbounds for l in 2:L
			@inbounds for m in 0:(l-2)
				P[index_p(l, m)] = coeff.A[index_p(l, m)] *(x * P[index_p(l - 1, m)]
						     + coeff.B[index_p(l, m)] * P[index_p(l - 2, m)])
			end
			P[index_p(l, l - 1)] = x * sqrt(2 * (l - 1) + 3) * temp
			temp = -sqrt(1.0 + 0.5 / l) * sintheta * temp
			P[index_p(l, l)] = temp
		end
	end
	return P
end

"""
	compute_p(L, x)

Compute an entire set of Associated Legendre Polynomials ``P_l^m(x)`` where
``0 ≤ l ≤ L`` and ``0 ≤ m ≤ l``. Assumes ``|x| ≤ 1``.
"""
function compute_p(L::Integer, x::Real)
	P = allocate_p(L)
	coeff = compute_coefficients(L)
	compute_p!(L, x, coeff, P)
	return P
end

"""
	compute_y!(L, x, φ, P, Y)

Compute an entire set of spherical harmonics ``Y_{l,m}(θ,φ)``
using the given Associated Legendre Polynomials ``P_l^m(cos θ)``
and store in array Y
"""
function compute_y!(L::Integer, x::Real, ϕ::Real,
	P::VectorLike{<:Real},Y::VectorLike{<:Complex})

	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)

	@inbounds for l in 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * 0.5 * √2
	end

	sig = 1
	@inbounds for m in 1:L
		sig *= -1
		ep = cis(m*ϕ) / √2
		em = sig * conj(ep)
		@inbounds for l in m:L
			p = P[index_p(l,m)]
			Y[index_y(l, -m)] = em * p
			Y[index_y(l,  m)] = ep * p
		end
	end

	return Y
end

"""
	compute_y!(L, φ, P, Y)

Compute an entire set of spherical harmonics ``Y_{l,m}(θ,φ)``
using the given Associated Legendre Polynomials ``P_l^m(cos θ)``
and store in array Y
"""
function compute_y!(L::Integer, ϕ::Real,
	P::VectorLike{<:Real},Y::VectorLike{<:Complex})

	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)

	@inbounds for l in 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * 0.5 * √2
	end

	sig = 1
	@inbounds for m in 1:L
		sig *= -1
		ep = cis(m*ϕ) / √2
		em = sig * conj(ep)
		@inbounds for l in m:L
			p = P[index_p(l,m)]
			Y[index_y(l, -m)] = em * p
			Y[index_y(l,  m)] = ep * p
		end
	end

	return Y
end

"""
	compute_y!(L, x, φ, Y)

Compute an entire set of spherical harmonics ``Y_{l,m}(θ,φ)``
using the given Associated Legendre Polynomials ``P_l^m(cos θ)``,
and store in array Y. 
The Associated Legendre Polynomials are computed on the fly.
"""
function compute_y!(L::Integer, x::Real, ϕ::Real,
	Y::VectorLike{<:Complex})

	@assert length(Y) >= sizeY(L)

	P = compute_p(L,x)

	@inbounds for l in 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * 0.5 * √2
	end

	sig = 1
	@inbounds for m in 1:L
		sig *= -1
		ep = cis(m*ϕ) / √2
		em = sig * conj(ep)
		@inbounds for l in m:L
			p = P[index_p(l,m)]
			Y[index_y(l, -m)] = em * p
			Y[index_y(l,  m)] = ep * p
		end
	end

	return Y
end

"""
	compute_y(L, x, φ)

Compute an entire set of spherical harmonics ``Y_{l,m}(θ,φ)`` for
``x = cos θ`` where ``0 ≤ l ≤ L`` and ``-l ≤ m ≤ l``.
"""
function compute_y(L::Integer, x::Real, ϕ::Real)
	P = allocate_p(L)
	coeff = compute_coefficients(L)
	compute_p!(L, x, coeff, P)
	Y = allocate_y(L)
	compute_y!(L, x, ϕ, P, Y)
	return Y
end