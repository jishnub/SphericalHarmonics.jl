

module SphericalHarmonics


"""
	sizeP(maxDegree)

Return the size of the set of Associated Legendre Polynomials ``P_l^m(x)`` of
degree less than or equal to the given maximum degree
"""
sizeP(maxDegree) = div((maxDegree + 1) * (maxDegree + 2), 2)

"""
	sizeY(maxDegree)

Return the size of the set of real spherical harmonics ``Y_{l,m}(θ,φ)`` of
degree less than or equal to the given maximum degree
"""
sizeY(maxDegree) = (maxDegree +1 ) * (maxDegree + 1)

"""
	index_p(l,m)

Return the index into a flat array of Associated Legendre Polynomials ``P_l^m``
for the given indices ``(l,m)``.
``P_l^m`` are stored in l-major order i.e. [P(0,0), [P(1,0), P(1,1), P(2,0), ...]
"""
index_p(l,m) = m + div(l*(l+1), 2) + 1

"""
	index_y(l,m)

Return the index into a flat array of real spherical harmonics ``Y_{l,m}``
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
	ALPCoefficients( Array{Float64}(undef, sizeP(maxDegree)),
						  Array{Float64}(undef, sizeP(maxDegree)) )

"""
	compute_coefficients(L)

Precompute coefficients ``a_l^m`` and ``b_l^m`` for all l <= L, m <= l
"""
function compute_coefficients(L::Int)
	coeff = ALPCoefficients(L)
	for l in 2:L
		ls = l*l
		lm1s = (l-1) * (l-1)
		for m in 0:(l-2)
			ms = m * m
			coeff.A[index_p(l, m)] = sqrt((4 * ls - 1.) / (ls - ms))
			coeff.B[index_p(l, m)] = -sqrt((lm1s - ms) / (4 * lm1s - 1.))
		end
	end
	return coeff
end

"""
	compute_coefficients(L)

Create an array large enough to store an entire set of Associated Legendre
Polynomials ``P_l^m(x)`` of maximum degree L.
"""
allocate_p(L::Int) = Array{Float64}(undef, sizeP(L))

"""
	compute_p(L, x, coeff, P)

Compute an entire set of Associated Legendre Polynomials ``P_l^m(x)``
using the given coefficients, and store in the array P.
"""
function compute_p(L::Int, x::Float64, coeff::ALPCoefficients,
					    P::Array{Float64,1})
	@assert length(coeff.A) >= sizeP(L)
	@assert length(coeff.B) >= sizeP(L)
	@assert length(P) >= sizeP(L)

	sintheta = sqrt(1. - x * x)
	temp = 0.39894228040143267794 # = sqrt(0.5/M_PI)
	P[index_p(0, 0)] = temp

	if (L > 0)
		SQRT3 = 1.7320508075688772935
		P[index_p(1, 0)] = x * SQRT3 * temp
		SQRT3DIV2 = -1.2247448713915890491
		temp = SQRT3DIV2 * sintheta * temp
		P[index_p(1, 1)] = temp

		for l in 2:L
			for m in 0:(l-2)
				P[index_p(l, m)] = coeff.A[index_p(l, m)] *(X * P[index_p(l - 1, m)]
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
function compute_p(L::Int, x::Float64)
	P = Array{Float64}(undef, sizeP(L))
	coeff = compute_coefficients(L)
	compute_p(L, x, coeff, P)   # TODO: should have ! in the name
	return P
end

"""
	compute_y(L, P, φ, Y)

Compute an entire set of real spherical harmonics ``Y_{l,m}(θ,φ)``
using the given Associated Legendre Polynomials ``P_l^m(cos θ)``
and store in array Y
"""
function compute_y(L::Int, P::Array{Float64,1}, phi::Float64,
						 Y::Array{Float64,1})
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)

	SQRT2 = 1.41421356237309504880
	for l in 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * 0.5 * SQRT2
	end

	# NR2 5.5.4-5.5.5
	c1 = 1.0; c2 = cos(phi)
	s1 = 0.0; s2 = -sin(phi)
	tc = 2.0 * c2
	for m in 1:L
		s = tc * s1 - s2
		c = tc * c1 - c2
		s2 = s1
		s1 = s
		c2 = c1
		c1 = c
		for l in m:L
			Y[index_y(l, -m)] = P[index_p(l, m)] * s
			Y[index_y(l, m)] = P[index_p(l, m)] * c
		end
	end
	return Y
end   # TODO: mutates Y => should have ! in the name

"""
	compute_y(L, x, φ)

Compute an entire set of real spherical harmonics ``Y_{l,m}(θ,φ)`` for
``x = cos θ`` where ``0 ≤ l ≤ L`` and ``-l ≤ m ≤ l``.
"""
function compute_y(L::Int, x::Float64, phi::Float64)
	P = Array{Float64}(undef, sizeP(L))
	coeff = compute_coefficients(L)
	compute_p(L, x, coeff, P)
	Y = Array{Float64}(sizeY(L))
	compute_y(L, x, phi, P, Y)
	return Y
end

end
