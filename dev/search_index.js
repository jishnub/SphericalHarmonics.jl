var documenterSearchIndex = {"docs":
[{"location":"","page":"Reference","title":"Reference","text":"CurrentModule = SphericalHarmonics","category":"page"},{"location":"#SphericalHarmonics.jl","page":"Reference","title":"SphericalHarmonics.jl","text":"","category":"section"},{"location":"","page":"Reference","title":"Reference","text":"Modules = [SphericalHarmonics]","category":"page"},{"location":"#SphericalHarmonics.NorthPole","page":"Reference","title":"SphericalHarmonics.NorthPole","text":"SphericalHarmonics.NorthPole <: SphericalHarmonics.Pole\n\nThe angle theta = 0 in spherical polar coordinates.\n\n\n\n\n\n","category":"type"},{"location":"#SphericalHarmonics.Pole","page":"Reference","title":"SphericalHarmonics.Pole","text":"SphericalHarmonics.Pole <: Real\n\nSupertype of NorthPole and SouthPole.\n\n\n\n\n\n","category":"type"},{"location":"#SphericalHarmonics.SouthPole","page":"Reference","title":"SphericalHarmonics.SouthPole","text":"SphericalHarmonics.SouthPole <: SphericalHarmonics.Pole\n\nThe angle theta = π in spherical polar coordinates.\n\n\n\n\n\n","category":"type"},{"location":"#SphericalHarmonics.allocate_p-Tuple{Type,Integer}","page":"Reference","title":"SphericalHarmonics.allocate_p","text":"SphericalHarmonics.allocate_p([T::Type = Float64], L::Integer)\n\nAllocate an array large enough to store an entire set of Associated Legendre Polynomials barP_l^m(x) of maximum degree L.\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.allocate_y-Tuple{Type,Integer}","page":"Reference","title":"SphericalHarmonics.allocate_y","text":"SphericalHarmonics.allocate_y([T::Type = ComplexF64], L::Integer)\n\nAllocate an array large enough to store an entire set of spherical harmonics Y_lm(θφ) of maximum degree L.\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.computePlmcostheta!-Union{Tuple{THETA}, Tuple{AbstractArray{var\"#s16\",1} where var\"#s16\"<:Real,THETA,Integer,AbstractArray{T,2} where T}} where THETA<:Real","page":"Reference","title":"SphericalHarmonics.computePlmcostheta!","text":"computePlmcostheta!(P::AbstractVector{<:Real}, θ::Real, lmax::Integer, coeff::AbstractMatrix)\ncomputePlmcostheta!(P::AbstractVector{<:Real}, θ::SphericalHarmonics.Pole, lmax::Integer)\n\nCompute an entire set of normalized Associated Legendre Polynomials barP_l^m(cos(θ)) using the given coefficients, and store in the array P. The matrix coeff may be computed  using compute_coefficients.\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.computePlmcostheta-Tuple{Real}","page":"Reference","title":"SphericalHarmonics.computePlmcostheta","text":"computePlmcostheta(θ::Real; lmax::Integer)\n\nCompute an entire set of normalized Associated Legendre Polynomials barP_l^m(cos(θ)) where 0  l  l_mathrmmax and 0  m  l for colatitude theta. \n\nThe polynomials are normalized as \n\nbarP_ell m = sqrtfrac(2ell + 1)(ell-m)2pi (ell+m) P_ell m\n\nReturns an SHVector that may be  indexed using (l,m) pairs aside from the canonical indexing with Ints.\n\nExamples\n\njulia> P = computePlmcostheta(pi/2, 1)\n3-element SHArray(::Array{Float64,1}, (ML(0:1, 0:1),)):\n  0.3989422804014327\n  4.231083042742082e-17\n -0.4886025119029199\n\njulia> P[(0,0)]\n0.3989422804014327\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.computePlmx!-Union{Tuple{THETA}, Tuple{AbstractArray{var\"#s16\",1} where var\"#s16\"<:Real,THETA,Integer,AbstractArray{T,2} where T}} where THETA<:Real","page":"Reference","title":"SphericalHarmonics.computePlmx!","text":"computePlmx!(P::AbstractVector{<:Real}, x::Real, lmax::Integer, coeff::AbstractMatrix)\n\nCompute an entire set of normalized Associated Legendre Polynomials barP_l^m(x) using the given coefficients, and store in the array P. The matrix coeff may be computed  using compute_coefficients.\n\nThe argument x needs to lie in -1  x  1. The function implicitly assumes that  x = cos(theta) where 0  theta  π.\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.computePlmx-Tuple{Real}","page":"Reference","title":"SphericalHarmonics.computePlmx","text":"computePlmx(x::Real; lmax::Integer)\n\nCompute an entire set of normalized Associated Legendre Polynomials barP_l^m(x) where 0  l  l_mathrmmax and 0  m  l.\n\nThe argument x needs to lie in -1  x  1. The function implicitly assumes that  x = cos(theta) where 0  theta  π.\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.computeYlm!-Tuple{AbstractArray{T,1} where T,Real,Real}","page":"Reference","title":"SphericalHarmonics.computeYlm!","text":"computeYlm!(Y::AbstractVector{<:Complex}, θ::Real, φ::Real; lmax::Integer, [SHtype = ComplexHarmonics()])\n\nCompute an entire set of spherical harmonics Y_lm(θφ) for 0  l  l_mathrmmax, and store them in the array Y.\n\nThe optional argument SHtype may be used to choose between real and complex harmonics. To compute real spherical harmonics, set this to RealHarmonics().\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.computeYlm!-Union{Tuple{R}, Tuple{AbstractArray{T,1} where T,AbstractArray{R,1},Real,Real,Integer}, Tuple{AbstractArray{T,1} where T,AbstractArray{R,1},Real,Real,Integer,SphericalHarmonics.HarmonicType}} where R<:Real","page":"Reference","title":"SphericalHarmonics.computeYlm!","text":"computeYlm!(Y::AbstractVector, P::AbstractVector{<:Real}, θ::Real, φ::Real, L::Integer, [SHtype = ComplexHarmonics()])\n\nCompute an entire set of spherical harmonics Y_lm(θφ) using the precomputed associated Legendre Polynomials barP_l^m(x = cos(θ)), and store in the array Y. The array P may be computed using computePlmcostheta.\n\nThe optional argument SHtype may be used to choose between real and complex harmonics. To compute real spherical harmonics, set this to RealHarmonics().\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.computeYlm-Union{Tuple{I}, Tuple{PHI}, Tuple{THETA}, Tuple{THETA,PHI,I}, Tuple{THETA,PHI,I,SphericalHarmonics.HarmonicType}} where I<:Integer where PHI<:Real where THETA<:Real","page":"Reference","title":"SphericalHarmonics.computeYlm","text":"computeYlm(θ::Real, ϕ::Real; lmax::Integer, [SHtype = ComplexHarmonics()])\n\nCompute an entire set of spherical harmonics Y_lm(θϕ) for  0  l  l_mathrmmax and -l  m  l, for colatitude theta and  azimuth phi.\n\nReturns an SHVector that may be  indexed using (l,m) pairs aside from the canonical indexing with Ints.\n\nThe optional argument SHtype may be used to choose between real and complex harmonics. To compute real spherical harmonics, set this to RealHarmonics().\n\nExamples\n\njulia> Y = computeYlm(pi/2, 0, 1)\n4-element SHArray(::Array{Complex{Float64},1}, (ML(0:1, -1:1),)):\n   0.28209479177387814 + 0.0im\n   0.34549414947133544 - 0.0im\n 2.991827511286337e-17 + 0.0im\n  -0.34549414947133544 - 0.0im\n\njulia> Y[(1,-1)] # l = 1, m = -1\n0.34549414947133544 - 0.0im\n\njulia> Y = computeYlm(big(pi)/2, 0, 1) # Arbitrary precision\n4-element SHArray(::Array{Complex{BigFloat},1}, (ML(0:1, -1:1),)):\n    0.2820947917738781434740397257803862929220253146644994284220428608553212342207478 + 0.0im\n    0.3454941494713354800004725866746756805800203549224377345215348347601343937267067 - 0.0im\n 2.679783085063171668225419916118067917387251852939708540164955895366691604430101e-78 + 0.0im\n   -0.3454941494713354800004725866746756805800203549224377345215348347601343937267067 - 0.0im\n\njulia> computeYlm(SphericalHarmonics.NorthPole(), 0, 1)\n4-element SHArray(::Array{Complex{Float64},1}, (ML(0:1, -1:1),)):\n 0.28209479177387814 + 0.0im\n                 0.0 + 0.0im\n  0.4886025119029199 + 0.0im\n                 0.0 + 0.0im \n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.compute_coefficients-Tuple{Integer}","page":"Reference","title":"SphericalHarmonics.compute_coefficients","text":"SphericalHarmonics.compute_coefficients(L::Integer)\n\nPrecompute coefficients a_l^m and b_l^m for all l  L and 0  m  l.\n\n\n\n\n\n","category":"method"}]
}
