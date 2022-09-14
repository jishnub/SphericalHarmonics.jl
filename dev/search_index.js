var documenterSearchIndex = {"docs":
[{"location":"","page":"Reference","title":"Reference","text":"CurrentModule = SphericalHarmonics","category":"page"},{"location":"#SphericalHarmonics.jl","page":"Reference","title":"SphericalHarmonics.jl","text":"","category":"section"},{"location":"","page":"Reference","title":"Reference","text":"Modules = [SphericalHarmonics]","category":"page"},{"location":"#SphericalHarmonics.NorthPole","page":"Reference","title":"SphericalHarmonics.NorthPole","text":"SphericalHarmonics.NorthPole <: SphericalHarmonics.Pole\n\nThe angle theta = 0 in spherical polar coordinates.\n\n\n\n\n\n","category":"type"},{"location":"#SphericalHarmonics.Pole","page":"Reference","title":"SphericalHarmonics.Pole","text":"SphericalHarmonics.Pole <: Real\n\nSupertype of NorthPole and SouthPole.\n\n\n\n\n\n","category":"type"},{"location":"#SphericalHarmonics.SouthPole","page":"Reference","title":"SphericalHarmonics.SouthPole","text":"SphericalHarmonics.SouthPole <: SphericalHarmonics.Pole\n\nThe angle theta = π in spherical polar coordinates.\n\n\n\n\n\n","category":"type"},{"location":"#SphericalHarmonics.SphericalHarmonicsCache","page":"Reference","title":"SphericalHarmonics.SphericalHarmonicsCache","text":"SphericalHarmonicsCache\n\nPreallocate arrays of associated Legendre polynomials and spherical harmonics. Such an object may be allocated using cache.\n\n\n\n\n\n","category":"type"},{"location":"#SphericalHarmonics.allocate_p-Tuple{Type, Integer}","page":"Reference","title":"SphericalHarmonics.allocate_p","text":"SphericalHarmonics.allocate_p([T::Type = Float64], lmax::Integer)\n\nAllocate an array large enough to store an entire set of Associated Legendre Polynomials barP_ℓ^m(x) of maximum degree ℓ.\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.allocate_y","page":"Reference","title":"SphericalHarmonics.allocate_y","text":"SphericalHarmonics.allocate_y([T::Type = ComplexF64], lmax::Integer)\n\nAllocate an array large enough to store an entire set of spherical harmonics Y_ℓm(θϕ) of maximum degree ℓ.\n\n\n\n\n\n","category":"function"},{"location":"#SphericalHarmonics.associatedLegendre-Tuple{Any}","page":"Reference","title":"SphericalHarmonics.associatedLegendre","text":"SphericalHarmonics.associatedLegendre(θ; l::Integer, m::Integer, [coeff = nothing], [norm = SphericalHarmonics.LMnorm()])\n\nEvaluate the normalized associated Legendre polynomial barP_ℓ^m(cos theta). Optionally a matrix of coefficients returned by compute_coefficients may be provided.\n\nThe keyword argument norm may be used to specify the how the polynomials are normalized. See computePlmcostheta for the possible normalization options.\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.associatedLegendrex-Tuple{Any}","page":"Reference","title":"SphericalHarmonics.associatedLegendrex","text":"SphericalHarmonics.associatedLegendrex(x; l::Integer, m::Integer, [coeff = nothing], [norm = SphericalHarmonics.LMnorm()])\n\nEvaluate the normalized associated Legendre polynomial barP_ℓ^m(x). Optionally a matrix of coefficients returned by compute_coefficients may be provided.\n\nThe keyword argument norm may be used to specify the how the polynomials are normalized. See computePlmcostheta for the possible normalization options.\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.cache-Tuple","page":"Reference","title":"SphericalHarmonics.cache","text":"cache([T::Type = Float64], lmax, [m_range = SphericalHarmonicModes.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])\n\nAllocate arrays to evaluate associated Legendre polynomials and spherical harmonics. The returned object may be passed to computePlmcostheta! and computeYlm!. The coefficients are cached and need not be recomputed.\n\nExamples\n\njulia> S = SphericalHarmonics.cache(1);\n\njulia> computePlmcostheta!(S, pi/3, 1)\n3-element normalized AssociatedLegendrePolynomials{Float64} for lmax = 1 and cosθ = 0.5:\n  0.3989422804014327\n  0.34549414947133555\n -0.4231421876608172\n\njulia> computeYlm!(S, pi/3, pi/4, 1)\n4-element SHArray(::Vector{ComplexF64}, (ML(0:1, -1:1),)):\n   0.2820947917738782 + 0.0im\n  0.21157109383040865 - 0.2115710938304086im\n  0.24430125595146002 + 0.0im\n -0.21157109383040865 - 0.2115710938304086im\n\nChoosing a new lmax in computePlmcostheta! expands the cache if necessary.\n\njulia> computePlmcostheta!(S, pi/3, 2)\n6-element normalized AssociatedLegendrePolynomials{Float64} for lmax = 2 and cosθ = 0.5:\n  0.3989422804014327\n  0.34549414947133555\n -0.4231421876608172\n -0.11150775725954817\n -0.4730873478787801\n  0.40970566147202964\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.computePlmcostheta!","page":"Reference","title":"SphericalHarmonics.computePlmcostheta!","text":"computePlmcostheta!(S::SphericalHarmonicsCache, θ, [lmax::Integer]; [norm = SphericalHarmonics.LMnorm()])\n\nCompute an entire set of normalized Associated Legendre Polynomials barP_ℓ^m(cos θ) using the pre-computed coefficients in S, and store the result in S. If lmax is not provided, the value of lmax for which coefficients have been computed in S is used.\n\nThe keyword argument norm may be used to specify the how the polynomials are normalized. See computePlmcostheta for the possible normalization options.\n\n\n\n\n\n","category":"function"},{"location":"#SphericalHarmonics.computePlmcostheta!-Tuple{AbstractVector, Any, Any, Vararg{Any}}","page":"Reference","title":"SphericalHarmonics.computePlmcostheta!","text":"computePlmcostheta!(P::AbstractVector, θ, lmax::Integer, coeff; [norm = SphericalHarmonics.LMnorm()])\ncomputePlmcostheta!(P::AbstractVector, θ::SphericalHarmonics.Pole, lmax::Integer; [norm = SphericalHarmonics.LMnorm()])\n\nCompute an entire set of normalized Associated Legendre Polynomials barP_ℓ^m(cos θ) using the given coefficients, and store in the array P. The matrix coeff may be computed using compute_coefficients.\n\nThe keyword argument norm may be used to specify the how the polynomials are normalized. See computePlmcostheta for the possible normalization options.\n\ncomputePlmcostheta!(P::AbstractVector, θ, lmax::Integer, m::Integer, coeff)\n\nCompute the Associated Legendre Polynomials barP_ℓ^m(cos θ) for for the specified m and all ℓ lying in m  ℓ  ℓ_mathrmmax. The array P needs to be large enough to hold all the polynomials for 0  ℓ  ℓ_mathrmmax and 0  m  ℓ, as the recursive evaluation requires the computation of extra elements. Pre-existing values in P may be overwritten, even for azimuthal orders not equal to m.\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.computePlmcostheta-Tuple{Any}","page":"Reference","title":"SphericalHarmonics.computePlmcostheta","text":"computePlmcostheta(θ; lmax::Integer, [m::Integer], [norm = SphericalHarmonics.LMnorm()])\ncomputePlmcostheta(θ, lmax::Integer[, m::Integer]; [norm = SphericalHarmonics.LMnorm()])\n\nCompute an entire set of normalized Associated Legendre Polynomials barP_ℓ^m(cos θ) where 0  ℓ  ℓ_mathrmmax and 0  m  ℓ for colatitude theta. If m is provided then only the polynomials corresponding to the specified m are computed.\n\nThe polynomials are normalized as\n\nbarP_ell^m = sqrtfrac(2ell + 1)(ell-m)2pi (ell+m) P_ell m\n\nwhere P_ell m are the standard associated Legendre polynomials, and are defined in terms of Legendre polynomials P_ell(x) as\n\nP_ell mleft(xright)=(-1)^m left(1-x^2right)^m2fracd^mdx^mP_ellleft(xright)\n\nThe normalized polynomials barP_ell^m satisfy\n\nint_-1^1 dxleft barP_ell^m(x) right^2 = frac1pi\n\nA different normalization may be chosen by specifying the keyword argument norm, with optionally the Condon-Shortley phase disabled by passing the keyword argument csphase to the constructor of the normalization specifier. The possible normalization options are:\n\nSphericalHarmonics.LMnorm([; csphase = true]): the default normalization described above\nSphericalHarmonics.Orthonormal([; csphase = true]): Orthonormal polynomials that are defined as\n\ntildeP_ell^m = sqrtfrac(2ell + 1)(ell-m)2(ell+m) P_ell m =\nsqrtpi barP_ell m\n\nand satisfy\n\nint_-1^1 tildeP_ell m(x) tildeP_k m(x) dx = delta_ℓk\n\nSphericalHarmonics.Unnormalized([; csphase = true]): The polynomials P_ℓm that satisfy P_ℓm(1)=delta_m0\n\nwithin numerical precision bounds, as well as\n\nint_-1^1 P_ell m(x) P_k m(x) dx = frac2(ell+m)(2ell+1)(ell-m)delta_ℓk\n\nIn each case setting csphase = false will lead to an overall factor of (-1)^m being multiplied to the polynomials.\n\nReturns an AbstractVector that may be indexed using (ℓ,m) pairs aside from the canonical indexing with Ints.\n\nThe precision of the result may be increased by using arbitrary-precision arguments.\n\nExamples\n\njulia> P = computePlmcostheta(pi/2, lmax = 1)\n3-element normalized AssociatedLegendrePolynomials{Float64} for lmax = 1 and cosθ = 6.123e-17:\n  0.3989422804014327\n  4.231083042742082e-17\n -0.4886025119029199\n\njulia> P[(0,0)]\n0.3989422804014327\n\njulia> P = computePlmcostheta(big(pi)/2, lmax = 1)\n3-element normalized AssociatedLegendrePolynomials{BigFloat} for lmax = 1 and cosθ = 5.485e-78:\n  0.3989422804014326779399460599343818684758586311649346576659258296706579258993008\n  3.789785583114350800838137317730900078444216599640987847808409161681770236721676e-78\n -0.4886025119029199215863846228383470045758856081942277021382431574458410003616367\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.computePlmx!","page":"Reference","title":"SphericalHarmonics.computePlmx!","text":"computePlmx!(S::SphericalHarmonicsCache, x[, lmax::Integer]; [norm = SphericalHarmonics.LMnorm()])\n\nCompute an entire set of normalized Associated Legendre Polynomials barP_ℓ^m(x) using the pre-computed coefficients in S, and store the result in S. If lmax is not provided, the value of lmax for which coefficients have been computed in S is used.\n\nThe keyword argument norm may be used to specify the how the polynomials are normalized. See computePlmcostheta for the possible normalization options.\n\n\n\n\n\n","category":"function"},{"location":"#SphericalHarmonics.computePlmx!-Tuple{AbstractVector, Any, Integer, Vararg{Any}}","page":"Reference","title":"SphericalHarmonics.computePlmx!","text":"computePlmx!(P::AbstractVector, x, lmax::Integer, coeff::AbstractMatrix; [norm = SphericalHarmonics.LMnorm()])\n\nCompute an entire set of normalized Associated Legendre Polynomials barP_ℓ^m(x) using the given coefficients, and store in the array P. The matrix coeff may be computed using compute_coefficients.\n\nThe argument x needs to lie in -1  x  1. The function implicitly assumes that x = costheta where 0  theta  π.\n\nThe keyword argument norm may be used to specify the how the polynomials are normalized. See computePlmcostheta for the possible normalization options.\n\ncomputePlmx!(P::AbstractVector, x, lmax::Integer, m::Integer, coeff::AbstractMatrix)\n\nCompute the set of normalized Associated Legendre Polynomials barP_ℓ^m(x) for for the specified m and all ℓ lying in m  ℓ  ℓ_mathrmmax .\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.computePlmx-Tuple{Any}","page":"Reference","title":"SphericalHarmonics.computePlmx","text":"computePlmx(x; lmax::Integer, [m::Integer], [norm = SphericalHarmonics.LMnorm()])\ncomputePlmx(x, lmax::Integer[, m::Integer]; [norm = SphericalHarmonics.LMnorm()])\n\nCompute an entire set of normalized Associated Legendre Polynomials barP_ℓ^m(x) where 0  ℓ  ℓ_mathrmmax and 0  m  ℓ. If m is provided then only the polynomials for that azimuthal order are computed.\n\nThe argument x needs to lie in -1  x  1. The function implicitly assumes that x = costheta where 0  theta  π.\n\nThe keyword argument norm may be used to specify the how the polynomials are normalized. See computePlmcostheta for the possible normalization options.\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.computeYlm","page":"Reference","title":"SphericalHarmonics.computeYlm","text":"computeYlm(θ, ϕ; lmax::Integer, [m::Integer], [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])\ncomputeYlm(θ::SphericalHarmonics.Pole; lmax::Integer, [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])\n\nCompute an entire set of spherical harmonics Y_ℓm(θϕ) for 0  ℓ  ℓ_mathrmmax and -ℓ  m  ℓ, for colatitude theta and azimuth phi. If m is provided then only the polynomials for the specified m are computed.\n\nReturns an AbstractVector that may be indexed using (l,m) pairs aside from the canonical indexing with Ints.\n\nThe optional argument m_range decides if the spherical harmonics for negative m values are computed. By default the functions for all values of m are evaluated. Setting m_range to SphericalHarmonics.ZeroTo would result in only functions for m ≥ 0 being evaluated.\n\nThe optional argument SHType may be used to choose between real and complex harmonics. To compute real spherical harmonics, set this to SphericalHarmonics.RealHarmonics(). The complex harmonics are defined as\n\nY_ellmleft(thetaphiright)=begincases\nfrac1sqrt2barP_ell^mleft(costhetaright)expleft(imphiright)  mgeq0\nleft(-1right)^mY_ell-m^*left(thetaphiright)  m0\nendcases\n\nThis definition corresponds to Laplace sphercial harmonics, whereas the quantum mechanical definition prepends a Condon-Shortley phase on the harmonics.\n\nThe real spherical harmonics are defined as\n\nY_ellmleft(thetaphiright)=begincases\nbarP_ell^leftmrightleft(costhetaright)sinleftmrightphi  m0\nbarP_ell^0left(costhetaright)sqrt2  m=0\nbarP_ell^mleft(costhetaright)cos mphi  m0\nendcases\n\nThe precision of the result may be increased by using arbitrary-precision arguments.\n\nExamples\n\njulia> Y = computeYlm(pi/2, 0, lmax = 1)\n4-element SHArray(::Vector{ComplexF64}, (ML(0:1, -1:1),)):\n     0.2820947917738782 + 0.0im\n     0.3454941494713355 - 0.0im\n 2.9918275112863375e-17 + 0.0im\n    -0.3454941494713355 - 0.0im\n\njulia> Y[(1,-1)] # index using (l,m)\n0.3454941494713355 - 0.0im\n\njulia> Y = computeYlm(big(pi)/2, big(0), lmax = big(1)) # Arbitrary precision\n4-element SHArray(::Vector{Complex{BigFloat}}, (ML(0:1, -1:1),)):\n    0.2820947917738781434740397257803862929220253146644994284220428608553212342207478 + 0.0im\n    0.3454941494713354792652446460318896831393773703262433134867073548945156550201567 - 0.0im\n 2.679783085063171668225419916118067917387251852939708540164955895366691604430101e-78 + 0.0im\n   -0.3454941494713354792652446460318896831393773703262433134867073548945156550201567 - 0.0im\n\njulia> computeYlm(SphericalHarmonics.NorthPole(), 0, lmax = 1)\n4-element SHArray(::Vector{ComplexF64}, (ML(0:1, -1:1),)):\n 0.2820947917738782 + 0.0im\n               -0.0 + 0.0im\n   0.48860251190292 + 0.0im\n                0.0 + 0.0im\n\njulia> Y = computeYlm(pi/2, pi/3, lmax = 1, m_range = SphericalHarmonics.ZeroTo, SHType = SphericalHarmonics.RealHarmonics())\n3-element SHArray(::Vector{Float64}, (ML(0:1, 0:1),)):\n  0.2820947917738782\n  2.9918275112863375e-17\n -0.24430125595146002\n\n\n\n\n\n","category":"function"},{"location":"#SphericalHarmonics.computeYlm!","page":"Reference","title":"SphericalHarmonics.computeYlm!","text":"computeYlm!(S::SphericalHarmonicsCache, θ, ϕ, [lmax::Integer])\n\nCompute an entire set of spherical harmonics Y_ℓm(θϕ) for 0  ℓ  ℓ_mathrmmax using the pre-computed associated Legendre polynomials saved in S, and store the result in S. If lmax is not provided, the value of lmax for which associated Legendre polynomials have been computed in S is used.\n\nnote: Note\nThis function assumes that the associated Legendre polynomials have been pre-computed, and does not perform any check on their values. In general computeYlm!(S::SphericalHarmonicsCache, θ, ϕ, lmax) should only be called after a preceeding call to computePlmcostheta!(S, θ, lmax) in order to obtain meaningful results.\n\n\n\n\n\n","category":"function"},{"location":"#SphericalHarmonics.computeYlm!-2","page":"Reference","title":"SphericalHarmonics.computeYlm!","text":"computeYlm!(Y::AbstractVector, P::AbstractVector, θ, ϕ; lmax::Integer, [m::Integer], [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])\ncomputeYlm!(Y::AbstractVector, P::AbstractVector, θ, ϕ, lmax::Integer, [m::Integer, [m_range = SphericalHarmonics.FullRange, [SHType = SphericalHarmonics.ComplexHarmonics()]]])\n\nCompute an entire set of spherical harmonics Y_ℓm(θϕ) using the precomputed associated Legendre Polynomials barP_ℓ^m(cos θ), and store in the array Y. The array P may be computed using computePlmcostheta.\n\nThe optional argument m_range decides if the spherical harmonics for negative m values are computed. By default the functions for all values of m are evaluated. Setting m_range to SphericalHarmonics.ZeroTo would result in only functions for m ≥ 0 being evaluated. Providing m would override this, in which case only the polynomials corresponding to the azimuthal order m would be computed.\n\nThe optional argument SHType may be used to choose between real and complex harmonics. To compute real spherical harmonics, set this to RealHarmonics().\n\nnote: Note\nThis function assumes that the associated Legendre Polynomials have been pre-computed, and does not perform any check on the values of P.\n\n\n\n\n\n","category":"function"},{"location":"#SphericalHarmonics.computeYlm!-3","page":"Reference","title":"SphericalHarmonics.computeYlm!","text":"computeYlm!(Y::AbstractVector, θ, ϕ; lmax::Integer, [m::Integer] [m_range = SphericalHarmonics.FullRange], [SHType = SphericalHarmonics.ComplexHarmonics()])\n\nCompute an entire set of spherical harmonics Y_ℓm(θϕ) for 0  ℓ  ℓ_mathrmmax, and store them in the array Y.\n\nThe optional argument m_range decides if the spherical harmonics for negative m values are computed. By default the functions for all values of m are evaluated. Setting m_range to SphericalHarmonics.ZeroTo would result in only functions for m ≥ 0 being evaluated. Providing m would override this, in which case only the polynomials corresponding to the azimuthal order m would be computed.\n\nThe optional argument SHType may be used to choose between real and complex harmonics. To compute real spherical harmonics, set this to SphericalHarmonics.RealHarmonics().\n\n\n\n\n\n","category":"function"},{"location":"#SphericalHarmonics.compute_coefficients-Tuple{Integer}","page":"Reference","title":"SphericalHarmonics.compute_coefficients","text":"SphericalHarmonics.compute_coefficients(lmax)\n\nPrecompute coefficients a_ℓ^m and b_ℓ^m for all 2  ℓ  ℓ_mathrmmax and 0  m  ℓ-2.\n\nSphericalHarmonics.compute_coefficients(lmax, m)\n\nPrecompute coefficients a_ℓ^m and b_ℓ^m for all m + 2  ℓ  ℓ_mathrmmax and the specified m.\n\n\n\n\n\n","category":"method"},{"location":"#SphericalHarmonics.sphericalharmonic-Tuple{Any, Any}","page":"Reference","title":"SphericalHarmonics.sphericalharmonic","text":"SphericalHarmonics.sphericalharmonic(θ, ϕ; l, m, [SHType = ComplexHarmonics()], [coeff = nothing])\n\nEvaluate the spherical harmonic Y_ℓm(θ ϕ). The flag SHType sets the type of the harmonic computed, and setting this to RealHarmonics() would evaluate real spherical harmonics. Optionally a precomputed matrix of coefficients returned by compute_coefficients may be provided.\n\nExample\n\njulia> SphericalHarmonics.sphericalharmonic(π/3, π/3, l = 500, m = 250)\n-0.18910100312194328 - 0.32753254516944075im\n\njulia> SphericalHarmonics.sphericalharmonic(π/3, π/3, l = 500, m = 250, SHType = SphericalHarmonics.RealHarmonics())\n-0.26742920327340913\n\n\n\n\n\n","category":"method"}]
}
