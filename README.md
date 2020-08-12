# Spherical Harmonics

[![Build Status](https://travis-ci.com/jishnub/SphericalHarmonics.jl.svg?branch=master)](https://travis-ci.com/jishnub/SphericalHarmonics.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jishnub/SphericalHarmonics.jl?svg=true)](https://ci.appveyor.com/project/jishnub/SphericalHarmonics-jl)
[![codecov](https://codecov.io/gh/jishnub/SphericalHarmonics.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jishnub/SphericalHarmonics.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jishnub.github.io/SphericalHarmonics.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jishnub.github.io/SphericalHarmonics.jl/dev)

For a full description of the code, please see:

[**Associated Legendre Polynomials and Spherical Harmonics Computation for Chemistry Applications**](http://arxiv.org/abs/1410.1748) (2014). Taweetham Limpanuparb and Josh Milthorpe. arXiv: 1410.1748 [physics.chem-ph]

# Quick start

The normalized associated Legendre polynomials for an angle `θ` for all `l` in `0 <= l <= lmax` and all `m` in `-l <= m <= l` may be generated using the signature `computePlm(θ; lmax)`, eg.

```julia
julia> P = computePlmcostheta(pi/2, lmax = 1)
3-element SHArray(::Array{Float64,1}, (ML(0:1, 0:1),)):
  0.3989422804014327
  4.231083042742082e-17
 -0.4886025119029199
```

The polynomials are ordered with `m` increasing faster than `l`, and the returned array may be indexed using `(l,m)` Tuples as 

```julia
julia> P[(0,0)]
0.3989422804014327

julia> P[(1,1)] == P[3]
true
```

Spherical harmonics for a colatitude `θ` and azimuth `ϕ` may be generated using the signature `computeYlm(θ, ϕ; lmax)`, eg.

```julia
julia> Y = computeYlm(pi/3, 0, lmax = 1) 
4-element SHArray(::Array{Complex{Float64},1}, (ML(0:1, -1:1),)):
  0.2820947917738782 + 0.0im
  0.2992067103010745 - 0.0im
 0.24430125595146002 + 0.0im
 -0.2992067103010745 - 0.0im
```

The returned array may be indexed using `(l,m)` Tuples as well, as 

```julia
julia> Y[(1,-1)]
0.2992067103010745 - 0.0im

julia> Y[(1,-1)] == Y[2]
true
```

Special angles `SphericalHarmonics.NorthPole()` and `SphericalHarmonics.SouthPole()` may be passed as `θ` to use efficient algorithms.

## Increased precision

Arguments of `BigInt` and `BigFloat` types would increase the precision of the result.

```julia
julia> Y = computeYlm(big(pi)/2, big(0), lmax = big(1)) # Arbitrary precision
4-element SHArray(::Array{Complex{BigFloat},1}, (ML(0:1, -1:1),)):
    0.2820947917738781434740397257803862929220253146644994284220428608553212342207478 + 0.0im
    0.3454941494713354792652446460318896831393773703262433134867073548945156550201567 - 0.0im
 2.679783085063171668225419916118067917387251852939708540164955895366691604430101e-78 + 0.0im
   -0.3454941494713354792652446460318896831393773703262433134867073548945156550201567 - 0.0im
```

## Semi-positive harmonics

For real functions it might be sufficient to compute only the functions for `m >= 0`. These may be computed by passing the flag `m_range = SphericalHarmonics.ZeroTo`.

```julia
julia> computeYlm(pi/3, 0, lmax = 1, m_range = SphericalHarmonics.ZeroTo)
3-element SHArray(::Array{Complex{Float64},1}, (ML(0:1, 0:1),)):
  0.2820947917738782 + 0.0im
 0.24430125595146002 + 0.0im
 -0.2992067103010745 - 0.0im
```

## Real harmonics

It's also possible to compute real spherical harmonics by passing the flag `SHType = SphericalHarmonics.RealHarmonics()`, eg.

```julia
julia> Y = computeYlm(pi/3, pi/3, lmax = 1, SHType = SphericalHarmonics.RealHarmonics())
4-element SHArray(::Array{Float64,1}, (ML(0:1, -1:1),)):
  0.2820947917738782
 -0.3664518839271899
  0.24430125595146002
 -0.21157109383040865
```

These are faster to evaluate and require less memory to store.