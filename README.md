# Spherical Harmonics

[![Build Status](https://travis-ci.com/jishnub/SphericalHarmonics.jl.svg?branch=master)](https://travis-ci.com/jishnub/SphericalHarmonics.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jishnub/SphericalHarmonics.jl?svg=true)](https://ci.appveyor.com/project/jishnub/SphericalHarmonicModes-jl)
[![Coverage Status](https://coveralls.io/repos/github/jishnub/SphericalHarmonics.jl/badge.svg?branch=master)](https://coveralls.io/github/jishnub/SphericalHarmonics.jl?branch=master)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jishnub.github.io/SphericalHarmonics.jl/dev)

For a full description of the code, please see:

[**Associated Legendre Polynomials and Spherical Harmonics Computation for Chemistry Applications**](http://arxiv.org/abs/1410.1748) (2014). Taweetham Limpanuparb and Josh Milthorpe. arXiv: 1410.1748 [physics.chem-ph]

# Quick start

The normalized associated Legendre polynomials for an angle `θ` for all `l` in `0 <= l <= lmax` and all m `-l <= m <= l` may be generated using the signature `computePlm(θ; lmax)`, eg.

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
 0.28209479177387814 + 0.0im
  0.2992067103010745 - 0.0im
    0.24430125595146 + 0.0im
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