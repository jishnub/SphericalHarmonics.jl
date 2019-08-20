# Spherical Harmonics

[![Build Status](https://travis-ci.com/jishnub/SphericalHarmonics.jl.svg?branch=master)](https://travis-ci.com/jishnub/SphericalHarmonics.jl)

This is a fork of an older Julia repository to experiment with spherical
harmonics computations. The original code was modified to use complex
instead of real spherical harmonics.

This is still in testing stages, and probably not useful for real development.


# ORIGINAL README: 

# Associated Legendre Polynomials and Spherical Harmonics Computation for Chemistry Applications

A Julia module for accurate and efficient computation of associated Legendre polynomials and real spherical harmonics for use in chemistry applications.

Our algorithms are based on the following design principles:
- Normalize polynomials P*<sub>l</sub><sup>m</sup>* to avoid overflow/underflow.
- Use a RR in the direction of increasing *l* for ALPs for stability.
- Use trigonometric RRs for sin and cos functions in SHs to save time.
- Precompute coefficients in the RRs to reduce computational cost.
- Compute an entire set of normalized P*<sub>l</sub><sup>m</sup>* where *m* ≥ 0 in a single function call to reduce overhead.
- Avoid loop dependencies in inner loops, allowing operations to be vectorized and pipelined for execution.

For a full description of the code, please see:

[**Associated Legendre Polynomials and Spherical Harmonics Computation for Chemistry Applications**](http://arxiv.org/abs/1410.1748) (2014). Taweetham Limpanuparb and Josh Milthorpe. arXiv: 1410.1748 [physics.chem-ph]
