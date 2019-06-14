
using SphericalHarmonics
using Test, LinearAlgebra, StaticArrays, BenchmarkTools

@testset "SphericalHarmonics" begin
   include("test_ylm.jl")
   include("test_cg.jl")
end 
