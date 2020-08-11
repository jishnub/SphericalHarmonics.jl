using SphericalHarmonics
using Test

import SphericalHarmonics: NorthPole, SouthPole, allocate_y, allocate_p

@test isempty(Test.detect_ambiguities(Base, Core, SphericalHarmonics))

@testset "allocate" begin
    lmax = 4
    @test size(allocate_y(lmax)) == size(allocate_y(ComplexF64, lmax))
    @test eltype(allocate_y(Complex{BigFloat}, lmax)) == Complex{BigFloat}
    @test eltype(allocate_y(BigFloat, lmax)) == BigFloat

    @test size(allocate_p(lmax)) == size(allocate_p(Float64, lmax))
    @test eltype(allocate_p(BigFloat, lmax)) == BigFloat
end

@testset "indexing" begin
    @testset "p" begin
        ind1 = SphericalHarmonics.index_p(3, 0)
        ind2 = SphericalHarmonics.index_p(3, 3)
        @test SphericalHarmonics.index_p(3) == ind1:ind2
    end
    @testset "y" begin
        ind1 = SphericalHarmonics.index_y(3, -3)
        ind2 = SphericalHarmonics.index_y(3, 3)
        @test SphericalHarmonics.index_y(3) == ind1:ind2
    end
end

@testset "Ylm explicit" begin

    function explicit_shs_complex(θ, φ)
        Y00 = 0.5 * sqrt(1/π)
        Y1m1 = 0.5 * sqrt(3/(2*π))*sin(θ)*exp(-im*φ)
        Y10 = 0.5 * sqrt(3/π)*cos(θ)
        Y11 = -0.5 * sqrt(3/(2*π))*sin(θ)*exp(im*φ)
        Y2m2 = 0.25 * sqrt(15/(2*π))*sin(θ)^2*exp(-2*im*φ)
        Y2m1 = 0.5 * sqrt(15/(2*π))*sin(θ)*cos(θ)*exp(-im*φ)
        Y20 = 0.25 * sqrt(5/π)*(3*cos(θ)^2 - 1)
        Y21 = -0.5 * sqrt(15/(2*π))*sin(θ)*cos(θ)*exp(im*φ)
        Y22 = 0.25 * sqrt(15/(2*π))*sin(θ)^2*exp(2*im*φ)
        Y3m3 = 1/8 * exp(-3 * im * φ) * sqrt(35/π) * sin(θ)^3
        Y3m2 = 1/4 * exp(-2 * im * φ) * sqrt(105/(2*π)) * cos(θ) * sin(θ)^2
        Y3m1 = 1/8 * exp(-im * φ) * sqrt(21/π) * (-1 + 5 * cos(θ)^2) * sin(θ)
        Y30 = 1/4 * sqrt(7/π) * (-3 * cos(θ) + 5 * cos(θ)^3)
        Y31 = -(1/8) * exp(im * φ) * sqrt(21/π) * (-1 + 5 * cos(θ)^2) * sin(θ)
        Y32 = 1/4 * exp(2 * im * φ) * sqrt(105/(2*π)) * cos(θ) * sin(θ)^2
        Y33 = -(1/8) * exp(3 * im * φ) * sqrt(35/π) * sin(θ)^3
        return [Y00, Y1m1, Y10, Y11, Y2m2, Y2m1, Y20, Y21, Y22,
               Y3m3, Y3m2, Y3m1, Y30, Y31, Y32, Y33]
    end

    @testset "complex harmonics" begin
        
        for θ in LinRange(0, π, 100), ϕ in LinRange(0, 2π, 200)
            Y = computeYlm(θ, ϕ, 3)
            Yex = explicit_shs_complex(θ, ϕ)
            @test Y ≈ Yex
        end
    end

    @testset "real harmonics" begin
        function explicit_shs_real(θ, φ)
            Y00 = 0.5 * sqrt(1/π)
            Y1m1 = -0.5 * sqrt(3/π)*sin(θ)*sin(φ)
            Y10 = 0.5 * sqrt(3/π)*cos(θ)
            Y11 = -0.5 * sqrt(3/π)*sin(θ)*cos(φ)
            Y2m2 = 0.25 * sqrt(15/π)*sin(θ)^2*sin(2*φ)
            Y2m1 = -0.5 * sqrt(15/π)*sin(θ)*cos(θ)*sin(φ)
            Y20 = 0.25 * sqrt(5/π)*(3*cos(θ)^2 - 1)
            Y21 = -0.5 * sqrt(15/π)*sin(θ)*cos(θ)*cos(φ)
            Y22 = 0.25 * sqrt(15/π)*sin(θ)^2*cos(2*φ)
            Y3m3 = -1/8 * sin(3 * φ) * sqrt(2*35/π) * sin(θ)^3
            Y3m2 = 1/4 * sin(2 * φ) * sqrt(105/π) * cos(θ) * sin(θ)^2
            Y3m1 = -1/8 * sin(φ) * sqrt(2*21/π) * (-1 + 5 * cos(θ)^2) * sin(θ)
            Y30 = 1/4 * sqrt(7/π) * (-3 * cos(θ) + 5 * cos(θ)^3)
            Y31 = -(1/8) * cos(φ) * sqrt(2*21/π) * (-1 + 5 * cos(θ)^2) * sin(θ)
            Y32 = 1/4 * cos(2 * φ) * sqrt(105/π) * cos(θ) * sin(θ)^2
            Y33 = -(1/8) * cos(3 * φ) * sqrt(2*35/π) * sin(θ)^3
            return [Y00, Y1m1, Y10, Y11, Y2m2, Y2m1, Y20, Y21, Y22,
                   Y3m3, Y3m2, Y3m1, Y30, Y31, Y32, Y33]
        end

        for θ in LinRange(0, π, 100), ϕ in LinRange(0, 2π, 100)
            Y = computeYlm(θ, ϕ, 3, SphericalHarmonics.RealHarmonics())
            Yex = explicit_shs_real(θ, ϕ)
            @test Y ≈ Yex
        end
    end
    
    @testset "Pole" begin
        @testset "NorthPole" begin
            Y = computeYlm(NorthPole(), 3)
            Yex = explicit_shs_complex(0, 0)
            @test Y ≈ Yex
        end
        @testset "SouthPole" begin
            Y = computeYlm(SouthPole(), 3)
            Yex = explicit_shs_complex(π, 0)
            @test Y ≈ Yex 
        end
        @testset "RealHarmonics == ComplexHarmonics" begin
            for x in (NorthPole(), SouthPole())
                YC = computeYlm(x, 4, SphericalHarmonics.ComplexHarmonics())
                YR = computeYlm(x, 4, SphericalHarmonics.RealHarmonics())
                @test YC == YR
            end
        end
    end
end

@testset "computePlm!" begin
    lmax = 4
    θ = pi/3
    coeff = SphericalHarmonics.compute_coefficients(lmax)
    P = SphericalHarmonics.allocate_p(Float64, lmax)
    computePlmcostheta!(P, θ, lmax, coeff)

    @test P == computePlmcostheta(θ, lmax)

    computePlmcostheta!(P, NorthPole(), lmax, coeff)
    @test P == computePlmcostheta(NorthPole(), lmax)
end

@testset "computePlm" begin
    θ = π/3
    @test SphericalHarmonics.computePlmx(cos(θ), 4) ≈ computePlmcostheta(θ, 4)
    @test SphericalHarmonics.computePlmx(cos(θ), lmax = 4) ≈ computePlmcostheta(θ, 4)
    @test SphericalHarmonics.computePlmx(cos(θ), lmax = 4) ≈ computePlmcostheta(θ, lmax = 4)
end

@testset "computeYlm" begin
    θ, ϕ = pi/3, pi/3
    lmax = 4

    @test SphericalHarmonics.computeYlm(θ, ϕ, lmax) == SphericalHarmonics.computeYlm(θ, ϕ, lmax = lmax)
end

@testset "computeYlm!" begin
    θ, ϕ = pi/3, pi/3
    lmax = 4
    P = SphericalHarmonics.computePlmcostheta(θ, lmax)
    Y1 = SphericalHarmonics.allocate_y(ComplexF64, lmax)
    Y2 = SphericalHarmonics.computeYlm(θ, ϕ, lmax)
    
    SphericalHarmonics.computeYlm!(Y1, θ, ϕ, lmax)
    @test Y1 ≈ Y2

    SphericalHarmonics.computeYlm!(Y1, θ, ϕ; lmax = lmax)
    @test Y1 ≈ Y2
    
    SphericalHarmonics.computeYlm!(Y1, P, θ, ϕ; lmax = lmax)
    @test Y1 ≈ Y2
end

@testset "Pole" begin
   @testset "utils" begin
      @test one(NorthPole()) == 1
      @test one(SouthPole()) == 1
      @test zero(NorthPole()) == 0
      @test zero(SouthPole()) == 0
      
      @test float(NorthPole()) == 0
      @test float(SouthPole()) == float(pi)
      
      @test Float64(NorthPole()) == 0
      @test Float64(SouthPole()) == float(pi)

      @test promote_rule(SouthPole, Float64) == Float64
      @test promote_rule(SouthPole, BigFloat) == BigFloat
   end

   @testset "North pole" begin
      @testset "Plm" begin
         @test computePlmcostheta(0,10) ≈ computePlmcostheta(NorthPole(),10)
      end
      @testset "Ylm" begin
         @test computeYlm(0,0,10) ≈ computeYlm(NorthPole(),10)
         @test computeYlm(0,0,10) ≈ computeYlm(NorthPole(),π/2,10)
      end
      @testset "trignometric functions" begin
          @test cos(NorthPole()) == 1
          @test sec(NorthPole()) == 1
          @test sin(NorthPole()) == 0
      end
   end
   @testset "south pole" begin
      @testset "Plm" begin
         @test computePlmcostheta(π,10) ≈ computePlmcostheta(SouthPole(),10)
      end
      @testset "Ylm" begin
         @test computeYlm(π,0,10) ≈ computeYlm(SouthPole(),10)
         @test computeYlm(π,0,10) ≈ computeYlm(SouthPole(),π/2,10)
      end
      @testset "trignometric functions" begin
          @test cos(SouthPole()) == -1
          @test sec(SouthPole()) == -1
          @test sin(SouthPole()) == 0
      end
   end
end

@testset "accuracy" begin

    @testset "lmax 1000" begin
        lmax = 1000

        coeff1 = SphericalHarmonics.compute_coefficients(big(lmax))
        coeff2 = SphericalHarmonics.compute_coefficients(lmax)

        Y1 = SphericalHarmonics.allocate_y(Complex{BigFloat}, lmax)
        Y2 = SphericalHarmonics.allocate_y(Complex{Float64}, lmax)

        P1 = SphericalHarmonics.allocate_p(BigFloat, lmax)
        P2 = SphericalHarmonics.allocate_p(Float64, lmax)
        
        for θ in LinRange(0, big(pi), 10)
            computePlmcostheta!(P1, θ, lmax, coeff1)
            computePlmcostheta!(P2, Float64(θ), lmax, coeff2)
            for ϕ in LinRange(0, 2big(pi), 10)
                computeYlm!(Y1, P1, θ, ϕ, lmax)
                computeYlm!(Y2, P2, Float64(θ), Float64(ϕ), lmax)
                @test all(isapprox.(ComplexF64.(Y1), Y2, atol=1e-11,  rtol=1e-11))
            end
        end
    end
    @testset "lmax 100" begin
        lmax = 100

        coeff1 = SphericalHarmonics.compute_coefficients(big(lmax))
        coeff2 = SphericalHarmonics.compute_coefficients(lmax)

        Y1 = SphericalHarmonics.allocate_y(Complex{BigFloat}, lmax)
        Y2 = SphericalHarmonics.allocate_y(Complex{Float64}, lmax)

        P1 = SphericalHarmonics.allocate_p(BigFloat, lmax)
        P2 = SphericalHarmonics.allocate_p(Float64, lmax)
        
        for θ in LinRange(0, big(pi), 10)
            computePlmcostheta!(P1, θ, lmax, coeff1)
            computePlmcostheta!(P2, Float64(θ), lmax, coeff2)
            for ϕ in LinRange(0, 2big(pi), 10)
                computeYlm!(Y1, P1, θ, ϕ, lmax)
                computeYlm!(Y2, P2, Float64(θ), Float64(ϕ), lmax)
                @test all(isapprox.(ComplexF64.(Y1), Y2, atol=1e-13,  rtol=1e-13))
            end
        end 
    end
end