using SphericalHarmonics
using Test

import SphericalHarmonics: NorthPole, SouthPole

@test isempty(Test.detect_ambiguities(Base, Core, SphericalHarmonics))

@testset "Ylm explicit" begin

    function explicit_shs(θ, φ)
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

    for θ in LinRange(0, π, 100), ϕ in LinRange(0, 2π, 200)
        Y = computeYlm(θ, ϕ, 3)
        Yex = explicit_shs(θ, ϕ)
        @test Y ≈ Yex
    end

    @testset "Pole" begin
        @testset "NorthPole" begin
            Y = computeYlm(NorthPole(), 3)
            Yex = explicit_shs(0, 0)
            @test Y ≈ Yex
        end
        @testset "SouthPole" begin
            Y = computeYlm(SouthPole(), 3)
            Yex = explicit_shs(π, 0)
            @test Y ≈ Yex 
        end
    end

end # @testset

@testset "Pole" begin
   @testset "one" begin
      @test one(NorthPole()) == 1       
      @test one(SouthPole()) == 1       
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
