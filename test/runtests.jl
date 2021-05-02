using SphericalHarmonics
using SphericalHarmonicModes
using Test
using HCubature
using Aqua
using LegendrePolynomials
using WignerD
using OffsetArrays

import SphericalHarmonics: NorthPole, SouthPole, allocate_y, allocate_p, RealHarmonics, ComplexHarmonics

@testset "project quality" begin
    if VERSION >= v"1.6"
        Aqua.test_all(SphericalHarmonics, ambiguities = (recursive = false,))
    else
        Aqua.test_all(SphericalHarmonics, ambiguities = false)
    end
end

@testset "allocate" begin
    lmax = 4
    @test size(allocate_y(lmax)) == size(allocate_y(ComplexF64, lmax))
    @test eltype(allocate_y(Complex{BigFloat}, lmax)) == Complex{BigFloat}
    @test eltype(allocate_y(BigFloat, lmax)) == BigFloat

    @test size(allocate_p(lmax)) == size(allocate_p(Float64, lmax))
    @test eltype(allocate_p(BigFloat, lmax)) == BigFloat
end

@testset "Ylm explicit" begin

    function explicit_shs_complex_fullrange(θ, φ)
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

    function explicit_shs_complex_zeroto(θ, φ)
        Y00 = 0.5 * sqrt(1/π)
        Y10 = 0.5 * sqrt(3/π)*cos(θ)
        Y11 = -0.5 * sqrt(3/(2*π))*sin(θ)*exp(im*φ)
        Y20 = 0.25 * sqrt(5/π)*(3*cos(θ)^2 - 1)
        Y21 = -0.5 * sqrt(15/(2*π))*sin(θ)*cos(θ)*exp(im*φ)
        Y22 = 0.25 * sqrt(15/(2*π))*sin(θ)^2*exp(2*im*φ)
        Y30 = 1/4 * sqrt(7/π) * (-3 * cos(θ) + 5 * cos(θ)^3)
        Y31 = -(1/8) * exp(im * φ) * sqrt(21/π) * (-1 + 5 * cos(θ)^2) * sin(θ)
        Y32 = 1/4 * exp(2 * im * φ) * sqrt(105/(2*π)) * cos(θ) * sin(θ)^2
        Y33 = -(1/8) * exp(3 * im * φ) * sqrt(35/π) * sin(θ)^3
        return [Y00, Y10, Y11, Y20, Y21, Y22, Y30, Y31, Y32, Y33]
    end

    @testset "complex harmonics" begin

        for θ in LinRange(0, π, 100), ϕ in LinRange(0, 2π, 200)
            Y = computeYlm(θ, ϕ, 3)
            Yex = explicit_shs_complex_fullrange(θ, ϕ)
            for (ind,mode) in enumerate(ML(0:3))
                @test Y[mode] ≈ Yex[ind]
            end

            Y = computeYlm(θ, ϕ, 3, ZeroTo)
            Yex = explicit_shs_complex_zeroto(θ, ϕ)
            for (ind,mode) in enumerate(ML(0:3, ZeroTo))
                @test Y[mode] ≈ Yex[ind]
            end
        end
    end

    @testset "real harmonics" begin
        function explicit_shs_real_fullrange(θ, φ)
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

        function explicit_shs_real_zeroto(θ, φ)
            Y00 = 0.5 * sqrt(1/π)
            Y10 = 0.5 * sqrt(3/π)*cos(θ)
            Y11 = -0.5 * sqrt(3/π)*sin(θ)*cos(φ)
            Y20 = 0.25 * sqrt(5/π)*(3*cos(θ)^2 - 1)
            Y21 = -0.5 * sqrt(15/π)*sin(θ)*cos(θ)*cos(φ)
            Y22 = 0.25 * sqrt(15/π)*sin(θ)^2*cos(2*φ)
            Y30 = 1/4 * sqrt(7/π) * (-3 * cos(θ) + 5 * cos(θ)^3)
            Y31 = -(1/8) * cos(φ) * sqrt(2*21/π) * (-1 + 5 * cos(θ)^2) * sin(θ)
            Y32 = 1/4 * cos(2 * φ) * sqrt(105/π) * cos(θ) * sin(θ)^2
            Y33 = -(1/8) * cos(3 * φ) * sqrt(2*35/π) * sin(θ)^3
            return [Y00, Y10, Y11, Y20, Y21, Y22, Y30, Y31, Y32, Y33]
        end

        for θ in LinRange(0, π, 100), ϕ in LinRange(0, 2π, 100)
            Y = computeYlm(θ, ϕ, 3, FullRange, SphericalHarmonics.RealHarmonics())
            Yex = explicit_shs_real_fullrange(θ, ϕ)
            for (ind,mode) in enumerate(ML(0:3))
                @test Y[mode] ≈ Yex[ind]
            end

            Y = computeYlm(θ, ϕ, 3, ZeroTo, SphericalHarmonics.RealHarmonics())
            Yex = explicit_shs_real_zeroto(θ, ϕ)
            for (ind,mode) in enumerate(ML(0:3, ZeroTo))
                @test Y[mode] ≈ Yex[ind]
            end
        end
    end

    @testset "Pole" begin
        @testset "NorthPole" begin
            Y = computeYlm(NorthPole(), 3)
            Yex = explicit_shs_complex_fullrange(0, 0)
            @test Y ≈ Yex

            Y = computeYlm(NorthPole(), 3, ZeroTo)
            Yex = explicit_shs_complex_zeroto(0, 0)
            @test Y ≈ Yex
        end
        @testset "SouthPole" begin
            Y = computeYlm(SouthPole(), 3)
            Yex = explicit_shs_complex_fullrange(π, 0)
            @test Y ≈ Yex
        end
        @testset "RealHarmonics == ComplexHarmonics" begin
            for x in (NorthPole(), SouthPole())
                YC = computeYlm(x, 4, FullRange, SphericalHarmonics.ComplexHarmonics())
                YR = computeYlm(x, 4, FullRange, SphericalHarmonics.RealHarmonics())
                @test YC == YR

                YC = computeYlm(x, 4, ZeroTo, SphericalHarmonics.ComplexHarmonics())
                YR = computeYlm(x, 4, ZeroTo, SphericalHarmonics.RealHarmonics())
                @test YC == YR
            end
        end
    end
end

@testset "computePlm!" begin
    lmax = 10
    θ = pi/3
    coeff = SphericalHarmonics.compute_coefficients(lmax);
    P = SphericalHarmonics.allocate_p(Float64, lmax);
    computePlmcostheta!(P, θ, lmax, coeff);

    @test P == computePlmcostheta(θ, lmax)

    computePlmcostheta!(P, NorthPole(), lmax, coeff);
    @test P == computePlmcostheta(NorthPole(), lmax)

    @testset "single m" begin
        computePlmcostheta!(P, θ, lmax, coeff);
        P2 = SphericalHarmonics.allocate_p(Float64, lmax);
        P3 = SphericalHarmonics.allocate_p(Float64, lmax);
        computePlmcostheta!(P3, θ, lmax, nothing, coeff);
        @test P3 ≈ P

        for l in 0:lmax, m in 0:l
            computePlmcostheta!(P2, θ, l, m, coeff);
            for l2 in m:l
                @test P2[(l2,m)] == P[(l2,m)]
            end
        end

        @test_throws ArgumentError computePlmcostheta!(P2, θ, 2, 3, coeff)
        @test_throws ArgumentError computePlmcostheta!(P2, θ, 2, -1, coeff)
    end
end

@testset "computePlmx and computePlmcostheta" begin
    θ = pi/3
    @test SphericalHarmonics.computePlmx(cos(θ), 4) ≈ computePlmcostheta(θ, 4)
    @test SphericalHarmonics.computePlmx(cos(θ), lmax = 4) ≈ computePlmcostheta(θ, 4)
    @test SphericalHarmonics.computePlmx(cos(θ), 4) ≈ computePlmcostheta(θ, lmax = 4)
    @test SphericalHarmonics.computePlmx(cos(θ), lmax = 4) ≈ computePlmcostheta(θ, lmax = 4)

    @test SphericalHarmonics.computePlmx(cos(θ), 4, 2) ≈ computePlmcostheta(θ, 4, 2)
    @test SphericalHarmonics.computePlmx(cos(θ), lmax = 4, m = 2) ≈ computePlmcostheta(θ, 4, 2)
    @test SphericalHarmonics.computePlmx(cos(θ), 4, 2) ≈ computePlmcostheta(θ, lmax = 4, m = 2)
    @test SphericalHarmonics.computePlmx(cos(θ), lmax = 4, m = 2) ≈ computePlmcostheta(θ, lmax = 4, m = 2)
end

@testset "computeYlm kwargs" begin
    θ, ϕ = pi/3, pi/3
    lmax = 10

    Y1 = SphericalHarmonics.computeYlm(θ, ϕ, lmax)
    Y2 = SphericalHarmonics.computeYlm(θ, ϕ, lmax = lmax)
    Y3 = SphericalHarmonics.computeYlm(θ, ϕ, lmax = lmax, m_range = FullRange)
    Y4 = SphericalHarmonics.computeYlm(θ, ϕ, lmax = lmax, m_range = FullRange, SHType = SphericalHarmonics.ComplexHarmonics())
    @test Y1 == Y2 == Y3 == Y4

    for m = -lmax:lmax
        Y1 = SphericalHarmonics.computeYlm(θ, ϕ, lmax, m)
        Y2 = SphericalHarmonics.computeYlm(θ, ϕ, lmax = lmax, m = m)
        Y3 = SphericalHarmonics.computeYlm(θ, ϕ, lmax = lmax, m = m, m_range = FullRange)
        Y4 = SphericalHarmonics.computeYlm(θ, ϕ, lmax = lmax, m = m, m_range = FullRange, SHType = SphericalHarmonics.ComplexHarmonics())
        @test Y1 == Y2 == Y3 == Y4
    end

    Y1 = SphericalHarmonics.computeYlm(θ, ϕ, lmax, ZeroTo)
    Y2 = SphericalHarmonics.computeYlm(θ, ϕ, lmax = lmax, m_range = ZeroTo)
    Y3 = SphericalHarmonics.computeYlm(θ, ϕ, lmax = lmax, m_range = ZeroTo, SHType = SphericalHarmonics.ComplexHarmonics())
    @test Y1 == Y2 == Y3

    for m in 0:lmax
        Y1 = SphericalHarmonics.computeYlm(θ, ϕ, lmax, m, ZeroTo)
        Y2 = SphericalHarmonics.computeYlm(θ, ϕ, lmax = lmax, m = m, m_range = ZeroTo)
        Y3 = SphericalHarmonics.computeYlm(θ, ϕ, lmax = lmax, m = m, m_range = ZeroTo, SHType = SphericalHarmonics.ComplexHarmonics())
        @test Y1 == Y2 == Y3
    end

    Y1 = SphericalHarmonics.computeYlm(θ, ϕ, lmax, ZeroTo, SphericalHarmonics.RealHarmonics())
    Y2 = SphericalHarmonics.computeYlm(θ, ϕ, lmax = lmax, m_range = ZeroTo, SHType = SphericalHarmonics.RealHarmonics())
    @test Y1 == Y2
end

@testset "computeYlm!" begin
    θ, ϕ = pi/3, pi/3
    lmax = 10
    P = SphericalHarmonics.computePlmcostheta(θ, lmax)
    Y1 = SphericalHarmonics.allocate_y(ComplexF64, lmax)
    Y2 = SphericalHarmonics.computeYlm(θ, ϕ, lmax)

    SphericalHarmonics.computeYlm!(Y1, θ, ϕ, lmax)
    @test Y1 ≈ Y2

    SphericalHarmonics.computeYlm!(Y1, θ, ϕ; lmax = lmax)
    @test Y1 ≈ Y2

    SphericalHarmonics.computeYlm!(Y1, P, θ, ϕ; lmax = lmax)
    @test Y1 ≈ Y2

    SphericalHarmonics.computeYlm!(Y1, θ, ϕ, lmax, ZeroTo)
    Y2 = SphericalHarmonics.computeYlm(θ, ϕ, lmax, ZeroTo)
    @test Y1[1:length(ML(ZeroTo(lmax),ZeroTo))] ≈ Y2

    @testset "single m" begin
        Y2 = SphericalHarmonics.computeYlm(θ, ϕ, lmax)
        Ym = SphericalHarmonics.allocate_y(ComplexF64, lmax)

        for l in 0:lmax, m in -l:l
            computeYlm!(Ym, P, θ, ϕ, l, m)
            for l2 in abs(m):l
                @test Ym[(l2,m)] == Y2[(l2,m)]
            end
        end

        Y2 = SphericalHarmonics.computeYlm(θ, ϕ; lmax = lmax, SHType = SphericalHarmonics.RealHarmonics())
        Ym = SphericalHarmonics.allocate_y(Float64, lmax)

        for l in 0:lmax, m in -l:l
            computeYlm!(Ym, P, θ, ϕ, l, m, SphericalHarmonicModes.FullRange, SphericalHarmonics.RealHarmonics())
            for l2 in abs(m):l
                @test Ym[(l2,m)] == Y2[(l2,m)]
            end
        end

        @test_throws ArgumentError computeYlm!(Ym, P, θ, ϕ, 2, 3)
        @test_throws ArgumentError computeYlm!(Ym, P, θ, ϕ, 2, -3)
    end
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
        @testset "computePlmcostheta!" begin
            lmax = 3
            P = SphericalHarmonics.allocate_p(lmax)
            for m = 0:lmax
                computePlmcostheta!(P, NorthPole(), lmax, m)

                P2 = computePlmcostheta(NorthPole(), lmax, m)
                P3 = computePlmcostheta(0, lmax, m)

                for l = m:lmax
                    @test P[(l,m)] == P2[(l,m)]
                    @test isapprox(P[(l,m)], P3[(l,m)], atol=1e-10, rtol=1e-10)
                end
            end
        end
        @testset "computeYlm!" begin
            lmax = 3
            Y = SphericalHarmonics.allocate_y(lmax)
            P = computePlmcostheta(NorthPole(), lmax)
            computeYlm!(Y, P, NorthPole(), 0, lmax)
            Y2 = computeYlm(NorthPole(), 0, lmax)
            @test all(Y ≈ Y2)

            for m in -lmax:lmax
                computeYlm!(Y, P, NorthPole(), 0, lmax, m)
                Y2 = computeYlm(NorthPole(), 0, lmax, m)
                Y3 = computeYlm(0, 0, lmax, m)
                for l in abs(m):lmax
                    @test Y[(l,m)] == Y2[(l,m)]
                    @test isapprox(Y[(l,m)], Y3[(l,m)], atol=1e-10, rtol=1e-10)
                end
            end
        end
        @testset "Ylm" begin
            Y0 = computeYlm(0,0,10,ZeroTo)
            Y0R = computeYlm(0,0,10,ZeroTo,RealHarmonics())
            Y1 = computeYlm(0,0,10)
            @test Y1 ≈ computeYlm(NorthPole(),10)
            @test Y1 ≈ computeYlm(NorthPole(),lmax = 10)
            @test Y1 ≈ computeYlm(NorthPole(),π/2,10)
            @test Y0 ≈ computeYlm(NorthPole(),10,ZeroTo)
            @test Y0 ≈ computeYlm(NorthPole(),π/2,10,ZeroTo)
            @test Y0R ≈ computeYlm(NorthPole(),10,ZeroTo,RealHarmonics())
            @test Y0R ≈ computeYlm(NorthPole(),π/2,10,ZeroTo,RealHarmonics())
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
        @testset "computePlmcostheta!" begin
            lmax = 3
            P = SphericalHarmonics.allocate_p(lmax)

            for m = 0:lmax
                computePlmcostheta!(P, SouthPole(), lmax, m)

                P2 = computePlmcostheta(SouthPole(), lmax, m)
                P3 = computePlmcostheta(π, lmax, m)
                for l = m:lmax
                    @test P[(l,m)] == P2[(l,m)]
                    @test isapprox(P[(l,m)], P3[(l,m)], atol=1e-10, rtol=1e-10)
                end
            end
        end
        @testset "computeYlm!" begin
            lmax = 3
            Y = SphericalHarmonics.allocate_y(lmax)
            P = computePlmcostheta(SouthPole(), lmax)
            computeYlm!(Y, P, SouthPole(), 0, lmax)
            Y2 = computeYlm(SouthPole(), 0, lmax)
            @test all(Y ≈ Y2)

            for m = -lmax:lmax
                computeYlm!(Y, P, SouthPole(), 0, lmax, m)
                Y2 = computeYlm(SouthPole(), 0, lmax, m)
                Y3 = computeYlm(π, 0, lmax, m)
                for l in abs(m):lmax
                    @test Y[(l,m)] == Y2[(l,m)]
                    @test isapprox(Y[(l,m)], Y2[(l,m)], atol=1e-10, rtol=1e-10)
                end
            end
        end
        @testset "Ylm" begin
            @test computeYlm(π,0,10) ≈ computeYlm(SouthPole(),10)
            @test computeYlm(π,0,10) ≈ computeYlm(SouthPole(),π/2,10)
            @test computeYlm(π,0,10,ZeroTo) ≈ computeYlm(SouthPole(),10,ZeroTo)
            @test computeYlm(π,0,10,ZeroTo) ≈ computeYlm(SouthPole(),π/2,10,ZeroTo)
            @test computeYlm(π,0,10,ZeroTo,RealHarmonics()) ≈ computeYlm(SouthPole(),10,ZeroTo,RealHarmonics())
            @test computeYlm(π,0,10,ZeroTo,RealHarmonics()) ≈ computeYlm(SouthPole(),π/2,10,ZeroTo,RealHarmonics())
        end
        @testset "trignometric functions" begin
            @test cos(SouthPole()) == -1
            @test sec(SouthPole()) == -1
            @test sin(SouthPole()) == 0
        end
    end
end

@testset "precision" begin

    isapproxdefault((x,y); kw...) = isapprox(ComplexF64(x), y; kw...)

    @testset "θ = pi/2" begin
        P = computePlmcostheta(big(pi)/2, lmax = big(10))
        @test P[(1,1)] ≈ -(√(3/big(pi)))/2
        @test P[(2,2)] ≈ (√(15/big(pi)))/4
        @test P[(3,3)] ≈ -(√(35/(2big(pi))))/4
        @test P[(4,4)] ≈ (√(35/big(pi)))*3/16
        @test P[(5,5)] ≈ -(√(77/(2big(pi))))*3/16
        @test P[(6,6)] ≈ (√(3003/(2big(pi))))/32
        @test P[(7,7)] ≈ -(√(715/big(pi)))*3/64
        @test P[(8,8)] ≈ (√(12155/big(pi)))*3/256
        @test P[(9,9)] ≈ -(√(230945/(2big(pi))))/256
        @test P[(10,10)] ≈ (√(969969/(2big(pi))))/512
    end

    θ_range_big = LinRange(0, big(pi), 20)
    ϕ_range_big = LinRange(0, 2big(pi), 5)

    @testset "lmax 1000" begin
        lmax = 1000

        coeff_big = SphericalHarmonics.compute_coefficients(big(lmax))
        coeff_Float64 = SphericalHarmonics.compute_coefficients(lmax)

        Y1 = SphericalHarmonics.allocate_y(Complex{BigFloat}, lmax)
        Y2 = SphericalHarmonics.allocate_y(Complex{Float64}, lmax)

        P1 = SphericalHarmonics.allocate_p(BigFloat, lmax)
        P2 = SphericalHarmonics.allocate_p(Float64, lmax)

        for θ in θ_range_big
            computePlmcostheta!(P1, θ, lmax, coeff_big)
            computePlmcostheta!(P2, Float64(θ), lmax, coeff_Float64)
            for ϕ in ϕ_range_big
                computeYlm!(Y1, P1, θ, ϕ, lmax)
                computeYlm!(Y2, P2, Float64(θ), Float64(ϕ), lmax)
                @test all(x -> isapproxdefault(x, atol=1e-11,  rtol=1e-11), zip(Y1, Y2))
            end
        end
    end
    @testset "lmax 100" begin
        lmax = 100

        coeff_big = SphericalHarmonics.compute_coefficients(big(lmax))
        coeff_Float64 = SphericalHarmonics.compute_coefficients(lmax)

        Y1 = SphericalHarmonics.allocate_y(Complex{BigFloat}, lmax)
        Y2 = SphericalHarmonics.allocate_y(Complex{Float64}, lmax)

        P1 = SphericalHarmonics.allocate_p(BigFloat, lmax)
        P2 = SphericalHarmonics.allocate_p(Float64, lmax)

        for θ in θ_range_big
            computePlmcostheta!(P1, θ, lmax, coeff_big)
            computePlmcostheta!(P2, Float64(θ), lmax, coeff_Float64)
            for ϕ in ϕ_range_big
                computeYlm!(Y1, P1, θ, ϕ, lmax)
                computeYlm!(Y2, P2, Float64(θ), Float64(ϕ), lmax)
                @test all(x -> isapproxdefault(x, atol=1e-13,  rtol=1e-13), zip(Y1, Y2))
            end
        end
    end
end

@testset "single mode" begin
    @testset "Associated Legendre Polynomials" begin
        lmax = 20
        coeff = SphericalHarmonics.compute_coefficients(lmax)

        @testset "explicit coeff" begin
            P = SphericalHarmonics.allocate_p(lmax)
            for θ in LinRange(0, pi, 10)
                computePlmcostheta!(P, θ, lmax, coeff)
                for l = 0:lmax, m = 0:l
                    Plm = SphericalHarmonics.associatedLegendre(θ, l, m, coeff)
                    @test isapprox(P[(l,m)], Plm, atol = 1e-14, rtol = 1e-14)
                end
            end

            P = computePlmcostheta(SphericalHarmonics.NorthPole(), lmax = lmax)
            for l = 0:lmax, m = 0:l
                Plm = SphericalHarmonics.associatedLegendre(SphericalHarmonics.NorthPole(), l, m, coeff)
                @test isapprox(P[(l,m)], Plm, atol = 1e-14, rtol = 1e-14)
            end

            P = computePlmcostheta(SphericalHarmonics.SouthPole(), lmax = lmax)
            for l = 0:lmax, m = 0:l
                Plm = SphericalHarmonics.associatedLegendre(SphericalHarmonics.SouthPole(), l, m, coeff)
                @test isapprox(P[(l,m)], Plm, atol = 1e-14, rtol = 1e-14)
            end
        end
        @testset "implicit coeff" begin
            P = SphericalHarmonics.allocate_p(lmax)
            for θ in LinRange(0, pi, 10)
                computePlmcostheta!(P, θ, lmax, coeff)
                for l = 0:lmax, m = 0:l
                    Plm = SphericalHarmonics.associatedLegendre(θ, l, m)
                    Plm2 = SphericalHarmonics.associatedLegendre(θ, l = l, m = m)
                    @test Plm2 == Plm
                    @test isapprox(P[(l,m)], Plm, atol = 1e-14, rtol = 1e-14)
                end
            end

            P = computePlmcostheta(SphericalHarmonics.NorthPole(), lmax = lmax)
            for l = 0:lmax, m = 0:l
                Plm = SphericalHarmonics.associatedLegendre(SphericalHarmonics.NorthPole(), l, m)
                @test isapprox(P[(l,m)], Plm, atol = 1e-14, rtol = 1e-14)
            end

            P = computePlmcostheta(SphericalHarmonics.SouthPole(), lmax = lmax)
            for l = 0:lmax, m = 0:l
                Plm = SphericalHarmonics.associatedLegendre(SphericalHarmonics.SouthPole(), l, m)
                @test isapprox(P[(l,m)], Plm, atol = 1e-14, rtol = 1e-14)
            end
        end
    end
    @testset "SphericalHarmonics" begin
        lmax = 20
        coeff = SphericalHarmonics.compute_coefficients(lmax);

        P = SphericalHarmonics.allocate_p(lmax)
        Y = SphericalHarmonics.allocate_y(lmax)
        Yreal = SphericalHarmonics.allocate_y(Float64, lmax)
        @testset "ComplexHarmonics" begin
            @testset "explicit coeff" begin
                for θ in LinRange(0, pi, 10)
                    computePlmcostheta!(P, θ, lmax, coeff)
                    for ϕ in LinRange(0, 2pi, 10)[1:end-1]
                        computeYlm!(Y, P, θ, ϕ, lmax = lmax)
                        for l = 0:lmax, m = -l:l
                            Ylm = SphericalHarmonics.sphericalharmonic(θ, ϕ, l, m,
                                SphericalHarmonics.ComplexHarmonics(), coeff)

                            Ylm2 = SphericalHarmonics.sphericalharmonic(θ, ϕ; l = l, m = m,
                                SHType = SphericalHarmonics.ComplexHarmonics(), coeff = coeff)

                            @test Ylm == Ylm2

                            @test isapprox(Y[(l,m)], Ylm, atol = 1e-14, rtol = 1e-14)
                        end
                    end
                end
            end
            @testset "implicit coeff" begin
                for θ in LinRange(0, pi, 10)
                    computePlmcostheta!(P, θ, lmax, coeff)
                    for ϕ in LinRange(0, 2pi, 10)[1:end-1]
                        computeYlm!(Y, P, θ, ϕ, lmax = lmax)
                        for l = 0:lmax, m = -l:l
                            Ylm = SphericalHarmonics.sphericalharmonic(θ, ϕ, l, m,
                                SphericalHarmonics.ComplexHarmonics())

                            Ylm2 = SphericalHarmonics.sphericalharmonic(θ, ϕ; l = l, m = m,
                                SHType = SphericalHarmonics.ComplexHarmonics())

                            @test Ylm == Ylm2

                            @test isapprox(Y[(l,m)], Ylm, atol = 1e-14, rtol = 1e-14)
                        end
                    end
                end
            end
        end
        @testset "RealHarmonics" begin
            @testset "explicit coeff" begin
                for θ in LinRange(0, pi, 10)
                    computePlmcostheta!(P, θ, lmax, coeff)
                    for ϕ in LinRange(0, 2pi, 10)[1:end-1]
                        computeYlm!(Yreal, P, θ, ϕ, lmax = lmax, SHType = SphericalHarmonics.RealHarmonics())
                        for l = 0:lmax, m = -l:l
                            Ylm = SphericalHarmonics.sphericalharmonic(θ, ϕ, l, m,
                                SphericalHarmonics.RealHarmonics(), coeff)

                            Ylm2 = SphericalHarmonics.sphericalharmonic(θ, ϕ; l = l, m = m,
                                SHType = SphericalHarmonics.RealHarmonics(), coeff = coeff)

                            @test Ylm == Ylm2

                            @test isapprox(Yreal[(l,m)], Ylm, atol = 1e-14, rtol = 1e-8)
                        end
                    end
                end
            end
            @testset "implicit coeff" begin
                for θ in LinRange(0, pi, 10)
                    computePlmcostheta!(P, θ, lmax, coeff)
                    for ϕ in LinRange(0, 2pi, 10)[1:end-1]
                        computeYlm!(Yreal, P, θ, ϕ, lmax = lmax, SHType = SphericalHarmonics.RealHarmonics())
                        for l = 0:lmax, m = -l:l
                            Ylm = SphericalHarmonics.sphericalharmonic(θ, ϕ, l, m,
                                SphericalHarmonics.RealHarmonics())

                            Ylm2 = SphericalHarmonics.sphericalharmonic(θ, ϕ; l = l, m = m,
                                SHType = SphericalHarmonics.RealHarmonics())

                            @test Ylm == Ylm2

                            @test isapprox(Yreal[(l,m)], Ylm, atol = 1e-14, rtol = 1e-8)
                        end
                    end
                end
            end
        end
    end
end

@testset "orthonormality" begin

    @testset "Associated Legendre Polynomials" begin
        function testnorm(f)
            I, E = hcubature(f, [0], [π]);
            @test I ≈ 1/π
        end
        function testortho(f)
            I, E = hcubature(f, [0], [π], atol=1e-10);
            @test isapprox(abs(I), 0, atol = max(abs(E), 1e-10))
        end
        lmax = 10
        coeff = SphericalHarmonics.compute_coefficients(lmax)
        P = SphericalHarmonics.allocate_p(lmax)

        @testset "Normalization" begin
            function f!(x, P, l, m)
                θ = x[1]
                computePlmcostheta!(P, θ, l, m, coeff)
                sin(θ) * P[(l,m)]^2
            end

            for m = 0:lmax, l = m:lmax
                testnorm(x -> f!(x, P, l, m))
            end
        end

        @testset "Orthogonality" begin
            @testset "same m, different l" begin
                function f!(x, P, k, l, m)
                    θ = x[1]
                    computePlmcostheta!(P, θ, max(k, l), m, coeff)
                    sin(θ) * P[(k,m)] * P[(l,m)]
                end

                for m = 0:lmax, l = m:lmax, k = m:lmax
                    k == l && continue
                    testortho(x -> f!(x, P, k, l, m))
                end
            end

            @testset "same l, different m" begin
                function f!(x, P, l, m1, m2)
                    θ = x[1]
                    computePlmcostheta!(P, θ, l, coeff)
                    P[(l,m1)] * P[(l,m2)] / sin(θ)
                end

                for l = 0:lmax, m1 in 0:l, m2 in m1+1:l
                    testortho(x -> f!(x, P, l, m1, m2))
                end
            end
        end
    end

    @testset "SphericalHarmonics" begin
        function testnorm(f)
            I, E = hcubature(f, [0, 0], [π, 2π]);
            @test all(x->isapprox(x, 1), I)
        end
        function testortho(f)
            I, E = hcubature(f, [0, 0], [π, 2π], atol=1e-10);
            @test isapprox(abs(I), 0, atol = max(abs(E), 1e-10))
        end

        @testset "Normalization" begin
            lmax = 10
            @testset "Complex harmonics" begin
                f(θϕ) = sin(θϕ[1]) .* abs2.(computeYlm(θϕ[1], θϕ[2], lmax = lmax))
                testnorm(f)
            end
            @testset "Real harmonics" begin
                f(θϕ) = sin(θϕ[1]) .* abs2.(computeYlm(θϕ[1], θϕ[2], lmax = lmax, SHType = SphericalHarmonics.RealHarmonics()))
                testnorm(f)
            end
        end
        @testset "Orthogonality" begin
            lmax = 3
            coeff = SphericalHarmonics.compute_coefficients(lmax)

            @testset "Complex harmonics" begin
                Y = SphericalHarmonics.allocate_y(lmax)
                P = SphericalHarmonics.allocate_p(lmax)

                function f!(θϕ, Y, P, l, m1, m2)
                    θ, ϕ = θϕ
                    computePlmcostheta!(P, θ, l, coeff)
                    computeYlm!(Y, P, θ, ϕ, l)
                    sin(θ) * Y[(l,m1)] * conj(Y[(l,m2)])
                end

                for l in 0:lmax, m1 in -l:0, m2 in m1+1:l
                    testortho(x->f!(x, Y, P, l, m1, m2))
                end
            end

            @testset "Real harmonics" begin
                Y = SphericalHarmonics.allocate_y(Float64, lmax)
                P = SphericalHarmonics.allocate_p(lmax)

                function f!(θϕ, Y, P, l, m1, m2)
                    θ, ϕ = θϕ
                    computePlmcostheta!(P, θ, l, coeff)
                    computeYlm!(Y, P, θ, ϕ, l, SphericalHarmonicModes.FullRange, SphericalHarmonics.RealHarmonics())
                    sin(θ) * Y[(l,m1)] * Y[(l,m2)]
                end

                for l in 0:lmax, m1 in -l:0, m2 in m1+1:l
                    testortho(x->f!(x, Y, P, l, m1, m2))
                end
            end
        end
    end
end

@testset "cache" begin
    @testset "FullRange, ComplexHarmonics" begin
        S = SphericalHarmonics.cache(3);
        @test S.lmax == 3
        θ, ϕ = pi/3, pi/4
        SphericalHarmonics.computePlmx!(S, cos(θ), 1)
        @test S.P.lmax == 1
        P2 = SphericalHarmonics.computePlmx(cos(θ), lmax = 1)
        @test S.P[1:length(P2)] == P2
        SphericalHarmonics.computePlmx!(S, cos(2θ))
        @test S.P.lmax == S.lmax
        @test S.P == SphericalHarmonics.computePlmx(cos(2θ), lmax = 3)
        computePlmcostheta!(S, θ, 3)
        @test S.lmax == 3
        @test S.P == computePlmcostheta(θ, lmax = 3)
        # This should be a no-op
        computePlmcostheta!(S, θ, 3)
        @test S.P == computePlmcostheta(θ, lmax = 3)
        computePlmcostheta!(S, 2θ)
        @test S.P ≈ computePlmcostheta(2θ, lmax = 3)

        computePlmcostheta!(S, θ)
        computeYlm!(S, θ, ϕ, 3)
        @test S.Y == computeYlm(θ, ϕ, 3)
        computePlmcostheta!(S, θ, 4)
        @test S.lmax == 4
        @test S.P == computePlmcostheta(θ, lmax = 4)
        computeYlm!(S, θ, ϕ, 4)
        @test S.Y == computeYlm(θ, ϕ, 4)
        computeYlm!(S, θ, ϕ)
        @test S.Y == computeYlm(θ, ϕ, 4)
        computeYlm!(S, θ, ϕ, 2)
        Y2 = computeYlm(θ, ϕ, 2)
        @test S.Y[1:length(Y2)] == Y2

        computePlmcostheta!(S, NorthPole(), 4)
        @test S.P.cosθ == cos(NorthPole())
        @test S.P == computePlmcostheta(NorthPole(), 4)
    end
    @testset "ZeroTo, ComplexHarmonics" begin
        S = SphericalHarmonics.cache(3, m_range = ZeroTo);
        @test S.lmax == 3
        θ, ϕ = pi/3, pi/4
        SphericalHarmonics.computePlmx!(S, cos(θ), 3)
        @test S.P == SphericalHarmonics.computePlmx(cos(θ), lmax = 3)
        computePlmcostheta!(S, θ, 3)
        @test S.P == computePlmcostheta(θ, lmax = 3)
        # This should be a no-op
        computePlmcostheta!(S, θ, 3)
        @test S.P == computePlmcostheta(θ, lmax = 3)
        computeYlm!(S, θ, ϕ, 3)
        @test S.Y == computeYlm(θ, ϕ, lmax = 3, m_range = ZeroTo)
        computePlmcostheta!(S, θ, 4)
        @test S.lmax == 4
        @test S.P == computePlmcostheta(θ, lmax = 4)
        computeYlm!(S, θ, ϕ, 4)
        @test S.Y == computeYlm(θ, ϕ, lmax = 4, m_range = ZeroTo)
        computeYlm!(S, θ, ϕ)
        @test S.Y == computeYlm(θ, ϕ, lmax = 4, m_range = ZeroTo)
        computeYlm!(S, θ, ϕ, 2)
        Y2 = computeYlm(θ, ϕ, lmax = 2, m_range = ZeroTo)
        @test S.Y[1:length(Y2)] == Y2

        computePlmcostheta!(S, NorthPole(), 4)
        @test S.P.cosθ == cos(NorthPole())
        @test S.P == computePlmcostheta(NorthPole(), 4)
    end
    @testset "FullRange, RealHarmonics" begin
        S = SphericalHarmonics.cache(3, SHType = RealHarmonics());
        @test S.lmax == 3
        θ, ϕ = pi/3, pi/4
        SphericalHarmonics.computePlmx!(S, cos(θ), 3)
        @test S.P == SphericalHarmonics.computePlmx(cos(θ), lmax = 3)
        # This should be a no-op
        SphericalHarmonics.computePlmx!(S, cos(θ), 3)
        @test S.P == SphericalHarmonics.computePlmx(cos(θ), lmax = 3)
        computePlmx!(S, cos(θ), 4)
        @test S.lmax == 4
        @test S.P == computePlmx(cos(θ), lmax = 4)

        S = SphericalHarmonics.cache(3, SHType = RealHarmonics());
        computePlmcostheta!(S, θ, 3)
        @test S.P == computePlmcostheta(θ, lmax = 3)
        # This should be a no-op
        computePlmcostheta!(S, θ, 3)
        @test S.P == computePlmcostheta(θ, lmax = 3)
        computeYlm!(S, θ, ϕ, 3)
        @test S.Y == computeYlm(θ, ϕ, lmax = 3, SHType = RealHarmonics())

        computePlmcostheta!(S, θ, 4)
        @test S.lmax == 4
        @test S.P == computePlmcostheta(θ, lmax = 4)
        computeYlm!(S, θ, ϕ, 4)
        @test S.Y == computeYlm(θ, ϕ, lmax = 4, SHType = RealHarmonics())
        computeYlm!(S, θ, ϕ)
        @test S.Y == computeYlm(θ, ϕ, lmax = 4, SHType = RealHarmonics())
        computeYlm!(S, θ, ϕ, 2)
        Y2 = computeYlm(θ, ϕ, lmax = 2, SHType = RealHarmonics())
        @test S.Y[1:length(Y2)] == Y2

        computePlmcostheta!(S, NorthPole(), 4)
        @test S.P.cosθ == cos(NorthPole())
        @test S.P == computePlmcostheta(NorthPole(), 4)
    end
    @testset "ZeroTo, RealHarmonics" begin
        S = SphericalHarmonics.cache(3, m_range = ZeroTo, SHType = RealHarmonics());
        @test S.lmax == 3
        θ, ϕ = pi/3, pi/4
        SphericalHarmonics.computePlmx!(S, cos(θ), 3)
        @test S.P == SphericalHarmonics.computePlmx(cos(θ), lmax = 3)
        computePlmcostheta!(S, θ, 3)
        @test S.P == computePlmcostheta(θ, lmax = 3)
        # This should be a no-op
        computePlmcostheta!(S, θ, 3)
        @test S.P == computePlmcostheta(θ, lmax = 3)
        computeYlm!(S, θ, ϕ, 3)
        @test S.Y == computeYlm(θ, ϕ, lmax = 3, m_range = ZeroTo, SHType = RealHarmonics())
        computePlmcostheta!(S, θ, 4)
        @test S.lmax == 4
        @test S.P == computePlmcostheta(θ, lmax = 4)
        computeYlm!(S, θ, ϕ, 4)
        @test S.Y == computeYlm(θ, ϕ, lmax = 4, m_range = ZeroTo, SHType = RealHarmonics())
        computeYlm!(S, θ, ϕ)
        @test S.Y == computeYlm(θ, ϕ, lmax = 4, m_range = ZeroTo, SHType = RealHarmonics())
        computeYlm!(S, θ, ϕ, 2)
        Y2 = computeYlm(θ, ϕ, lmax = 2, m_range = ZeroTo, SHType = RealHarmonics())
        @test S.Y[1:length(Y2)] == Y2

        computePlmcostheta!(S, NorthPole(), 4)
        @test S.P.cosθ == cos(NorthPole())
        @test S.P == computePlmcostheta(NorthPole(), 4)
    end
    @testset "show" begin
        S = SphericalHarmonics.cache(3);
        io = IOBuffer()
        show(io, S)
        s = String(take!(io))
        s_exp = "$(SphericalHarmonics.SphericalHarmonicsCache)(Float64, 3, m_range = $(SphericalHarmonicModes.FullRange), SHType = $(SphericalHarmonics.ComplexHarmonics)())"
        @test s == s_exp
        summary(io, S.P)
        s = String(take!(io))
        s_exp = "10-element AssociatedLegendrePolynomials{Float64} for lmax = 3 (uninitialized)"
        @test s == s_exp
        θ = 0
        computePlmcostheta!(S, θ, 3)
        summary(io, S.P)
        s = String(take!(io))
        s_exp = "10-element AssociatedLegendrePolynomials{Float64} for lmax = 3 and cosθ = 1"
        @test s == s_exp

        S = SphericalHarmonics.cache(BigFloat, 3);
        summary(io, S.P)
        s = String(take!(io))
        s_exp = "10-element AssociatedLegendrePolynomials{BigFloat} for lmax = 3 (uninitialized)"
    end
end

@testset "Parity" begin
    lmax = 10
    S1 = SphericalHarmonics.cache(lmax)
    S2 = SphericalHarmonics.cache(lmax)
    for θ in LinRange(0, pi, 10)
        P1 = computePlmcostheta!(S1, θ)
        P2 = computePlmcostheta!(S2, pi - θ)
        for l in 0:lmax, m in 0:l
            @test isapprox(P2[(l,m)], (-1)^(l + m) * P1[(l,m)], atol = 1e-13, rtol = 1e-13)
        end
        for ϕ in LinRange(0, 2pi, 10)
            Y1 = computeYlm!(S1, θ, ϕ)
            Y2 = computeYlm!(S2, pi - θ, pi + ϕ)
            for l in 0:lmax, m in -l:l
                @test isapprox(Y2[(l,m)], (-1)^l * Y1[(l,m)], atol = 1e-13, rtol = 1e-13)
            end
        end
    end
end

@testset "m = 0 Legendre" begin
    # for m = 0, the spherical harmonics are normalized Legendre Polynomials and are purely real
    lmax = 100
    @testset "ComplexHarmonics" begin
        S = SphericalHarmonics.cache(lmax)
        LP = zeros(0:lmax)
        for θ in LinRange(0, pi, 10)
            LegendrePolynomials.collectPl!(LP, cos(θ), lmax = lmax)
            computePlmcostheta!(S, θ)
            for ϕ in LinRange(0, 2pi, 10)
                computeYlm!(S, θ, ϕ)
                for l in 0:lmax
                    @test isapprox(imag(S.Y[(l,0)]), 0, atol = 1e-13)
                    @test isapprox(S.Y[(l,0)], √((2l+1)/4pi) * LP[l], atol = 1e-13, rtol = 1e-13)
                end
            end
        end
    end
    @testset "RealHarmonics" begin
        S = SphericalHarmonics.cache(lmax, SHType = RealHarmonics())
        LP = zeros(0:lmax)
        for θ in LinRange(0, pi, 10)
            LegendrePolynomials.collectPl!(LP, cos(θ), lmax = lmax)
            computePlmcostheta!(S, θ)
            for ϕ in LinRange(0, 2pi, 10)
                computeYlm!(S, θ, ϕ)
                for l in 0:lmax
                    @test isapprox(imag(S.Y[(l,0)]), 0, atol = 1e-13)
                    @test isapprox(S.Y[(l,0)], √((2l+1)/4pi) * LP[l], atol = 1e-13, rtol = 1e-13)
                end
            end
        end
    end
end

@testset "rotation of coordinates" begin
    # Ylm(θ, ϕ) = Yl0(0, 0) D^l_{0,-m}(0, θ, ϕ)
    lmax = 10
    S = SphericalHarmonics.cache(lmax);
    Dvec = OffsetArray([zeros(ComplexF64, 2l+1, 2l+1) for l in 0:lmax], 0:lmax);
    Jyvec = OffsetArray([zeros(ComplexF64, 2l+1, 2l+1) for l in 0:lmax], 0:lmax);
    @testset "NorthPole" begin
        Yl0NP = computeYlm(NorthPole(), 0, lmax = lmax)
        for θ in LinRange(0, pi, 10)
            computePlmcostheta!(S, θ)
            for ϕ in LinRange(0, 2pi, 10)
                Ylmθϕ = computeYlm!(S, θ, ϕ)
                for l in 0:lmax
                    Dp = wignerD!(Dvec[l], l, 0, θ, ϕ, Jyvec[l])
                    D = OffsetArray(Dp, -l:l, -l:l)
                    for m in -l:l
                        @test isapprox(Ylmθϕ[(l,m)], Yl0NP[(l,0)] * D[0,-m], atol = 1e-13, rtol = 1e-8)
                    end
                end
            end
        end
    end
    @testset "arbitrary point" begin
        # We represent an arbitrary point in two coordinate systems
        # We assume that it has coordinates (θ1,ϕ1) in the first frame and (θ2,ϕ2) in the second frame
        # The frames may be shown to be related through S2 = Rz(ϕ1)Ry(θ1-θ2)Rz(-ϕ2) S1,
        # where Ri(ω) represents a rotation about the axis i by an angle ω.
        # We note that the rotation operator may be represented as D = exp(-iαJz)exp(-iβJy)exp(-iγJz)
        # Comparing, we obtain α = ϕ1, β = θ1-θ2, γ = -ϕ2
        # The Spherical harmonics transform through Ylm(n2) = ∑_m′ conj(D^l_{m,m′}(-γ,-β,-α)) Ylm(n1)
        # Substituting, we obtain the matrix conj(D^l_{m,m′}(ϕ2,θ2-θ1,-ϕ1))
        # We note that choosing (θ1,ϕ1) = (0,0) in this gives us the matrix conj(D^l_{m,m′}(ϕ2,θ2,0)),
        # which, using symmetries, reduces to the previous case with the point at the north pole

        θ1, ϕ1 = pi/6, pi/3

        Y1lm = computeYlm(θ1, ϕ1, lmax = lmax)
        for θ2 in LinRange(0, pi, 10)
            computePlmcostheta!(S, θ2)
            for ϕ2 in LinRange(0, 2pi, 10)
                Ylmθϕ = computeYlm!(S, θ2, ϕ2)
                for l in 0:lmax
                    Dp = wignerD!(Dvec[l], l, ϕ2, θ2-θ1, -ϕ1, Jyvec[l])
                    D = OffsetArray(Dp, -l:l, -l:l)
                    for m in -l:l
                        rotY1lm = sum(conj(D[m, m′]) * Y1lm[(l,m′)] for m′ = -l:l)
                        @test isapprox(Ylmθϕ[(l,m)], rotY1lm, atol = 1e-13, rtol = 1e-8)
                    end
                end
            end
        end
    end
end

@testset "negative theta" begin
    lmax = 10
    S1 = SphericalHarmonics.cache(lmax);
    S2 = SphericalHarmonics.cache(lmax);
    for θ in LinRange(0, pi, 10)
        computePlmcostheta!(S1, θ)
        θ2 = -θ
        computePlmcostheta!(S2, θ2)
        for ϕ in LinRange(0, 2pi, 10)
            Ylmθϕ = computeYlm!(S1, θ, ϕ)
            Ylmθ2ϕ = computeYlm!(S2, θ2, ϕ)
            for l in 0:lmax, m in -l:l
                @test isapprox(Ylmθ2ϕ[(l,m)], (-1)^m * Ylmθϕ[(l,m)], atol = 1e-13, rtol = 1e-8)
            end
        end
    end
end

@testset "theta ± nπ" begin
    lmax = 10
    S1 = SphericalHarmonics.cache(lmax);
    S2 = SphericalHarmonics.cache(lmax);
    S3 = SphericalHarmonics.cache(lmax);
    for θ in LinRange(0, pi, 10)
        computePlmcostheta!(S1, θ)
        θ2 = pi + θ
        computePlmcostheta!(S2, θ2)
        θ3 = 2pi + θ
        computePlmcostheta!(S3, θ3)
        for ϕ in LinRange(0, 2pi, 10)
            Ylmθϕ = computeYlm!(S1, θ, ϕ)
            Ylmθ2ϕ = computeYlm!(S2, θ2, ϕ)
            Ylmθ3ϕ = computeYlm!(S3, θ3, ϕ)
            for l in 0:lmax, m in -l:l
                @test isapprox(Ylmθ2ϕ[(l,m)], (-1)^l * Ylmθϕ[(l,m)], atol = 1e-13, rtol = 1e-8)
                @test isapprox(Ylmθ3ϕ[(l,m)], Ylmθϕ[(l,m)], atol = 1e-13, rtol = 1e-8)
            end
        end
    end
end
