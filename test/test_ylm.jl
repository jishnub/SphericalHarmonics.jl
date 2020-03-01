@testset "Ylm" begin

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

@info("Test 1: check complex spherical harmonics against explicit expressions")
nsamples = 30
for n = 1:nsamples
   θ = rand() * π
   φ = (rand()-0.5) * 2*π
   Y = compute_y(3, cos(θ), φ)
   Yex = explicit_shs(θ, φ)
   @test Y ≈ Yex
end
println()

end # @testset

@testset "Pole" begin
   @testset "North pole" begin
      @testset "Plm" begin
         @test compute_p(10,1) ≈ compute_p(10,NorthPole())
      end
      @testset "Ylm" begin
         @test compute_y(10,1,0) ≈ compute_y(10,NorthPole())
         @test compute_y(10,1,0) ≈ compute_y(10,NorthPole(),π/2)
      end
   end
   @testset "south pole" begin
      @testset "Plm" begin
         @test compute_p(10,-1) ≈ compute_p(10,SouthPole())
      end
      @testset "Ylm" begin
         @test compute_y(10,-1,0) ≈ compute_y(10,SouthPole())
         @test compute_y(10,-1,0) ≈ compute_y(10,SouthPole(),π/2)
      end
   end
end

@testset "multiples of π/2" begin
   @testset "integer multiplication" begin
      @testset "Piby2" begin
         @test 0*Piby2() == TwoPi()
         @test 1*Piby2() == Piby2()
         @test 2*Piby2() == Pi()
         @test 3*Piby2() == ThreePiby2()
         @test 4*Piby2() == TwoPi()
         @test 5*Piby2() == Piby2()
      end
      @testset "Pi" begin
         @test 0*Pi() == TwoPi()
         @test 1*Pi() == Pi()
         @test 2*Pi() == TwoPi()
         @test 3*Pi() == Pi()
         @test 4*Pi() == TwoPi()
         @test 5*Pi() == Pi()
      end
      @testset "ThreePiby2" begin
         @test 0*ThreePiby2() == TwoPi()
         @test 1*ThreePiby2() == ThreePiby2()
         @test 2*ThreePiby2() == Pi()
         @test 3*ThreePiby2() == Piby2()
         @test 4*ThreePiby2() == TwoPi()
         @test 5*ThreePiby2() == ThreePiby2()
      end
      @testset "TwoPi" begin
         @test 0*TwoPi() == TwoPi()
         @test 1*TwoPi() == TwoPi()
         @test 2*TwoPi() == TwoPi()
         @test 3*TwoPi() == TwoPi()
         @test 4*TwoPi() == TwoPi()
         @test 5*TwoPi() == TwoPi()
      end
   end
   @testset "float multiplication" begin
      @testset "Piby2" begin
         for n in -5:5
            @test float(n)*Piby2() == float(n)*π/2
         end
      end
      @testset "Pi" begin
         for n in -5:5
            @test float(n)*Pi() == float(n)*π
         end
      end
      @testset "ThreePiby2" begin
         for n in -5:5
            @test float(n)*ThreePiby2() == float(n)*3π/2
         end
      end
      @testset "TwoPi" begin
         for n in -5:5
            @test float(n)*TwoPi() == float(n)*2π
         end
      end
   end
   @testset "cis" begin
      @test cis(Piby2()) ≈ cis(π/2)
      @test cis(Pi()) ≈ cis(π)
      @test cis(ThreePiby2()) ≈ cis(3π/2)
      @test cis(TwoPi()) ≈ cis(2π)
      @test cis(TwoPi()) ≈ cis(0)
   end
   @testset "Y" begin
      @testset "cosθ specified" begin
         @test compute_y(10,0.5,Piby2()) ≈ compute_y(10,0.5,π/2)
         @test compute_y(10,0.5,Pi()) ≈ compute_y(10,0.5,π)
         @test compute_y(10,0.5,ThreePiby2()) ≈ compute_y(10,0.5,3π/2)
         @test compute_y(10,0.5,TwoPi()) ≈ compute_y(10,0.5,2π)
      end
      @testset "special points" begin
         @testset "north pole" begin
            @test compute_y(10,NorthPole(),Piby2()) ≈ compute_y(10,NorthPole(),π/2)
            @test compute_y(10,NorthPole(),Pi()) ≈ compute_y(10,NorthPole(),π)
            @test compute_y(10,NorthPole(),ThreePiby2()) ≈ compute_y(10,NorthPole(),3π/2)
            @test compute_y(10,NorthPole(),TwoPi()) ≈ compute_y(10,NorthPole(),2π)
         end
         @testset "south pole" begin
            @test compute_y(10,SouthPole(),Piby2()) ≈ compute_y(10,SouthPole(),π/2)
            @test compute_y(10,SouthPole(),Pi()) ≈ compute_y(10,SouthPole(),π)
            @test compute_y(10,SouthPole(),ThreePiby2()) ≈ compute_y(10,SouthPole(),3π/2)
            @test compute_y(10,SouthPole(),TwoPi()) ≈ compute_y(10,SouthPole(),2π)
         end
      end
   end
end
