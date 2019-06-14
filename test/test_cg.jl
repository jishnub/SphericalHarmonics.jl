
@testset "Clebsch-Gordan" begin
using PyCall, Test, SphericalHarmonics

sympy = pyimport("sympy")
spin = pyimport("sympy.physics.quantum.spin")


pycg(j1, m1, j2, m2, j3, m3, T=Float64) =
      spin.CG(j1, m1, j2, m2, j3, m3).doit().evalf().__float__()

@info("Testing cg1 implementation against sympy ... ")
for j1 = 0:2, j2=0:2, j3=0:4
   for m1 = -j1:j1, m2=-j2:j2, m3=-j3:j3
      @test cg1(j1,m1,j2,m2,j3,m3) ≈ pycg(j1,m1, j2,m2, j3,m3)
   end
end

@info("Check CG Coefficients for some higher quantum numbers...")
j1 = 8
j2 = 11
j3 = j1+j2
for m1 = -j1:j1, m2=-j2:j2, m3=-j3:j3
   @test cg1(j1,m1,j2,m2,j3,m3) ≈ pycg(j1,m1, j2,m2, j3,m3)
end

end # @testset
