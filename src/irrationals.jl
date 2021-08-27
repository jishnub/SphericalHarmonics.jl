Base.@irrational _invsqrt2 0.7071067811865476 1/√(big(2))
Base.@irrational _sqrtpi 1.772453850905516 √(big(pi))
Base.@irrational _invsqrt2pi 0.3989422804014327 1/√(2*big(pi))
Base.@irrational _sqrt3by4pi 0.4886025119029199 √(3/(4big(pi)))
Base.@irrational _sqrt3by2pi 0.690988298942671 √(3/(2big(pi)))
Base.@irrational _log2pi 1.8378770664093456 log(2big(pi))

"""
    SphericalHarmonics.Pole <: Real

Supertype of [`NorthPole`](@ref) and [`SouthPole`](@ref).
"""
abstract type Pole <: Real end
"""
    SphericalHarmonics.NorthPole <: SphericalHarmonics.Pole

The angle ``\\theta = 0`` in spherical polar coordinates.
"""
struct NorthPole <: Pole end
"""
    SphericalHarmonics.SouthPole <: SphericalHarmonics.Pole

The angle ``\\theta = π`` in spherical polar coordinates.
"""
struct SouthPole <: Pole end

Base.cos(::NorthPole) = one(Float64)
Base.cos(::SouthPole) = -one(Float64)
Base.sin(::Pole) = zero(Float64)

for DT in [:zero, :one]
    @eval Base.$DT(::Type{<:Pole}) = $DT(Float64)
    @eval Base.$DT(x::Pole) = $DT(typeof(x))
end

Base.promote_rule(::Type{<:Pole}, T::Type{<:Real}) = promote_type(Float64, T)

# Return the value of θ corresponding to the poles
Base.Float64(::SouthPole) = Float64(pi)
Base.Float64(::NorthPole) = zero(Float64)

Base.float(p::Pole) = Float64(p)
