IrrationalConstants.@irrational _sqrt3by4pi √(3 / (4 * big(pi))) _Sqrt3by4pi
IrrationalConstants.@irrational _sqrt3by2pi √(3 / (2 * big(pi))) _Sqrt3by2pi

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
