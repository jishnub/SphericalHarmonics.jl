using Documenter
using SphericalHarmonics

DocMeta.setdocmeta!(SphericalHarmonics, :DocTestSetup, :(using SphericalHarmonics); recursive=true)

makedocs(;
    modules=[SphericalHarmonics],
    authors="Jishnu Bhattacharya",
    repo="https://github.com/jishnub/SphericalHarmonics.jl/blob/{commit}{path}#L{line}",
    sitename="SphericalHarmonics.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jishnub.github.io/SphericalHarmonics.jl",
        assets=String[],
    ),
    pages=[
        "Reference" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jishnub/SphericalHarmonics.jl",
)
