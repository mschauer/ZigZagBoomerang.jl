using ZigZagBoomerang
using Documenter

makedocs(;
    modules=[ZigZagBoomerang],
    authors="mschauer <moritzschauer@web.de> and contributors",
    repo="https://github.com/mschauer/ZigZagBoomerang.jl/blob/{commit}{path}#L{line}",
    sitename="ZigZagBoomerang.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mschauer.github.io/ZigZagBoomerang.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mschauer/ZigZagBoomerang.jl",
)
