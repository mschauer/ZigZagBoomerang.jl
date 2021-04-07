In the Julia REPL enter:
```julia
using Pkg
path = mktempdir()
project = download("https://raw.githubusercontent.com/mschauer/ZigZagBoomerang.jl/master/game/Project.toml…", joinpath(path, "Project.toml"))
file = download("https://raw.githubusercontent.com/mschauer/ZigZagBoomerang.jl/master/game/bouncygame.jl…", joinpath(path, "bouncygame.jl"))
include(file)
```
