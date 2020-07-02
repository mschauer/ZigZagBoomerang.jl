# ZigZagBoomerang

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mschauer.github.io/ZigZagBoomerang.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mschauer.github.io/ZigZagBoomerang.jl/dev/index.html)
[![Build Status](https://travis-ci.com/mschauer/ZigZagBoomerang.jl.svg?branch=master)](https://travis-ci.com/mschauer/ZigZagBoomerang.jl)

## Example
```julia
using ZigZagBoomerang

# negative log-density with respect to Lebesgue
ϕ(x) = cos(2pi*x) + x^2/2 # not needed

# gradient of ϕ(x)
∇ϕ(x) = -2*pi*sin(2*π*x) + x

x0, θ0 = randn(), 1.0
T = 100.0

c = 2π # parameter for the upper bound of the Poisson rate, will error if too small

# ZigZag
out1 = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, c, ZigZag())

# Boomerang with refreshment rate 0.5
out2 = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, c, Boomerang(0.5))
```

## Literature

* Joris Bierkens, Paul Fearnhead, Gareth Roberts: The Zig-Zag Process and Super-Efficient Sampling for Bayesian Analysis of Big Data. *The Annals of Statistics*, 2019, 47. Vol., Nr. 3, pp. 1288-1320. [https://arxiv.org/abs/1607.03188].
* Joris Bierkens, Sebastiano Grazzi, Kengo Kamatani and Gareth Robers: The Boomerang Sampler. *ICML 2020*. [https://arxiv.org/abs/2006.13777].
* Joris Bierkens, Sebastiano Grazzi, Frank van der Meulen, Moritz Schauer: A piecewise deterministic Monte Carlo method for diffusion bridges.  2020. [https://arxiv.org/abs/2001.05889].
* [https://github.com/jbierkens/ICML-boomerang/] (code accompanying the paper "The Boomerang Sampler")
