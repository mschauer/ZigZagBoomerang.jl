# ZigZagBoomerang
<img src="https://raw.githubusercontent.com/mschauer/ZigZagBoomerang.jl/master/figures/fastforward.png" width="200">


[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mschauer.github.io/ZigZagBoomerang.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mschauer.github.io/ZigZagBoomerang.jl/dev/index.html)
[![Build Status](https://travis-ci.com/mschauer/ZigZagBoomerang.jl.svg?branch=master)](https://travis-ci.com/mschauer/ZigZagBoomerang.jl)
[![DOI](https://zenodo.org/badge/276593775.svg)](https://zenodo.org/badge/latestdoi/276593775)

## Overview

The package provides efficient and modular implementations
of samplers of several piecewise deterministic Markov processes (PDMP), the Bouncy Particle, the Boomerang, the ZigZag, and the factorised Boomerang.
The sampler requires the gradient of the potential of the target density as input and are called through `pdmp` producing a trajectory. `pdmp_inner!` gives access to the transition function of the sampler.

The non-factorised samplers (Bouncy Particle, the Boomerang)
take a function `∇ϕ!(y, x)` writing the gradient of the potential `ϕ(x) = -log(p(x))` of the target density `p` into `y`. The factorised samplers (factorised Boomerang and ZigZag) take `∇ϕ(x, i)`, the `i`'s partial derivative of `ϕ` (that is `y[i]`.)


*Subsampling:* One highlight is that these samplers allow
exact MCMC with subsets of data. This is because they just need an unbiased estimate of the gradient of the log densities to sample from a target.

The factorised samplers make use of a sparse Gaussian approximation of the target density (in form of a sparse precision matrix `Γ`).

 *Local factorised samplers*. ZigZag and the factorised Boomerang can optionally make use of the sparsity of the gradient of the potential of the target density, see `spdmp`.

 See [https://github.com/mschauer/ZigZagBoomerang.jl/tree/master/scripts/logistic.jl] for a worked out example (logistic regression with n=8840, p=442 and sparse Gramian) sampling from a non-Gaussian target with structured sparse gradient of the potential and unbiased estimate of the gradient subsampling of the n observations.

<img src="https://raw.githubusercontent.com/mschauer/ZigZagBoomerang.jl/master/figures/logisticZigZag.png" width="600">

## Gallery

<img src="https://raw.githubusercontent.com/mschauer/ZigZagBoomerang.jl/master/figures/zigzag.png" width="300"><img src="https://raw.githubusercontent.com/mschauer/ZigZagBoomerang.jl/master/figures/boomerang.png" width="300">

<img src="https://raw.githubusercontent.com/mschauer/ZigZagBoomerang.jl/master/figures/localzigzag.png" width="300"><img src="https://raw.githubusercontent.com/mschauer/ZigZagBoomerang.jl/master/figures/factboom.png" width="300">

<img src="https://raw.githubusercontent.com/mschauer/ZigZagBoomerang.jl/master/figures/localzigzag3d.png" width="300"><img src="https://raw.githubusercontent.com/mschauer/ZigZagBoomerang.jl/master/figures/factboom3d.png" width="300">

<img src="https://raw.githubusercontent.com/mschauer/ZigZagBoomerang.jl/master/figures/surf10.png" width="200"><img src="https://raw.githubusercontent.com/mschauer/ZigZagBoomerang.jl/master/figures/surf30.png" width="200"><img src="https://raw.githubusercontent.com/mschauer/ZigZagBoomerang.jl/master/figures/surf50.png" width="200">

Showing mixing and convergence from white noise to a Gaussian random field: ZigZag after 10, after 30, after 50 time units. See the script folder for the example and a video.

## Examples
See [https://github.com/mschauer/ZigZagBoomerang.jl/tree/master/scripts].


## Literature

* Joris Bierkens, Paul Fearnhead, Gareth Roberts: The Zig-Zag Process and Super-Efficient Sampling for Bayesian Analysis of Big Data. *The Annals of Statistics*, 2019, 47. Vol., Nr. 3, pp. 1288-1320. [https://arxiv.org/abs/1607.03188].
* Joris Bierkens, Sebastiano Grazzi, Kengo Kamatani and Gareth Robers: The Boomerang Sampler. *ICML 2020*. [https://arxiv.org/abs/2006.13777].
* Joris Bierkens, Sebastiano Grazzi, Frank van der Meulen, Moritz Schauer: A piecewise deterministic Monte Carlo method for diffusion bridges.  2020. [https://arxiv.org/abs/2001.05889].
* [https://github.com/jbierkens/ICML-boomerang/] (code accompanying the paper "The Boomerang Sampler")
