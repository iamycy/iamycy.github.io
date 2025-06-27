---
title: 'Block-based Fast Differentiable IIR in PyTorch'
date: 2025-06-27
permalink: /posts/2025/06/27/unroll-ssm/
tags:
  - differentiable IIR
  - scientific computing
  - pytorch
  - state space model
---

I recently came across a presentation by Andres Ezequiel Viso from GPU Audio at ADC 2022, which he talked about how they accelerate IIR filters on the GPU.
The approach they make use of is instead of running sample-by-sample, they formulate the IIR filter as a state space model (SSM) and augment the transition matrix so each step processes multiple samples at once.
The main speedup comes from the fact that GPU is really good at performing large matrix multiplications, and the SSM formulation allows us to take advantage of that.

<iframe width="1024px" height="576px"
src="https://www.youtube.com/embed/UmYnoFo0Bb8?start=1356"
allowfullscreen>
</iframe><br>

Speed up IIR filters while maintaining differentiability has always been my interest.
The most recent method I worked on is from my recent [submission](https://arxiv.org/abs/2504.14735) to DAFx 25, where my co-author Ben proposed using parallel associative scan to speed up the recursion on the GPU.
Nevertheless, since PyTorch does not have a built-in associative scan operator (compared to JAX), we have to implement custom kernels for it, which is non-trivial.
It also requires that the filter has distinct poles so the state space transition matrix is diagonalisable.
The method that GPU Audio presented seems to be doable solely using PyTorch Python API and doesn't have the restrctions I mentioned, thus I decided to benchmark it and see how it performs.

Since it's just a proof of concept, the filter I'm going to test is a **time-invariant all-pole IIR filter**, which is the minimal case of a recursive filter.
This does let us to leverage some special optimisations that won't work on time-varying general IIR filters, but that won't affect the approach I'm going to present here.