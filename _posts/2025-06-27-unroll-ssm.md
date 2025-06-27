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


## Naive implementation of an all-pole IIR filter

The difference equation of a \\(M\\)-th order all-pole IIR filter is given by:

$$
y[n] = x[n] -\sum_{m=1}^{M} a_m y[n-m].
$$

Let's implement this in PyTorch:

```python
import torch
from torch import Tensor

@torch.jit.script
def naive_allpole(x: Tensor, a: Tensor) -> Tensor:
    """
    Naive all-pole filter implementation.
    
    Args:
        x (Tensor): Input signal.
        a (Tensor): All-pole coefficients.
        
    Returns:
        Tensor: Filtered output signal.
    """
    assert x.dim() == 2, "Input signal must be a 2D tensor (batch_size, signal_length)"
    assert a.dim() == 1, "All-pole coefficients must be a 1D tensor"

    # list to store output at each time step
    output = []
    # assume initial condition is zero
    zi = x.new_zeros(x.size(0), a.size(0))

    for xt in x.unbind(1):
        # use addmv for efficient matrix-vector multiplication
        yt = torch.addmv(xt, zi, a, alpha=-1.0)
        output.append(yt)

        # update the state for the next time step
        zi = torch.cat([yt.unsqueeze(1), zi[:, :-1]], dim=1)

    return torch.stack(output, dim=1)
```

In this implementation I didn't use any in-place operations for speedup since it would break the differentiability of the function.
This naive implementation is not very efficient since `torch.addmv` and `torch.cat` are called at each time step, and usually the audio signal is hundreds of thousands of samples long, creating a lot of function call overhead.
For details please refer to my [tutorial on differentiable IIR filters](https://intro2ddsp.github.io/filters/iir_torch.html) at ISMIR 2023.

Notice that I used `torch.jit.script` to compile the function for some slight speedup.
I tried the newer compilation feature `torch.compile` but it didn't work.
The compilation just hangs forever I don't know why...
Personally I never found `torch.compile` to be useful in my research projects, and `torch.jit.*` has proven to be way more reliable.

Let's benchmark the speed of it on my Ubuntu with an Intel i7-7700K.
We'll use a batch size of 8, a signal length of 16384, and \\(M=2\\), which is a common setting for audio processing.

```python
from torch.utils.benchmark import Timer

batch_size = 8
signal_length = 16384
order = 2

def order2a(order: int) -> Tensor:
    a = torch.randn(order)
    # simple way to ensure stability
    a = a / a.abs().sum()
    return a

a = order2a(order)
x = torch.randn(batch_size, signal_length)

naive_allpole_t = Timer(
    stmt="naive_allpole(x, a)",
    globals={"naive_allpole": naive_allpole, "x": x, "a": a},
    label="naive_allpole",
    description="Naive All-Pole Filter",
    num_threads=4,
)
naive_allpole_t.blocked_autorange(min_run_time=1.0)
```
```shell
<torch.utils.benchmark.utils.common.Measurement object at 0x7d60c6a20770>
naive_allpole
Naive All-Pole Filter
  Median: 165.85 ms
  IQR:    4.73 ms (163.46 to 168.19)
  6 measurements, 1 runs per measurement, 4 threads
```

165.85 ms is quite slow, but it is expected.

## State space model formulation

Before we proceed to showing the sample unrolling trick, let's first introduce the state space model (SSM) formulation of the all-pole IIR filter.
The model is similar to the one used in the [TDF-II filter](https://iamycy.github.io/posts/2025/04/differentiable-tdf-ii/):

$$
\begin{align*}
\mathbf{h}[n] &= \begin{bmatrix}
    -a_1 & -a_2 & \cdots & -a_{M-1} & -a_M \\
    1 & 0 &\cdots & 0 & 0 \\
    0 & 1 & \cdots & 0 & 0 \\
    \vdots &  \vdots & \ddots & \vdots & \vdots \\
    0 & 0 & \cdots & 1 & 0 \\
\end{bmatrix} \mathbf{h}[n-1] + \begin{bmatrix}
    1 \\
    0 \\
    0 \\
    \vdots \\
    0 \\
\end{bmatrix} x[n] \\
&= \mathbf{A} \mathbf{h}[n-1] + \mathbf{B} x[n] \\

y[n] &= \mathbf{B}^\top \mathbf{h}[n].
\end{align*}
$$

Here I simplified the original SSM a bit by omitting the direct path since we can derive it from the state vector (for all-pole filter only).
Here's the PyTorch implementation of it:

```python
@torch.jit.script
def a2companion(a: Tensor) -> Tensor:
    """
    Convert all-pole coefficients to a companion matrix.

    Args:
        a (Tensor): All-pole coefficients.

    Returns:
        Tensor: Companion matrix.
    """
    assert a.dim() == 1, "All-pole coefficients must be a 1D tensor"
    order = a.size(0)
    c = torch.diag(a.new_ones(order - 1), -1)
    c[0, :] = -a
    return c

@torch.jit.script
def state_space_allpole(x: Tensor, a: Tensor) -> Tensor:
    """
    State-space implementation of all-pole filtering.

    Args:
        x (Tensor): Input signal.
        a (Tensor): All-pole coefficients.

    Returns:
        Tensor: Filtered output signal.
    """
    assert x.dim() == 2, "Input signal must be a 2D tensor (batch_size, signal_length)"
    assert a.dim() == 1, "All-pole coefficients must be a 1D tensor"

    c = a2companion(a).T

    output = []
    # assume initial condition is zero
    h = x.new_zeros(x.size(0), c.size(0))

    # B * x
    x = torch.cat(
        [x.unsqueeze(-1), x.new_zeros(x.size(0), x.size(1), c.size(0) - 1)], dim=2
    )

    for xt in x.unbind(1):
        h = torch.addmm(xt, h, c)
        # B^T @ h
        output.append(h[:, 0])
    return torch.stack(output, dim=1)
```

`a2companion` converts the all-pole coefficients to a [companion matrix](https://en.wikipedia.org/wiki/Companion_matrix), which is \\(\\mathbf{A}\\) in the SSM formulation.

Before we benchmark the speed of this implementation, let's predict how fast it will be.
Intuitively, since the complexity of vector-dot product is \\(O(M)\\) and matrix-vector multiplication is \\(O(M^2)\\), the SSM implementation use more computational resources so it should be slower than the naive implementation.
Let's benchmark the speed of it:

```python
state_space_allpole_t = Timer(
    stmt="state_space_allpole(x, a)",
    globals={"state_space_allpole": state_space_allpole, "x": x, "a": a},
    label="state_space_allpole",
    description="State-Space All-Pole Filter",
    num_threads=4,
)
state_space_allpole_t.blocked_autorange(min_run_time=1.0)
```
```shell
<torch.utils.benchmark.utils.common.Measurement object at 0x7d62d32a5160>
state_space_allpole
State-Space All-Pole Filter
  Median: 115.29 ms
  IQR:    1.47 ms (114.87 to 116.34)
  9 measurements, 1 runs per measurement, 4 threads
```
Interesting, the SSM implementation is actually faster by about 30 ms!