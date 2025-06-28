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
\begin{align}
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
\end{align}
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

By using `torch.profiler.profile`, I found that, in the naive implementation, `torch.cat` for updating the last M outputs takes a significant amount of the total time (~20%).
The actual computation, `torch.addmv`, takes only about 10%.
Regarding the memory usage, the most memory-consuming operation is `torch.addmv`, which uses about 512 Kb of memory.
In contrast, the SSM implementation uses more memory (> 1 Mb) due to the matrix multiplication, but roughly 38% of the time is spent on filtering since it doesn't have to call `torch.cat` at each time step anymore.
The state vector (a.k.a the last M outputs) automatically get updated during the matrix multiplication.

**Conclusion**: tensor concatenation (including `torch.cat` and `torch.stack`) is expensive, and we should avoid it if possible.


## Unrolling the SSM

Now we can apply the unrolling trick to the SSM implementation.
The idea is to divide the input signal into blocks of size \\(T\\) and perform the recursion on the blocks instead of sample-by-sample.
Each recursion takes the last vector state \\(\\mathbf{h}[n-1]\\) and predicts the next \\(T\\) states \\([\\mathbf{h}[n], \mathbf{h}[n+1], \ldots, \mathbf{h}[n+T-1]]^\top\\) at once.
To see how to calculate these states, let's unroll the SSM recursion for \\(T\\) steps:

$$
\begin{align}
\mathbf{h}[n] &= \mathbf{A} \mathbf{h}[n-1] + \mathbf{B} x[n] \\
\mathbf{h}[n+1] &= \mathbf{A} \mathbf{h}[n] + \mathbf{B} x[n+1] \\
&= \mathbf{A} (\mathbf{A} \mathbf{h}[n-1] + \mathbf{B} x[n]) + \mathbf{B} x[n+1] \\
&= \mathbf{A}^2 \mathbf{h}[n-1] + \mathbf{A} \mathbf{B} x[n] + \mathbf{B} x[n+1] \\
\mathbf{h}[n+2] &= \mathbf{A} \mathbf{h}[n+1] + \mathbf{B} x[n+2] \\
&= \mathbf{A} (\mathbf{A}^2 \mathbf{h}[n-1] + \mathbf{A} \mathbf{B} x[n] + \mathbf{B} x[n+1]) + \mathbf{B} x[n+2] \\
&= \mathbf{A}^3 \mathbf{h}[n-1] + \mathbf{A}^2 \mathbf{B} x[n] + \mathbf{A} \mathbf{B} x[n+1] + \mathbf{B} x[n] \\
& \vdots \\
\mathbf{h}[n+T-1] &= \mathbf{A}^{T} \mathbf{h}[n-1] + \sum_{t=0}^{T-1} \mathbf{A}^t \mathbf{B} x[n+t] \\
\end{align}
$$

We can rewrite the above equation in matrix form as follows:

$$
\begin{align}
\begin{bmatrix}
    \mathbf{h}[n] \\
    \mathbf{h}[n+1] \\
    \vdots \\
    \mathbf{h}[n+T-1]
\end{bmatrix} &= \begin{bmatrix}
    \mathbf{A} \\
    \mathbf{A}^2 \\
    \vdots \\
    \mathbf{A}^T \\
\end{bmatrix} \mathbf{h}[n-1]
+ \begin{bmatrix}
    \mathbf{I} & 0 & \cdots & 0 \\
    \mathbf{A} & \mathbf{I} & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    \mathbf{A}^{T-1} & \mathbf{A}^{T-2} & \cdots & \mathbf{I}
\end{bmatrix}
\begin{bmatrix}
    \mathbf{B}x[n] \\
    \mathbf{B}x[n+1] \\
    \vdots \\
    \mathbf{B}x[n+T-1]
\end{bmatrix} \\
& = \begin{bmatrix}
    \mathbf{A} \\
    \mathbf{A}^2 \\
    \vdots \\
    \mathbf{A}^T \\
\end{bmatrix} \mathbf{h}[n-1]
+ \begin{bmatrix}
    \mathbf{I}_{.1} & 0 & \cdots & 0 \\
    \mathbf{A}_{.1} & \mathbf{I}_{.1} & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    \mathbf{A}_{.1}^{T-1} & \mathbf{A}_{.1}^{T-2} & \cdots & \mathbf{I}_{.1}
\end{bmatrix}
\begin{bmatrix}
    x[n] \\
    x[n+1] \\
    \vdots \\
    x[n+T-1]
\end{bmatrix} \\
& = \mathbf{M} \mathbf{h}[n-1] + \mathbf{V} \begin{bmatrix}
    x[n] \\
    x[n+1] \\
    \vdots \\
    x[n+T-1]
\end{bmatrix} \\
\end{align}
$$

Notice that in the second line, I utilised the fact the \\(\mathbf{B}\\) has only one non-zero entry to simplify the matrix.
(This is not possible if the filter is not strictly all-pole.)
\\(\mathbf{I}_{.1}\\) denotes the first column of the identity matrix and so on.

Now, the number of autoregressive steps is reduced from \\(T\\) to \\(\lceil \frac{T}{M}\\rceil\\) and the matrix multiplication is done in parallel for every \\(T\\) samples.
There are added costs for pre-computing the transition matrix \\(\mathbf{M}\\) and the input matrix \\(\mathbf{V}\\), though.
However, as long as the extra cost is relatively small compared to the cost of \\(T\\) autoregressive steps, we should see a speedup.

Here's the PyTorch implementation of the unrolled SSM:

```python
@torch.jit.script
def state_space_allpole_unrolled(
    x: Tensor, a: Tensor, unroll_factor: int = 1
) -> Tensor:
    """
    Unrolled state-space implementation of all-pole filtering.

    Args:
        x (Tensor): Input signal.
        a (Tensor): All-pole coefficients.
        unroll_factor (int): Factor by which to unroll the loop.

    Returns:
        Tensor: Filtered output signal.
    """
    if unroll_factor == 1:
        return state_space_allpole(x, a)
    elif unroll_factor < 1:
        raise ValueError("Unroll factor must be >= 1")

    assert x.dim() == 2, "Input signal must be a 2D tensor (batch_size, signal_length)"
    assert a.dim() == 1, "All-pole coefficients must be a 1D tensor"
    assert (
        x.size(1) % unroll_factor == 0
    ), "Signal length must be divisible by unroll factor"

    c = a2companion(a)

    # create an initial identity matrix
    initial = torch.eye(c.size(0), device=c.device, dtype=c.dtype)
    c_list = [initial]
    # TODO: use parallel scan to improve speed
    for _ in range(unroll_factor):
        c_list.append(c_list[-1] @ c)

    # c_list = [I c c^2 ... c^unroll_factor]
    M = torch.cat(c_list[1:], dim=0).T
    flatten_c_list = torch.cat(
        [c.new_zeros(c.size(0) * (unroll_factor - 1))]
        + [xx[:, 0] for xx in c_list[:-1]],
        dim=0,
    )
    V = flatten_c_list.unfold(0, c.size(0) * unroll_factor, c.size(0)).flip(0)

    # divide the input signal into blocks of size unroll_factor
    unrolled_x = x.unflatten(1, (-1, unroll_factor)) @ V

    output = []
    # assume initial condition is zero
    h = x.new_zeros(x.size(0), c.size(0))
    for xt in unrolled_x.unbind(1):
        h = torch.addmm(xt, h, M)
        # B^T @ h
        output.append(h[:, :: c.size(0)])
        h = h[
            :, -c.size(0) :
        ]  # take the last state vector as the initial condition for the next step
    return torch.cat(output, dim=1)
```

The `unroll_factor` parameter controls how many samples to process in parallel.
If it is set to 1, the function behaves like the original SSM implementation.

Now let's benchmark the speed of the unrolled SSM implementation.
We'll use `unroll_factor=128` since I already tested that it is the optimal value :)

```python
state_space_allpole_unrolled_t = Timer(
    stmt="state_space_allpole_unrolled(x, a, unroll_factor=unroll_factor)",
    globals={
        "state_space_allpole_unrolled": state_space_allpole_unrolled,
        "x": x,
        "a": a,
        "unroll_factor": 128,
    },
    label="state_space_allpole_unrolled",
    description="State-Space All-Pole Filter Unrolled",
    num_threads=4,
)
state_space_allpole_unrolled_t.blocked_autorange(min_run_time=1.0)
```
```shell
<torch.utils.benchmark.utils.common.Measurement object at 0x71c957fecd40>
state_space_allpole_unrolled
State-Space All-Pole Filter Unrolled
  1.89 ms
  1 measurement, 1000 runs , 4 threads
```
1.89 ms! What sorcery is this? That's a whopping 60x speedup compared to the standard SSM implementation!

A closer look at the profiling results shows in total 25% of the time is spent on matrix multiplication and addition.
Interestingly, around 30% of the time is spent on the `torch.flip` operation for preparing the input matrix \\(\mathbf{V}\\).
The speedup does comes with a cost of more memory usage, requiring more than 2 Mb for filtering.
Honestly, not a significant cost for modern Hardwares.

Notice that for convenience I ran the above benchmarks using CPU, which has very limited parallelism compared to GPU.
Thus, the huge speedup we see tells us that function call overhead is the major bottleneck for running recursions.


## More comparison

Since \\(T\\) is an important parameter for the unrolled SSM, I did some more benchmarks to see how it affects the speed.

### Varying sequence length

In this benchmark, I fixed the batch size to 8 and the order to 2, and varied the sequence length from 4096 to 262144.
The results suggest that the best unroll factor increase as the sequence length increases, and it's very likely to be \\(\\sqrt{N}\\).
Also, the longer the sequence length, the more speedup we get from the unrolled SSM.

![](/images/unroll-ssm/benchmark_seq_len.png)

### Varying filter order

To see how the filter order affects the speed, I fixed the batch size to 8 and the sequence length to 16384, and varied the filter order from 2 to 16.
It looks like my hypothesis that the best factor is \\(\\sqrt{M}\\) still holds, but the peak gradually shifts to the left as the order increases.
Moreover, the speedup is less significant for higher orders, which is expected as the \\(\mathbf{V}\\) matrix becomes larger.

![](/images/unroll-ssm/benchmark_order.png)

### Varying batch size

The speedup is less as the batch size increases, which is expected.
However, the peak of the best unroll factor also leans towards the left a bit when the batch size increases.

![](/images/unroll-ssm/benchmark_batch.png)

### Memory usage

To see how the memory usage changes with the unroll factor, I ran the unrolled SSM on a 5060 ti so I can use `torch.cuda.max_memory_allocated()` to measure the memory usage.
When batch size is 1 and \\(T > 1\\), the memory usage increases as the unroll factor increases, which is expected.

![](/images/unroll-ssm/mem_batch_1.png)

However, when using a larger batch size (32 in this case), the memory usage saturates and there's barely any difference between different unroll factors.

![](/images/unroll-ssm/mem_batch_32.png)


## Discussion

So far we have seen that the unrolled SSM can achieve a significant speedup for IIR filtering in PyTorch.
However, how to automatically determine the best unrolling factor is still not clear.
From the benchmarks I did on an i7 CPU, it seems that the optimal \\(T^*\\) is \\(\sqrt{N}\alpha\\) and \\(0 < \alpha \leq 1\\) is given by a polynomial function of the filter order and batch size.
However, this may not hold for other hardware.

One thing I didn't mention is about numerical accuracy.
If \\(|\mathbf{A}|\\) is very small, the precomputed exponentials \\(\mathbf{A}^T \to \mathbf{0}\\) which may not be accurately represented in floating point, especially in deep learning applications we use single precision a lot.
This is less of a problem for the standard SSM since at each time step the input mixed with the state vector, thus could help cancel out the numerical errors.

The idea should apply when there are zeros in the filter.
\\(\\mathbf{B}\\) will not be a simple one-hot vector anymore so \\(\mathbf{V}\\) has to be a full \\(MT\\times MT\\) square matrix.
Time-varying filters will benefit less from the unrolling trick since \\(\mathbf{V}\\) will also be time-varying, and computing \\(\frac{N}{T}\\) such matrices in advance increase the cost a lot.


## Conclusion & Thoughts

In this post I show that the unrolling trick can significantly speed up differentiable IIR filtering in PyTorch.
Due to the backend that computes matrix multiplication, the extra memory cost is less for larger batch sizes.
Although the filter I tested is a simple all-pole filter, it's trivial to extend the idea to general IIR filters.

This idea might help addresing one of the issues for future TorchAudio, after the Meta developers [announced](https://github.com/pytorch/audio/issues/3902) their future plan for it.
In the next major release, all the specialised kernels written in C++, including the `lfilter` I contirbuted years ago, will be removed from TorchAudio.
The filter I presented here is purely written in Python and as long as we have a clever way to determine the best unrolling factor depends on the filter parameters, it should be a easy drop-in replacement for the current `lfilter` implementation.