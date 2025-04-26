---
title: 'Notes on differentiable TDF-II filter'
date: 2025-04-26
permalink: /posts/2025/04/differentiable-tdf-ii/
tags:
  - differentiable IIR
  - scientific computing
  - pytorch
---

This blog is a continuation of some of my early calculations for propagating gradients through general IIR filters, including direct-form and transposed-direct-form.

## Back story

In the early 2021, I made an implementation of a differentiable `lfilter` function for `torchaudio` (a few core details were published two years later [here](/publications/2023-11-4-golf)). 
The basic idea is implementing the backpropagation of gradients in C++ for optimal performance.
The implementation was based on Direct-Form-I (DF-I).
This is different from the popular implementation of SciPy's `lfilter`, which is based on Transposed-Direct-Form-II (TDF-II) and more numerically stable[^1].

It'll be better to implement it in this form, but... at the time, my knowledge base was not enough to generalise the idea to TDF-II.
In DF-I/II, the gradients of FIR and all-pole filters can be treated independently so I was only working on the recursive part of the filter (the all-pole).

<div style="display: flex; justify-content: space-between;">
  <div style="width: 55%;">
    <img src="https://ccrma.stanford.edu/~jos/filters/img1127_2x.png" alt="TDF-II" style="width: 100%; height: auto;"/>
  </div>
  <div style="width: 45%;">
    <img src="https://ccrma.stanford.edu/~jos/filters/img1144_2x.png" alt="DF-I" style="width: 100%; height: auto;"/>
  </div>
</div>

However, in TDF-II, the two parts are combined and the registers are shared, so my previous approach does not work.
I left this as a TODO for the future[^2].

<img src="https://ccrma.stanford.edu/~jos/filters/img1147_2x.png" alt="DF-I" style="width: 50%; height: auto;"/>

Many things have changed since then.
I started my PhD in 2022 and have more time to think thoroughly about the problem.
My understanding of filters also improved a bit after exploring the filter idea a few times with some publications.
It's time to revisit the problem, a differentiable TDF-II filter.

**TL;DR**, *the backpropagation of TDF-II filter is a DF-II filter, and vice versa.*

The following calucation consider the a general case when the filters parameters are **time-varying**.
Time-invariant systems are a special case of this and is trivial once we have the time-varying results.


## (Transposed-)Direct-Form II
Given time-varying coefficients \\(\\{b_0[n], b_1[n],\dots,b_M[n]\\}\\) and \\(\\{a_1[n],\dots,a_N[n]\\}\\), the TDF-II filter can be expressed as:

$$
y[n] = s_1[n] + b_0[n] x[n]
$$

$$
s_1[n+1] = s_2[n] + b_1[n] x[n] - a_1[n] y[n]\\
$$

$$
s_2[n+1] = s_3[n] + b_2[n] x[n] - a_2[n] y[n]
$$

$$
\vdots
$$

$$
s_M[n+1] = b_M[n] x[n] - a_M[n] y[n].
$$

We can also write it in observable canonical form:

$$
\mathbf{s}[n+1]
=
\mathbf{A}[n] \mathbf{s}[n] + \mathbf{B}[n] x[n]
$$

$$
y[n] = \mathbf{C}\mathbf{s}[n] + b_0[n] x[n]
$$

$$
\mathbf{A}[n] =
\begin{bmatrix}
  -a_1[n] & 1 & 0 & \cdots & 0 \\
  -a_2[n] & 0 & 1 & \cdots & 0 \\
  \vdots & \vdots & \vdots & \ddots & \vdots \\
  -a_M[n] & 0 & 0 & \cdots & 1
\end{bmatrix}
$$

$$
\mathbf{C} =
\begin{bmatrix}
  1 & 0 & 0 & \cdots & 0 \\
\end{bmatrix}.
$$

The values of \\(\\mathbf{B}[n] \\) can be referred from Julius' blog[^4].

Regarding DF-II, its difference equations are:

$$
v[n] = x[n] - \sum_{i=1}^{M} a_i[n] s[n-i]
$$

$$
y[n] = \sum_{i=0}^{M} b_i[n] v[n-i].
$$

Similarly, it can be expressed as a state-space model using the controller canonical form:

$$
\mathbf{v}[n+1]
=
\begin{bmatrix}
  -a_1[n] & -a_2[n] & \cdots & -a_M[n] \\
  1 & 0 & \cdots & 0 \\
  \vdots & \vdots & \ddots & \vdots \\
  0 & 0 & \cdots & 1
\end{bmatrix}
\mathbf{v}[n] +
\begin{bmatrix}
  1 \\
  0 \\
  \vdots \\
  0
\end{bmatrix}
x[n]
= \mathbf{A}^\top[n]\mathbf{v}[n] + \mathbf{C}^\top x[n]
$$

$$
y[n] = \mathbf{B}^\top[n] \mathbf{v}[n] + b_0[n] x[n].
$$

As I have shown above, the forms are very similar.
The transition matrix of TDF-II is the transpose of the DF-II and the column vectors **B** and **C** are swapped.
(This is the reason why we call it transposed-DF-II.)
Note that the resulting transfer function is not the same due to the difference in computation order in time-varying case.
(They are the same if the coefficients are time-invariant!)
I will use the state-space form for simplicity in the following sections.

## Backpropagation through TDF-II

Supposed we have evaluated some loss function \\(\mathcal{L}\\) on the output of the filter \\(y[n]\\) and has the instantaneous gradients \\(\\frac{\\partial \mathcal{L}}{\\partial \mathbf{s}[n]}\\).
We want to backpropagate the gradients through the filter to get the gradients of the input \\\(\\frac{\\partial \mathcal{L}}{\\partial x[n]}\\) and the filter coefficients \\(\\frac{\\partial \mathcal{L}}{\\partial a_i[n]}\\) and \\(\\frac{\\partial \mathcal{L}}{\\partial b_i[n]}\\).
Let's first denote \\(\mathbf{z}[n] = \mathbf{B}[n] x[n]\\) since once we get the gradients of \\(\\mathbf{z}[n]\\), it's easy to get the gradients of the two using the chain rule.
<!-- Also we'll assume the length of the signal is bounded in the range \\([1, N]\\). -->
The recursion in TDF-II state-space form becomes:

$$
\mathbf{s}[n+1] = \mathbf{A}[n] \mathbf{s}[n] + \mathbf{z}[n].
$$

If we unroll the recursion so there's no **s** in the right-hand side, we get:

$$
\mathbf{s}[n+1] = \sum_{i=1}^{\infty} \left(\prod_{j=1}^{i} \mathbf{A}[n-j+1]\right) \mathbf{z}[n-i] + \mathbf{z}[n].
$$ 

The gradients with respect to **z** can be computed as:

$$
\frac{\partial \mathbf{s}[n]}{\partial \mathbf{z}[i]} 
= 
\begin{cases}
  \prod_{j=1}^{n-i-1} \mathbf{A}[n-j] & i < n - 1 \\
  \mathbf{I} & i = n -1 \\
  0 & i \geq n
\end{cases}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}[n]}
= \sum_{i=n+1}^{\infty} \frac{\partial \mathcal{L}}{\partial \mathbf{s}[i]} \frac{\partial \mathbf{s}[i]}{\partial \mathbf{z}[n]}
$$

$$
= \frac{\partial \mathcal{L}}{\partial \mathbf{s}[n+1]} + \sum_{i=n+2}^{\infty} \frac{\partial \mathcal{L}}{\partial \mathbf{s}[i]} \prod_{j=1}^{i-n-1} \mathbf{A}[i-j] 
$$

$$
= \frac{\partial \mathcal{L}}{\partial \mathbf{s}[n+1]} + \sum_{i=n+2}^{\infty} \left( \prod_{j=i-n-1}^{1} \mathbf{A}^\top[i-j] \right) \frac{\partial \mathcal{L}}{\partial \mathbf{s}[i]}
$$

$$
= \frac{\partial \mathcal{L}}{\partial \mathbf{s}[n+1]} + \sum_{i=n+2}^{\infty} \left( \prod_{j=1}^{i-n-1} \mathbf{A}^\top[n+j] \right) \frac{\partial \mathcal{L}}{\partial \mathbf{s}[i]}.
$$

$$
= \mathbf{A}^\top[n+1] \frac{\partial \mathcal{L}}{\partial \mathbf{z}[n+1]} + \frac{\partial \mathcal{L}}{\partial \mathbf{s}[n+1]}.
$$

I omitted the transpose sign for the vector for simplicity.
The last recursion involves \\(\mathbf{A}^\top\\), implies that, to backpropagate the gradients through the recursion of TDF-II, we need to use the **recursion of DF-II but in the opposite direction**!
Their roles will be swapped if we compute the gradients of DF-II using the same procedure, but I'll leave it as an exercise for the reader :D

For completeness, the following are the procedure to compute the gradients of the input and filter coefficients.

### Gradients of the input

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{s}[n]} 
= \mathbf{C}^\top[n] \frac{\partial \mathcal{L}}{\partial y[n]}
% \begin{bmatrix}
%   \frac{\partial \mathcal{L}}{\partial y[n]} \\
%   0 \\
%   \vdots \\
%   0
% \end{bmatrix}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}[n]}
= \mathbf{A}^\top[n+1] \frac{\partial \mathcal{L}}{\partial \mathbf{z}[n+1]} + \mathbf{C}^\top[n] \frac{\partial \mathcal{L}}{\partial y[n]}
$$

(Note that the above line is the same as the one in DF-II! Just the input and output variables are changed.)

$$
\frac{\partial \mathcal{L}}{\partial x[n]}
= \frac{\partial \mathcal{L}}{\partial \mathbf{z}^\top}[n] \mathbf{B}[n]
+ \frac{\partial \mathcal{L}}{\partial y[n]} b_0[n]
$$


### Gradients of the **b** coefficients

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{B}[n]}
= \frac{\partial \mathcal{L}}{\partial \mathbf{z}[n]} x[n]
$$

$$
\frac{\partial \mathcal{L}}{\partial b_0[n]}
= \frac{\partial \mathcal{L}}{\partial y[n]} x[n]
$$

### Gradients of the **a** coefficients

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{A}[n]}
= \frac{\partial \mathcal{L}}{\partial \mathbf{z}[n]} \mathbf{s}^\top[n]
\to a_i[n] = -\frac{\partial \mathcal{L}}{\partial z_i[n]} s_1[n]
$$

### Time-invariant case
In the time-invariant case, the parameters are constant.

$$
a_i[n] = a_i[m] \quad \forall n, m, \quad i = 1, \dots, M
$$

$$
b_i[n] = b_i[m] \quad \forall n, m, \quad i = 0, \dots, M
$$

In this case, we can just simply sum the gradients over time:

$$
\frac{\partial \mathcal{L}}{\partial a_i} = \sum_{n} \frac{\partial \mathcal{L}}{\partial a_i[n]},~\ \frac{\partial \mathcal{L}}{\partial b_i} = \sum_{n} \frac{\partial \mathcal{L}}{\partial b_i[n]}.
$$


## Summary

The above findings suggest a way to compute the gradients of TDF-II filter efficiently.
To do this, the following steps are needed:

1. Implement the recursions of TDF-II and DF-II filters in C++/CUDA/Metal/etc.
2. After doing the forward pass of TDF-II, store \\(s_1[n]\\), \\(\\mathbf{a}[n]\\), \\(\\mathbf{b}[n]\\), and \\(x[n]\\).
3. When doing backpropagation, filter the output gradients \\(\frac{\partial \mathcal{L}}{\partial y[n]}\\) through the DF-II filter's recusions in the opposite direction using the same **a** coefficients.
4. Compute the gradients of the input and filter coefficients using the equations above. Note that although \\(\frac{\partial \mathcal{L}}{\partial \mathbf{z}[n]}\\) is a sequence of vectors, since the higher-order states in DF-II are just time-delayed versions of the first state (\\(v_M[n] = v_{M-1}[n-1] = \cdots = v_1[n-M+1]\\)), we can just store \\(\frac{\partial \mathcal{L}}{\partial z_1[n]}\\) for gradient computation, reducing the memory usage by a factor of \\(M\\).

### Final thoughts
The procedure above can be applied to derive the gradients DF-II filter as well.
The resulting algorithm is identical but the roles of TDF-II and DF-II are swapped.
Personally I found using state-space formulation is much easier, straightforward, and elegant than the [derivation I did in 2024](/publications/2024-9-3-diffapf) to calculate the gradients of time-varying all-pole filters, which is basically the same problem.
(Man, I was basically bruteforcing it...)
Applying the method to TDF-I is straightfowrad, just set \\(\\mathbf{B}[n] = 0\\).

Interestingly, since the backpropagation of TDF-II is a DF-II filter, which means it's less numerically stable than TDF-II; in contrast, the backpropagation of DF-II is a TDF-II filter and is more stable.
We'll always have this trade-off, so is it really necessary to have TDF-II if we want differentiability?
Probably yes, since besides backpropagation, the gradients can also be computed using **forward-mode** automatic differentiation, which basically computes the Jacobian from an opposite direction.
In this way, the forwarded gradients are computed in the same way as the filter's forward pass, and the math is much easier to show than the backpropagation I wrote above. (Should realise earlier...)
Also, in the time-varying case and \\(M > 1\\), none of the two forms guarantee BIBO stability.
This is a another interesting topic but let's just leave it for now.
I hope this post is useful for those who are interested in differentiable IIR filters.

## Notes

The figures are from [Julius O. Smith III](https://ccrma.stanford.edu/~jos/filters/Implementation_Structures_Recursive_Digital.html) and the notations are adapted from his blog[^4].
The algorithm is based on the following papers:

1. Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables (doi: 10.5281/zenodo.13916489)
2. Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis (doi: 10.21437/Interspeech.2024-1187)
3. Differentiable All-pole Filters for Time-varying Audio Systems
4. GOLF: A Singing Voice Synthesiser with Glottal Flow Wavetables and LPC Filters (doi: 10.5334/tismir.210)


***
**References:**

[^1]: https://ccrma.stanford.edu/~jos/filters/Numerical_Robustness_TDF_II.html
[^2]: https://github.com/pytorch/audio/pull/1310#issuecomment-790408467
[^3]: https://transactions.ismir.net/articles/10.5334/tismir.210
[^4]: https://ccrma.stanford.edu/~jos/fp/Converting_State_Space_Form_Hand.html