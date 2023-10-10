---
layout: post
title:  "Generalized funtions"
date:   2021-03-28 14:00:00 -0400
tags: math, algebra
categories: math
math: true
header-includes:
  - \usepackage{mathrsfs}
  - \usepackage{amsbsy}
---

# **Functional**

[Functional](https://mathworld.wolfram.com/Functional.html) refers to a mapping from a space $\mathit{X}$ (usually of functions) into $\boldsymbol{R}$ (real numbers). (Or complex numbers, in a more general case.) 

# **Generalized Function**

## Definition

Let  \\(\varphi(x)\\)  be an *argument of the function*, which is a function where $x \in \boldsymbol{R}$, and $\forall x$, $\varphi(x) \in \mathscr{D}$. 

 * $\varphi(x)$ is *smooth*, if it has derivatives of all orders.

 * $\varphi(x)$ is *compact*, if it has a bounded *support*. The *support* of a function $\varphi(x)$ is written as $\text{supp}(\varphi)$,  $\text{supp}(\varphi) := \\{x \in \mathit{X} \mid \varphi(x) \ne 0 \\}$. 

Let $f(x) := \boldsymbol{R} \rightarrow \boldsymbol{R}$ be the kernel of $\varphi(x)$.

We have the following [*linear functional*](https://mathworld.wolfram.com/LinearFunctional.html):

$$T[\varphi] = \int f(x) \varphi(x) dx$$

* $T[\cdot]$ is *continuous*, if for any sequence $\\{\varphi_k\\}_{k = 1, \cdots, \infty}$, when the sequence converges to $\varphi$ at $k \rightarrow \infty$, the corresponding sequence $\\{T[\varphi_k]\\}$ converges to $T[\varphi]$.

* $T[\cdot]$ maps inputs from $\mathscr{D}$ (function space) to $\boldsymbol{R}$.

 <span style="color: green"> Any *linear functional* $T[\varphi]$, which is *continuous* on the set $\mathscr{D}$ of
*smooth compact functions*, is called a *generalized function*. A *generalized function* is a *linear functional*, but not a *function*.  </span> 

## Linearity of Generalized Function

$\forall \varphi(x), \psi(x) \in \mathscr{D}$, $\forall a, b \in \boldsymbol{R}$, we have:

$$T[a\varphi + b\psi] = a T[\varphi] + b T[\psi]$$

## Regular Generalized Function

If the kernal $f(x)$ is an everywhere continuous, bounded function, $T[\varphi]$ is called a *regular generalized function* identified with the kernal $f(x)$.

## Singular Generalized Function

A *singular generlized function* $T[\varphi]$ is still linear continuous, but its kernel is not continuous. 

The most important example is the function:

$$T[\varphi] =  \varphi(0)$$

Though the kernel is not continuous, we still give it a symbol $\delta(x)$ and write it as:

$$T[\varphi] = \int \delta(x) \varphi(x) dx = \varphi(0)$$

Although the above thing is well known as the delta function, according to [Wikipedia](https://en.wikipedia.org/wiki/Dirac_delta_function) and [Wolfram](https://mathworld.wolfram.com/DeltaFunction.html), given different contexts, the delta function can refer to different things. As a distribution, the delta function is $T[\cdot]$, the linear functional, and is written as $\delta[\varphi] =  \varphi(0)$. As a measure, or in the engineering context, the kernel $\delta(x)$ in  $\int \delta(x) \varphi(x) dx$ is the delta function. A very confusing expression is:

$$\delta [\varphi] = \int \delta(x) \varphi(x) dx = \varphi(0)$$

If $\varphi(x) = 1$:

$$\delta [1] = \int \delta(x) dx = 1$$

We can also "shift" the delta function to map the function space to function value at arbitrary points: (again the first $\delta_a$ is a functional delta, the second $\delta$ in the integral is a function delta)

$$\delta_a [\varphi] = \int \delta(x - a) \varphi(x) dx = \varphi(a)$$

One approach to estimate the delta function is by the limit of a sequence of ordinary integrals, where the $k$ th element in the sequence looks like:

$$\delta_k [\varphi] = \int \delta_k(x) \varphi(x) dx$$

$\delta_k(x)$ can be ordinary functions. One choice is the Gaussian probability density function: $\delta_k(x) = \frac{1}{k\sqrt{2 \pi}} e^{-\frac{x^2}{2k^2}}$

# Derivatives of Generalized Function

We define the derivative of a generalized function by the derivative of its kernel functions:

$$T'[\varphi] = \int f'(x) \varphi(x) dx $$

According to the rule of integration by parts, we have:

$$\int f'(x) \varphi(x) dx + \int f(x) \varphi'(x) dx = f(x)\varphi(x) |_{-\infty}^{\infty}$$

Recall that $\varphi(x)$ is compact (has a bounded support), so $\lim_{x \rightarrow \infty}\varphi(x) = 0$, $\lim_{x \rightarrow -\infty}\varphi(x) = 0$, therefore we have:

$$\int f'(x) \varphi(x) dx + \int f(x) \varphi'(x) dx = 0$$

Which means:

$$T'[\varphi] + T[\varphi'] = 0$$

So We can do derivative on $\varphi$ to get the derivative of a generalized function.

With the same logic, we have

$$T^{(n)}[\varphi] = (-1)^nT[\varphi^{(n)}(x)]$$

The Leibniz formula also works on the product of a generalized function and a function. First we have the product of a function and a generalized function defined as:

$$g(x)T[\varphi] = T[g\varphi]$$

Then the Leibniz formula is:

$$(g(x)T[\varphi])' = g'(x)T[\varphi] + g(x)T'[\varphi]$$

$$(g(x)T[\varphi])^{(n)} = \sum^{n}_{m = 0}g^{(m)}(x)T^{(n-m)}[\varphi]$$

This is more like a trick, since the symbol $g'(x)T[\varphi]$ is not an actual product, you just switch between different expressions to make it look like the Leibniz formula :

$$
\begin{align*}
(g(x)T[\varphi])' &= T'[g\varphi]
\\ &= T[(g\varphi)']
\\ &= T[g'\varphi + g\varphi']
\\ &= T[g'\varphi] + T[g\varphi']  \text{ (linearity)}
\\ &= g'(x)T[\varphi] + g(x)T'[\varphi]
\end{align*}
$$

#  Generalized Function of a Composite Argument

First we have the following equality:

$$
\int f(g(x))\varphi(x) dx = \int f(y)\varphi(g^{-1}(y)) g^{-1 \prime} (y) dy
$$

Where $g^{-1}(x)$ is the inverse of $g(x)$, i.e. $g(g^{-1}(x)) = x$. The above equality can be easily proved by replace $x$ on the left side with $g^{-1}(y)$. 

We name $\int f(g(x))\varphi(x) dx$ as a generalized function $T$ of a *composite argument* $g(x)$.



