---
layout: post
title:  Paper Reviewing - Self-Supervised Learning with Kernel Dependence Maximization
date:   2023-11-24 16:28:00
tags: math, metric, deep learning
categories: math
math: true
header-includes:
  - \usepackage{mathrsfs}
  - \usepackage{amsbsy}
---

https://proceedings.neurips.cc/paper_files/paper/2021/file/83004190b1793d7aa15f8d0d49a13eba-Paper.pdf

# Metric Space

A metric space is a set with a notion of distance defined between every pair of elements in the set, the distance is called metric or distance function. A complete metric space is a metric space where a Cauchy sequence of any point in the metric space is also in the same metric space. Examples: on the real number set interval (0, 1) with absolute difference metric,  a sequence $x_n = \frac{1}{n}$, the sequence is Cauchy, but it converges to 0 which is outside of the interval, so the metric space is not complete.

## Hilbert Space

A Hilbert space is a [vector space](https://en.wikipedia.org/wiki/Vector_space "Vector space") equipped with an [inner product](https://en.wikipedia.org/wiki/Inner_product "Inner product") that induces a [distance function](https://en.wikipedia.org/wiki/Distance_function) for which the space is a [complete metric space](https://en.wikipedia.org/wiki/Complete_metric_space "Complete metric space").  

Hilbert space adds an additional constraint on how the distance function is defined on a complete metric space. For example, on $\mathbb{R}^2$ with inner product $(x_1, x_2) \cdot (y_1, y_2) = x_1y_1 + x_2y_2$ , a norm $||x||$ is defined by $||x|| = \sqrt{x \cdot x}$, and the distance between two points is defined b $||x - y||$, which is just the common Euclidean distance on  $\mathbb{R}^2$, but constructed from perspective of inner product. 


## Reproducing Hilbert Kernel Space

https://www.youtube.com/watch?v=EoM_DF3VAO8
https://www.youtube.com/watch?v=ABOEE3ThPGQ

### Kernel

The idea of kernel is to connecting between linear space and high dimensional space. Suppose we have a random space $\mathcal{X}$ , and we have a function $\phi: \mathcal{X} \rightarrow \mathcal{H}$ that maps a point in $\mathcal{X}$ to a point in another space $\mathcal{H}$. The new space $\mathcal{H}$ is a Hilbert space which guarantees some good properties, and a typical example of a Hilbert space is $\mathbb{R}^d$ (d-dimensional Euclidean space). In machine learning, the intuition of seeking such a mapping function $\phi$ is that the classification algorithms such as SVM are easier to optimized on $\mathbb{R}^d$, which means with $\phi$ we can run the algorithms as efficiently as on $\mathbb{R}^d$  with data samples collected from more complex spaces. 

Because $\phi(x) \in \mathcal{H}$, it is natural that we look at the inner product between $\phi(x)$ and $\phi(y)$, i.e. $\phi(x) \cdot \phi(y)$. We hope there is a direct way to get this inner product scalar value from $\mathcal{X}$ instead of computing $\phi$, so we propose a new function $k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$, such that $k(x, y) = \phi(x) \cdot \phi(y)$. This approach to compute inner product directly is called kernel trick, and $k$ is called a kernel function. 

A kernel function always satisfies the following properties:

* Symmetry. $k(x, y) = k(y, x)$. Since the inner product on $\mathcal{H}$ is symmetric.
* Has a corresponding positive semi-definite kernel metric. We construct a kernel metric $K$ on any $n$ data points $x_1, x_2, \cdots, x_n \in \mathcal{X}$  by setting $K_{i, j} = k(x_i, x_j)$. $K$ is positive semi-definite, which means that $\forall c \in \mathbb{R}^n$, $c^TKc \ge 0$ 

### Constructing Reproducing Kernel Hilbert Space

Reproducing Kernel Hilbert Space answers the following question : given a set $\mathcal{X}$ and a kernel function $K: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$, how do we construct a Hilbert space $\mathcal{H}$ and the mapping function $\phi$ ? Basically this is an inverse version of how a kernel function is constructed, i.e. how we find the two component (a Hilbert space and a mapping function) that can give us the known $k$ .

We start by define the following function 

$$k_x(\cdot) = k(x, \cdot)$$

Using $k_x(\cdot)$, we map $x \in \mathcal{X}$ to an image of a function $k_x$.  We then construct a spamming set $\mathcal{G}$ from $\{k_x | x \in \mathcal{X} \}$ by taking all $k_x$ as elements and compute all of the finite linear combination of them:

$$\mathcal{G} = \{\sum_{i=1}^r m_i k_{x_i} | m_i \in \mathbb{R},  r \in \mathbb{N} \}$$

$\mathcal{G}$ is quite close to the Hilbert space we are looking for, but we need to do two things:

* Define the inner product. The way we define the inner product is 

$$k_{x_i} \cdot k_{x_j} = k(x_i, x_j)$$ for $x_i \in \mathcal{X}$ and $x_i \in \mathcal{X}$

$$\sum_{i=1}^r \alpha_i k_{x_i} \cdot \sum_{j=1}^l \beta_j k_{y_j} = \sum_{i, j} \alpha_i\beta_jk(x_i, y_j)$$

Assuming that all $x_i \in \mathcal{X}$ and all $y_j \in \mathcal{X}$ 

* Add all the complement to make it a complete space, i.e. add all limits of Cauchy sequences. 

We now call this new $\bar{\mathcal{G}}$ a Reproducing Kernel Hilbert Space $\mathcal{H}$, and by constructing it satisfies $k(x, y) = \phi(x) \cdot \phi(y)$ 

One key property of RHKS is the **reproducing property**:

Let $f = \sum_{i=1}^r m_i k_{x_i}$, i.e. an element in RHKS, then $f \cdot k_x = f(x)$ . Notice the difference between $f$ and $f(x)$ here, the LHS is the inner product on two elements in RHKS, i.e. two images, while the RHS is a function value "reproducing" one of the function on LHS. 


# Issues of InfoNCE in self supervised learning

https://lilianweng.github.io/posts/2021-05-31-contrastive/

InfoNCE performance cannot be explained solely by the properties of the mutual information, but is influenced more by other factors, such as the formulation of the estimator and the architecture of the feature extractor. Essentially, representations with the same MI can have drastically different representational qualities. 

The example in the paper:

Suppose we have two inputs $a$ and $b$, and an encoder $E(\cdot|M)$ parameterized by an integer $M$ that maps $a$ uniformly randomly to $\{0, 2, \cdots, 2M\}$ and $b$ uniformly randomly to $\{1, 3, \cdots, 2M+1\}$. Let us denote $z_a = E(a|M)$ and $z_b = E(b|M)$, 

$$\text{MI}(z_a; z_b) = \sum_{z_a, z_b}p(z_a, z_b)\log \frac{p(z_a, z_b)}{p(z_a)p(z_b)}$$

For the joint distribution, $p(z_a=m_a, z_b=m_b) = \frac{1}{(M+1)^2}$

# Hilbert-Schmidt Independence Criterion (HSIC)

Suppose we have two input space $\mathcal{X}$ and $\mathcal{Y}$ , and we have two mapping functions $\phi: \mathcal{X} \rightarrow \mathcal{F}$ and $\psi: \mathcal{Y} \rightarrow \mathcal{G}$ , $\mathcal{F}$ and $\mathcal{G}$ being two reproducing kernel Hilbert spaces. HSIC is defined as:

$$ \text{HSIC}(X; Y) = ||\mathbb{E}[\phi(X) \cdot \psi(Y)] - \mathbb{E}[\phi(X)] \cdot \mathbb{E}[\psi(Y)]||^2_{\text{HS}}$$

Which is the norm of the covariance between the mapped elements. The Hilbert-Schmidt norm $||\cdot||_{\text{HS}}$ in finite dimensions is the usual Frobenius norm ( the square root of the sum of the squares of its elements).

Remember that the mapping function $\phi(X)$ and the corresponding inner product on RKHS can be represented by the kernel function, so by expanding the expression of HSIC we get:

$$
\begin{align}
\text{HSIC}(X; Y) &= ||\mathbb{E}[\phi(X) \cdot \psi(Y)] - \mathbb{E}[\phi(X)] \cdot \mathbb{E}[\psi(Y)]||^2_{\text{HS}}
\\ &= <(\mathbb{E}[\phi(X) \cdot \psi(Y)] - \mathbb{E}[\phi(X)] \cdot \mathbb{E}[\psi(Y)]), (\mathbb{E}[\phi(X) \cdot \psi(Y)] - \mathbb{E}[\phi(X)] \cdot \mathbb{E}[\psi(Y)])>_{\text{HS}}
\\ &= <\mathbb{E}[\phi(X) \cdot \psi(Y)], \mathbb{E}[\phi(X) \cdot \psi(Y)]>_{\text{HS}} 
\\ & - 2<\mathbb{E}[\phi(X) \cdot \psi(Y)], \mathbb{E}[\phi(X)] \cdot \mathbb{E}[\psi(Y)]>_\text{HS} 
\\ & + <\mathbb{E}[\phi(X)] \cdot \mathbb{E}[\psi(Y)], \mathbb{E}[\phi(X)] \cdot \mathbb{E}[\psi(Y)]>_\text{HS}
\\ &= \mathbb{E}<[\phi(X) \cdot \psi(Y)], [\phi(X') \cdot \psi(Y')]>_{\text{HS}} 
\\ & - 2\mathbb{E}<[\phi(X) \cdot \psi(Y)], [\phi(X')] \cdot \mathbb{E}[\psi(Y'')]>_\text{HS} 
\\ & + <\mathbb{E}[\phi(X) \cdot \phi(X')], \mathbb{E}[\psi(Y) \cdot \psi(Y')]>_\text{HS}
\\ &= \mathbb{E}[k(X, X')l(Y, Y')] - 2\mathbb{E}[k(X, X')l(Y, Y'')] + 
\mathbb{E}[k(X, X')]\mathbb{E}[l(Y, Y')]]
\end{align}
$$

assuming that $k$ and $l$ are two kernel functions on the two RKHS. 

Given $N$ iid drawn samples $\{(X_1, Y_1), (X_2, Y_2), \cdots, (X_N, Y_N)\}$, an empirical estimator is proposed by Gretton et al.

$$\widehat{\text{HSIC}}(X; Y) = \frac{1}{(N-1)^2}\text{Tr}(KHLH)$$

where $K$ and $L$ are the kernel matrices of $k$ and $l$ respectively on the $N$ data points from $\mathcal{X}$ and $\mathcal{Y}$, and $H = I - \frac{1}{N}11^T$  is called the centering matrix. 