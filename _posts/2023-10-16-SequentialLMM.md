---
layout: post
title:  Reviewing the paper Longitudinal data analysis using matrix completion
date:   2023-10-16 21:44:00
tags: math, machine learning
categories: math
math: true
header-includes:
  - \usepackage{mathrsfs}
  - \usepackage{amsbsy}
---

Let $\mathbf{b}(t) = (b_1(t), b_2(t), \cdots, b_K(t))$ be a vector of $K$ basis elements at timepoint $t$. It is considered a truncated basis because the assumption of $K$ elements may not sufficiently cover the full vector space. 

# Sequential LMM

The general goal is to get a mapping from time $\mathbf{t}_i$ to  target space $\mathbf{y}_i$ for every observed sample $i \in \{1, 2, \cdots, N\}$. 

The paper called a mean at time t, i.e. $\mu(t)$ a _fixed effect_. $\mu(t)$ is defined as a weighted summary on the truncated vector space $\mathbf{b}(t)$ of length $K$, i.e. 

$$\mu(t) = \mathbf{m} \cdot \mathbf{b}(t)$$

where $\mathbf{m} = (m_1, m_2, \cdots, m_K)$, $\mathbf{b}(t) = (b_1(t), b_2(t), \cdots, b_K(t))$ 

The paper then introduces a per-sample coefficient ,  aka the individual _random effect_, $\mathbf{w}$. $\mathbf{w}$ can be an $N \times K$ matrix, where each row is for a data sample. For a sample's coefficient $\mathbf{w}_i, i\in \{1, 2, \cdots, N\}$ of dim $K$, we assume that there are $n_i$ time steps for this sample, i.e. the time sequence looks like $\{t_{i, 1}, t_{i. 2}, \cdots,  t_{i, n_i}\}$  of length $n_i$, and at every time step $t_{i, j}$ there is a fixed effect $\mu(t_{i, j})$ , leading to a $n_i$ - dim vector of $(\mu(t_{i, 1}), \mu(t_{i, 2}) \cdots, \mu(t_{i, n_i}))$ as the fixed effect for over all this individual's time sequence. We denote this fixed effect vector of dim $n_i$ over this $i-$th individual sample as $\boldsymbol{\mathbf{\mu}}_i$. For each scalar $\mu(t_{i, j})$, it follows the same definition above and takes the truncated basic vector at time $t_{i,j}$, i.e. $\mu(t_{i, j}) = \mathbf{m} \cdot \mathbf{b}(t_{i, j})$  . 

To model the target distribution $\mathbf{y}_i$ from the observed individual $\mathbf{w}_i$,. we follow the traditional LMM approach via a conditional gaussian:

$$\mathbf{y}_i|\mathbf{w}_i \sim \mathcal{N}(\boldsymbol{\mathbf{\mu}}_i + \mathbf{B}_i \cdot \mathbf{w}_i, \sigma^2 \mathbf{I}_{n_i})$$

assuming that the target distribution $\mathbf{y}_i$ is a sequence of scalar,  each scalar value representing the state at $t_{i, j}$, $j \in \{1, 2, \cdots, n_i\}$. Also, $\mathbf{B}_i$ is naturally a matrix of dimension $(n_i, K)$ to map the $K$-dim per-sample coefficient $\mathbf{w}_i$ to the same time sequence space. Intuitively, $\mathbf{B}_i = [\mathbf{b}(t_{i, 1}), \mathbf{b}(t_{i, 2}), \cdots, \mathbf{b}(t_{i, n_i})]$ 

# Low Rank Sequential LMM

We restrict the latent space to $q < K$ dimensions and learn from the data the mapping $\mathbf{A} \in \mathbb{R}^{K \times q}$  in this reduced dimension LMM:

$$\mathbf{y}_i|\mathbf{w}_i \sim \mathcal{N}(\boldsymbol{\mathbf{\mu}}_i + \mathbf{B}_i \cdot A \cdot \mathbf{w}_i, \sigma^2 \mathbf{I}_{n_i})$$
where now the individual sample $\mathbf{w}_i$ also reduces its dim to $q$ . Because the optimizations are on $\mathbf{w}_i$ and $\mathbf{A}$ , an EM algorithm can be applied. 


# This paper

[original paper](https://www.researchgate.net/publication/327858691_Longitudinal_data_analysis_using_matrix_completion)

Instead of using existing $\{t_{i, 1}, t_{i. 2}, \cdots,  t_{i, n_i}\}$  for every sample $i$, the paper defined a universal grid agnostic of actual time series per sample:

$$G = [\tau_1, \tau_2, \cdots, \tau_T]$$ 
where $\tau_1$ is the global minimal time across all sample $i$, $\tau_T$ is the global maximal time across all sample $i$, and $T$ is a hyperparameter defining how accurate the grid is. Ideally if $T$ is infinity the grid becomes a continuous space that can capture every $t_{i, j}$. 

For every $t_{i, j}$ , we locate its corresponding 

$$g_i(j) = \text{argmin}_{1 \le k \le T}(|\tau_k - t_{i, j}|)$$

i.e. the index of $\tau$ that is most close to $t_{i, j}$. We also define $\tilde{\mathbf{y}}_{i, j} = \mathbf{y}_{i, g_i(j)}$.

For sample $i$, now we can map the original time sequence $\{t_{i, 1}, t_{i. 2}, \cdots,  t_{i, n_i}\}$ to $\{g_i(1), g_i(2), \cdots,  g_i(n_i)\}$, and the paper called this new sequence of $g_i$ as $O_i$. For sample $i$, any index in $[1, T]$ that not in $O_i$ is marked as a missing target sample, which is quite natural because $G$ is designed to fit all samples so will always greater than or equal to the time sequence of individual sample. The paper introduces an additional notation $\Omega$ to represent $O_i$ over all samples, i.e. $\Omega = \{(i, j): 1 \le i \le N, j \in O_i \}$. Finally, the paper reformulated the label space to a $N \times T$ matrix $P_\Omega(\mathbf{Y})$ where:

$$P_\Omega(\mathbf{Y})[i, j] = \mathbf{y}_i[O_{i}[j]]$$

if target is not missing, 0 otherwise. The operator $P_\Omega(\cdot)$ is a general projection that projects input from space $\Omega$ to the full grid space, by keeping the original value if the input has a corresponding value at that grid position, otherwise fill the value at that grid position with 0. 

The paper still used the same $\mathbf{b}(t)$ of dim $K$, but $\mathbf{B}_i$ is now replaced by a sample-agnostic $\mathbf{B} = [\mathbf{b}(\tau_1), \mathbf{b}(\tau_2), \cdots, \mathbf{b}(\tau_T)]$  .

Again following the same approach we want to minimize the difference between every $\tilde{\mathbf{y}}_{i, j}$ and every predicted $\mathbf{w}_i \cdot \mathbf{b}(\tau_{g_i(j)})$, i.e. 

$$\text{argmin}_{\mathbf{w}_i} |\tilde{\mathbf{y}}_{i, j} - \mathbf{w}_i \cdot \mathbf{b}(\tau_{g_i(j)})|$$ 
for all $i$, which is equivalent to the matrix form of:

$$\text{argmin}_{\mathbf{W}}||P_\Omega(\mathbf{Y} - \mathbf{W}\cdot \mathbf{B}^T)||_F$$

where $\mathbf{W} = [\mathbf{w}_1, \mathbf{w}_2, \cdots, \mathbf{w}_N]$ of dim $(N \times K)$,   $\mathbf{B} = [\mathbf{b}(\tau_1), \mathbf{b}(\tau_2), \cdots, \mathbf{b}(\tau_T)]$ of dim $(T, K)$ , and $||\cdot||_F$ being Frobenius norm, i.e. the square root of the sum of matrix elements. Again projection operator $P_\Omega$ guarantees that we only keep all available $\mathbf{y}_i$ and skip all missing y in the full grid.

## Soft-Longitudinal-Impute

To optimize the above loss with missing ys, the paper utilized the fact that optimization of

$$\text{argmin}_\mathbf{W} (\frac{1}{2} ||\mathbf{Y} - \mathbf{W} \cdot \mathbf{B}^T||_F^2 + ||\mathbf{W}||_*) $$

has a unique solution $\mathbf{W} = S_\lambda(\mathbf{Y}\mathbf{B})$, where $S_\lambda(X) = UD_\lambda V$ and $X = UDV^T$ is the SVD of $X$. Notation:  $||\cdot||_*$ is the nuclear norm, i.e. the sum of singular values. $D_\lambda$ is a diagonal matrix called soft-thresholding, which does a threshold on the original diagonal matrix $D$ by $D_\lambda[i, i] = \text{max}(D[i, i]-\lambda, 0)$, i.e. subtract $\lambda$ from every diagonal value, keep the subtraction value if it is greater than 0 otherwise set it to 0. 

Intuitively, this algorithm approaches $\mathbf{W}$ iteratively by a new Y with imputed values to fill in the missing places. For a given approximation of the solution $\mathbf{W}^{\text{old}}$ , we use $\mathbf{W}^{\text{old}}$ to impute unknown elements of $\mathbf{Y}$ obtaining $\tilde{\mathbf{Y}}$ . Then, we construct the next approximation $\mathbf{W}^{\text{new}}$ by running the above SVD based solution on 

$$\mathbf{W}^{\text{new}} = \text{argmin}_\mathbf{W} (\frac{1}{2} ||\tilde{\mathbf{Y}} - \mathbf{W} \cdot \mathbf{B}^T||_F^2 + ||\mathbf{W}||_*) $$

So there wouldn’t be any projection operator. 

Also because the optimization is sensitive to $\lambda$, the paper proposed that run the optimization with decreasing $\lambda$ to refine the optimization results. 




