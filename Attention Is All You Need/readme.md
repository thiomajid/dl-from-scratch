---
paper: Attention Is All You Need
link: https://arxiv.org/abs/1706.03762
---

# Attention Is All You Need

## Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

where $Q, K, V$ are the query, key, and value matrices, respectively. The dot-product attention is scaled by the dimension of the key vectors, $\sqrt{d_k}$.
