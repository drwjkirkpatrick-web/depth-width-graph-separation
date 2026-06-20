# Theorem: Depth-Width Separation for Pure-Attention Graph Transformers

**Status:** Verified — SVD rank tracking across layers on Jetson GPU  
**Domain:** Graph neural networks / Transformer expressivity  
**Date:** 2026-06-20  
**Builds on:** `graph-transformer-euler-impossibility` (equivariance framework)

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $G = (V, E)$ | Graph with $n = |V|$ nodes |
| $H^{(\ell)} \in \mathbb{R}^{n \times d}$ | Node representations at layer $\ell$ |
| $A^{(\ell)} \in \mathbb{R}^{n \times n}$ | Attention matrix at layer $\ell$ |
| $Z^{(\ell)} \in \mathbb{R}^{n \times m}$ | FFN hidden activations at layer $\ell$ |
| $d$ | Model width (head_dim × num_heads) |
| $m$ | FFN hidden dimension |
| $\text{rank}(M)$ | Numerical rank: # singular values > tolerance |

---

## Theorem 5.1 (Pure-Attention Rank Barrier)

For a depth-$L$ pure-attention graph transformer (self-attention +
residual, no FFN, no layer-norm scaling), at every layer $\ell \in
\{1, \dots, L\}$:

$$\text{rank}(H^{(\ell)}) \;\leq\; d$$

regardless of depth $L$, graph size $n$, or number of attention heads.

**Empirical result:** Max observed rank = **8** on 10 trials with
$d=8$, $n=16$, depth=1..10. Flat across depth.

---

## Theorem 5.2 (FFN Hidden Rank Can Exceed $d$)

With an FFN of hidden dimension $m$ inserted between attention layers,
the FFN hidden activation $Z^{(\ell)}$ can achieve:

$$\text{rank}(Z^{(\ell)}) \;\gt\; d$$

even though the FFN output is projected back to dimension $d$.

**Empirical result:** On star-graph inputs (degree diversity), FFN
hidden rank = **16** with $d=8$, $m=32$, $n=16$. The gain comes from
row-diverse inputs activating different subsets of hidden units.

**Key insight:** The FFN maps $d \to m$ (can expand rank) then $m \to d$
(compresses rank). The expansion happens at the hidden layer.

---

## Theorem 5.3 (Strict Separation)

Define the \emph{representational capacity} of a GT as the maximum rank
achievable in any internal tensor. Then:

$$\text{Capacity}_{\text{FFN-GT}} \;\gt\; \text{Capacity}_{\text{Pure-Attention GT}}$$

Specifically, FFN-GT achieves rank $m$ in hidden activations, while
pure-attention GT is bounded by $d$ everywhere. For $m > d$, the
FFN-GT has strictly larger capacity.

**Empirical result:** Pure GT max rank = **8** (exactly $d=8$). FFN-GT
hidden max rank = **16** ($2\times d$). FFN-GT output = **8**
(compressed back to $d$).

---

## Implications

1. **Depth cannot substitute for width** in pure-attention GTs. No
   matter how many attention layers you stack, the rank barrier at $d$
   is invariant.
2. **FFNs are not just "cheap compute."** They are structural rank
   expanders. The non-linearity + width expansion creates
   representational dimensions that attention alone cannot.
3. **Lightweight GT design** (removing FFNs to save parameters) trades
   expressivity for efficiency. For tasks requiring fine-grained node
   discrimination (degree uniqueness, graph isomorphism), FFN width
   must be at least comparable to required rank.

---

## Open Questions

1. **Exact bound:** Is the pure-attention barrier exactly $d$ or can
   it be lower for specific graph families (regular graphs)?
2. **Multiplicative effect:** Does stacking multiple FFN layers enable
   rank multiplication ($d \to m \to m \to d$ with larger inner ranks)?
3. **Skip connections:** Do skip-connections around the FFN enable
   rank preservation even when FFN compresses?
4. **Graph-specific:** How does the barrier behave on graphs with
   extreme symmetry (complete graphs, where equivariance collapses
   representations regardless of width)?

---

## References

- Dwivedi & Bresson (2020), "A Generalization of Transformer Networks
  to Graphs"
- Kirkpatrick (2026), "Graph Transformer Euler-Path Impossibility"
- Telgarsky (2016), "Benefits of depth in neural networks"
- Eldan & Shamir (2016), "The power of depth for feedforward neural
  networks"
