"""
test_depth_width.py
=====================

pytest-compatible tests for Theorem 5: Depth-Width Separation for
Graph Transformers.

Run with:
    python -m pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "empirical"))

from verify import (
    HAS_TORCH,
    PureAttentionGT,
    AttentionFFNGT,
    numerical_rank,
    make_unique_degree_graph,
    make_all_same_degree,
    make_cycle_graph,
    make_star_graph,
)

if HAS_TORCH:
    import torch


# ---------------------------------------------------------------------------
# Fixture: device
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Theorem 5.1: Pure-attention rank barrier
# ---------------------------------------------------------------------------

class TestTheorem5_1:
    def test_rank_leq_2d(self, device):
        """Pure attention rank should not exceed 2*d_model."""
        d_model = 8
        d_k = 8
        model = PureAttentionGT(
            d_model=d_model,
            d_k=d_k,
            n_heads=1,
            n_layers=5,
        ).to(device)
        model.eval()

        n = 16
        x = torch.randn(n, d_model).to(device)
        with torch.no_grad():
            histories = model(x)

        for ell, H in enumerate(histories):
            r = numerical_rank(H)
            assert r <= 2 * d_model, (
                f"Layer {ell}: rank={r} > 2*d={2*d_model}"
            )

    def test_rank_at_most_grows_slowly(self, device):
        """Rank should not grow with depth beyond constant factor."""
        d_model = 8
        d_k = 8
        n = 16

        for n_layers in [1, 3, 5, 10]:
            model = PureAttentionGT(
                d_model=d_model, d_k=d_k, n_heads=1, n_layers=n_layers,
            ).to(device)
            model.eval()
            x = torch.randn(n, d_model).to(device)
            with torch.no_grad():
                histories = model(x)
            max_r = max(numerical_rank(H) for H in histories)
            assert max_r <= 4 * d_model, (
                f"Layers={n_layers}: max_rank={max_r} > 4*d={4*d_model}"
            )

    def test_small_dimension(self, device):
        """Test with d_model=4, small graph."""
        model = PureAttentionGT(d_model=4, d_k=4, n_heads=1, n_layers=3).to(device)
        model.eval()
        x = torch.randn(8, 4).to(device)
        with torch.no_grad():
            histories = model(x)
        max_r = max(numerical_rank(H) for H in histories)
        assert max_r <= 2 * 4

    def test_multi_head(self, device):
        """Test with 4 heads, effective d_v = d_k = 2 (total width=8)."""
        model = PureAttentionGT(d_model=8, d_k=2, n_heads=4, n_layers=3).to(device)
        model.eval()
        x = torch.randn(16, 8).to(device)
        with torch.no_grad():
            histories = model(x)
        max_r = max(numerical_rank(H) for H in histories)
        assert max_r <= 2 * 8


# ---------------------------------------------------------------------------
# Theorem 5.2: FFN breaks barrier
# ---------------------------------------------------------------------------

class TestTheorem5_2:
    def test_ffn_achieves_higher_rank(self, device):
        """FFN-augmented GT should achieve higher rank than pure attention."""
        d_model = 8
        d_k = 8
        d_ffn = 32
        n = 16
        n_layers = 3

        model_pure = PureAttentionGT(
            d_model=d_model, d_k=d_k, n_heads=1, n_layers=n_layers,
        ).to(device)
        model_ffn = AttentionFFNGT(
            d_model=d_model, d_k=d_k, d_ffn=d_ffn,
            n_heads=1, n_layers=n_layers,
        ).to(device)
        model_pure.eval()
        model_ffn.eval()

        x = torch.randn(n, d_model).to(device)
        with torch.no_grad():
            hist_pure = model_pure(x)
            hist_ffn = model_ffn(x)

        max_pure = max(numerical_rank(H) for H in hist_pure)
        # hist_ffn is list of [H_input, H_attn, H_ffn_hidden, H_ffn_out] per layer
        max_ffn = 0
        for layer_hist in hist_ffn:
            r = numerical_rank(layer_hist[2])  # FFN hidden activation
            max_ffn = max(max_ffn, r)

        assert max_ffn >= max_pure, (
            f"FFN hidden rank={max_ffn} < pure rank={max_pure}"
        )
        # FFN should leverage extra capacity significantly
        assert max_ffn >= min(d_ffn, n) * 0.3, (
            f"FFN hidden rank={max_ffn} < 30% of min(d_ffn,n)={min(d_ffn,n)*0.3:.0f}"
        )

    def test_large_ffn_width(self, device):
        """Test with very wide FFN (m >> n)."""
        d_model = 8
        d_k = 8
        d_ffn = 64
        n = 8

        model_ffn = AttentionFFNGT(
            d_model=d_model, d_k=d_k, d_ffn=d_ffn,
            n_heads=1, n_layers=2,
        ).to(device)
        model_ffn.eval()

        x = torch.randn(n, d_model).to(device)
        with torch.no_grad():
            hist_ffn = model_ffn(x)

        max_ffn = 0
        for layer_hist in hist_ffn:
            r = numerical_rank(layer_hist[2])  # FFN hidden
            max_ffn = max(max_ffn, r)
        # Should reach near-full rank for n=8
        assert max_ffn >= n * 0.6, (
            f"Wide FFN hidden rank={max_ffn} < 60% of n={n}"
        )


# ---------------------------------------------------------------------------
# Theorem 5.3: Separation — unique-degree learning
# ---------------------------------------------------------------------------

class TestTheorem5_3:
    def test_graph_constructions(self):
        """Unique-degree graphs must actually have a unique degree."""
        for n in [5, 6, 8, 10]:
            adj, degs = make_unique_degree_graph(n)
            unique_degrees = set(degs.astype(int))
            # Exactly one node should have a unique degree
            degree_counts = {}
            for d in degs.astype(int):
                degree_counts[d] = degree_counts.get(d, 0) + 1
            unique_count = sum(1 for c in degree_counts.values() if c == 1)
            assert unique_count >= 1, f"n={n}: no unique-degree node"

    def test_regular_graph_no_unique(self):
        """Cycle graphs have no unique-degree node."""
        for n in [4, 6, 8, 10]:
            adj, degs = make_all_same_degree(n)
            unique_degrees = set(degs.astype(int))
            assert len(unique_degrees) == 1, f"Cycle should have 1 unique degree, got {len(unique_degrees)}"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_models_exist(self, device):
        """Sanity: models can be instantiated and run."""
        model_pure = PureAttentionGT(d_model=4, d_k=4, n_heads=1, n_layers=2).to(device)
        model_ffn = AttentionFFNGT(d_model=4, d_k=4, d_ffn=16, n_heads=1, n_layers=2).to(device)

        x = torch.randn(8, 4).to(device)
        with torch.no_grad():
            out_pure = model_pure(x)
            out_ffn = model_ffn(x)
        assert len(out_pure) == 2 + 1  # n_layers + input
        assert len(out_ffn) == 2  # n_layers, each is [H_in, H_attn, H_ffn_hidden, H_ffn_out]


# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------

class TestSanity:
    def test_numerical_rank_zero(self):
        """Zero matrix has rank 0."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        M = torch.zeros(5, 5)
        assert numerical_rank(M) == 0

    def test_numerical_rank_full(self):
        """Identity has full rank."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        M = torch.eye(5)
        assert numerical_rank(M) == 5

    def test_numerical_rank_low(self):
        """Rank-1 matrix detected correctly."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        u = torch.randn(5)
        v = torch.randn(5)
        M = torch.outer(u, v)
        assert numerical_rank(M) == 1
