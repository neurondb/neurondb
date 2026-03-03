"""Unit tests for Olympus memory: updates, retrieval, consolidation."""

import torch

from olympus import OlympusMemory, OlympusMAC, MemoryMLP
from olympus.losses import associative_reconstruction_loss, contrastive_associative_loss, total_associative_loss


def test_memory_mlp_forward():
    dim = 32
    m = MemoryMLP(dim, num_layers=2, hidden_mult=2)
    x = torch.randn(4, dim)
    y = m(x)
    assert y.shape == (4, dim)


def test_olympus_retrieve_shape():
    dim = 32
    mem = OlympusMemory(dim=dim, chunk_size=8, lambda_contr=0.0)
    q = torch.randn(dim)
    ctx = torch.randn(dim)
    y = mem.retrieve(q, ctx)
    assert y.shape == (dim,)


def test_olympus_forward_fast_updates_params():
    """Forward with loss should update fast-scale parameters (no longer equal to init)."""
    dim = 16
    mem = OlympusMemory(dim=dim, chunk_size=4, lambda_contr=0.0)
    fast_params_before = [p.clone() for p in mem.fast_mlp.parameters()]
    k = torch.randn(dim)
    v = torch.randn(dim)
    q = torch.randn(dim)
    mem.forward_fast_with_loss(k, v, q, value_buffer=None)
    for before, p in zip(fast_params_before, mem.fast_mlp.parameters()):
        assert not torch.allclose(before, p), "Fast memory params should change after update"


def test_olympus_step_chunk_no_crash():
    """Chunk step (mid update + consolidate) should run without error."""
    dim = 16
    mem = OlympusMemory(dim=dim, chunk_size=4, slow_interval_chunks=2, consolidate_R=2)
    keys = torch.randn(4, dim)
    values = torch.randn(4, dim)
    queries = torch.randn(4, dim)
    seg_mean = queries.mean(0)
    mem.step_chunk(keys, values, queries, seg_mean, do_consolidate_fast_mid=True)
    mem.step_chunk(keys, values, queries, seg_mean, do_consolidate_fast_mid=True)


def test_reconstruction_loss():
    out = torch.randn(4, 8)
    v = torch.randn(4, 8)
    loss = associative_reconstruction_loss(out, v)
    assert loss.dim() == 0 and loss.item() >= 0


def test_contrastive_loss():
    out = torch.randn(2, 8)
    v_pos = torch.randn(2, 8)
    v_neg = torch.randn(2, 4, 8)
    loss = contrastive_associative_loss(out, v_pos, v_neg, temperature=0.1)
    assert loss.dim() == 0 and loss.item() >= 0


def test_total_loss():
    out = torch.randn(2, 8)
    v = torch.randn(2, 8)
    v_neg = torch.randn(2, 4, 8)
    total, lr, lc = total_associative_loss(out, v, v_neg, lambda_contr=0.5)
    assert total.dim() == 0 and lr.dim() == 0 and lc.dim() == 0


def test_olympus_mac_forward():
    model = OlympusMAC(vocab_size=1000, dim=32, num_heads=2, num_layers=1, max_len=64, num_persistent=2, chunk_size=8)
    x = torch.randint(0, 1000, (2, 32))
    logits, loss, info = model(x, labels=x, return_memory_loss=True)
    assert logits.shape == (2, 32 - 1 - 2 - 1, 1000)  # segment after prefix and h_t
    assert loss is not None
    assert info is not None and "mem_loss" in info
