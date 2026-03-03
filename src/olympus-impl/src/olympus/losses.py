"""Associative memory losses: reconstruction and InfoNCE-style contrastive."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def associative_reconstruction_loss(memory_out: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """L2 reconstruction: || M(k) - v ||^2."""
    return F.mse_loss(memory_out, value, reduction="mean")


def _cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    an = F.normalize(a, p=2, dim=dim)
    bn = F.normalize(b, p=2, dim=dim)
    return (an * bn).sum(dim=dim)


def contrastive_associative_loss(
    memory_out: torch.Tensor,  # (B, D) or (D,) positive retrieval
    value_pos: torch.Tensor,  # (B, D) or (D,) positive value
    values_neg: torch.Tensor,  # (B, N_neg, D) or (N_neg, D) negative values from buffer
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE-style contrastive loss for associative memory.
    s+ = sim(M(k), v); s_j = sim(M(k), v_j).
    loss = -log( exp(s+/tau) / (exp(s+/tau) + sum_j exp(s_j/tau)) ).
    """
    if memory_out.dim() == 1:
        memory_out = memory_out.unsqueeze(0)
        value_pos = value_pos.unsqueeze(0)
    if values_neg.dim() == 2:
        values_neg = values_neg.unsqueeze(0)
    B = memory_out.size(0)
    sim_pos = _cosine_sim(memory_out, value_pos, dim=-1)  # (B,)
    mem = memory_out.unsqueeze(1)  # (B, 1, D)
    sim_neg = _cosine_sim(mem, values_neg, dim=-1)  # (B, N_neg)
    logit_pos = sim_pos / temperature
    logit_neg = sim_neg / temperature
    logits = torch.cat([logit_pos.unsqueeze(1), logit_neg], dim=1)  # (B, 1+N_neg)
    labels = torch.zeros(B, dtype=torch.long, device=memory_out.device)
    return F.cross_entropy(logits, labels)


def total_associative_loss(
    memory_out: torch.Tensor,
    value_pos: torch.Tensor,
    values_neg: torch.Tensor | None,
    lambda_contr: float = 0.0,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total loss = L_recon + lambda * L_contr.
    Returns (total_loss, L_recon, L_contr).
    """
    L_recon = associative_reconstruction_loss(memory_out, value_pos)
    if values_neg is not None and values_neg.numel() > 0 and lambda_contr > 0:
        L_contr = contrastive_associative_loss(
            memory_out, value_pos, values_neg, temperature
        )
        total = L_recon + lambda_contr * L_contr
    else:
        L_contr = torch.tensor(0.0, device=memory_out.device, dtype=memory_out.dtype)
        total = L_recon
    return total, L_recon, L_contr
