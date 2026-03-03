"""
Olympus memory module: Multi-Resolution Hierarchy (MRH), ISF, curvature-aware updates,
consolidation, and gated fusion. Implements per-scale MLPs and the full update/retrieval logic.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import total_associative_loss


# ---------------------------------------------------------------------------
# Single-scale memory MLP (Titans-style, one layer or more)
# ---------------------------------------------------------------------------


class MemoryMLP(nn.Module):
    """MLP used as associative memory: input key -> output value. L layers, hidden mult."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 1,
        hidden_mult: int = 2,
        activation: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        act = nn.SiLU() if activation == "silu" else nn.GELU()
        layers = []
        in_d = dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_d, dim * hidden_mult))
            layers.append(nn.LayerNorm(dim * hidden_mult))
            layers.append(act)
            in_d = dim * hidden_mult
        layers.append(nn.Linear(in_d, dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def get_param_vector(self) -> torch.Tensor:
        """Flattened parameters for consolidation delta norm."""
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def set_param_vector(self, vec: torch.Tensor) -> None:
        """Restore from flattened vector (in-place)."""
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(vec[offset : offset + n].view_as(p.data))
            offset += n


# ---------------------------------------------------------------------------
# ISF + curvature state (per-parameter omega and h)
# ---------------------------------------------------------------------------


class ISFCurvatureState:
    """Tracks importance (omega) and curvature (h) for one memory module's parameters."""

    def __init__(
        self,
        param_shapes: list[tuple[int, ...]],
        device: torch.device,
        dtype: torch.dtype,
        gamma: float = 0.99,
        rho: float = 0.95,
        epsilon: float = 1e-8,
    ):
        self.gamma = gamma
        self.rho = rho
        self.epsilon = epsilon
        self.omega: list[torch.Tensor] = [
            torch.zeros(s, device=device, dtype=dtype) for s in param_shapes
        ]
        self.h: list[torch.Tensor] = [
            torch.zeros(s, device=device, dtype=dtype) for s in param_shapes
        ]

    @staticmethod
    def from_module(module: nn.Module, device: torch.device, dtype: torch.dtype, **kwargs) -> ISFCurvatureState:
        shapes = [tuple(p.shape) for p in module.parameters()]
        return ISFCurvatureState(shapes, device, dtype, **kwargs)

    def update(self, grads: list[torch.Tensor], alpha_base: float, beta: float) -> list[torch.Tensor]:
        """
        Update omega and h from gradients; return per-parameter decay alpha^(i) and step scale.
        alpha^(i) = alpha_base * sigmoid(-beta * omega^(i))
        theta^(i) = 1 / sqrt(h^(i) + eps)
        """
        alphas_out = []
        for i, g in enumerate(grads):
            g2 = g.detach().pow(2)
            self.omega[i] = self.gamma * self.omega[i] + (1 - self.gamma) * g2
            self.h[i] = self.rho * self.h[i] + (1 - self.rho) * g2
            alpha_i = alpha_base * torch.sigmoid(-beta * self.omega[i])
            alphas_out.append(alpha_i)
        return alphas_out

    def step_scales(self) -> list[torch.Tensor]:
        """Per-parameter step size scale: 1 / sqrt(h + eps)."""
        return [1.0 / (t.sqrt() + self.epsilon) for t in self.h]


# ---------------------------------------------------------------------------
# Momentum state S_t for one scale (Titans update)
# ---------------------------------------------------------------------------


def get_grad_list(module: nn.Module) -> list[torch.Tensor]:
    return [p.grad if p.grad is not None else torch.zeros_like(p) for p in module.parameters()]


def apply_surprise_update(
    module: nn.Module,
    S_list: list[torch.Tensor],
    eta: float,
    theta_scales: list[torch.Tensor],
    base_theta: float,
    grads: list[torch.Tensor],
) -> list[torch.Tensor]:
    """S_t = eta * S_{t-1} - theta^(i) * grad (Titans momentum + step)."""
    new_S = []
    for s, scale, g in zip(S_list, theta_scales, grads):
        step = base_theta * scale * g.detach()
        new_S.append(eta * s - step)
    return new_S


def apply_weight_update(
    module: nn.Module,
    alpha_list: list[torch.Tensor],
    S_list: list[torch.Tensor],
) -> None:
    """M_t = (1 - alpha^(i)) * M_{t-1} + S_t. In-place on module parameters."""
    for p, alpha, s in zip(module.parameters(), alpha_list, S_list):
        p.data.mul_(1 - alpha).add_(s)


# ---------------------------------------------------------------------------
# One scale with ISF + curvature (fast or mid)
# ---------------------------------------------------------------------------


class ScaleState:
    """Holds MLP, ISF/curvature state, and momentum S for one scale."""

    def __init__(
        self,
        mlp: MemoryMLP,
        device: torch.device,
        dtype: torch.dtype,
        gamma: float = 0.99,
        rho: float = 0.95,
        epsilon: float = 1e-8,
    ):
        self.mlp = mlp
        self.isf_state = ISFCurvatureState.from_module(mlp, device, dtype, gamma=gamma, rho=rho, epsilon=epsilon)
        self.S: list[torch.Tensor] = [
            torch.zeros_like(p) for p in mlp.parameters()
        ]


# ---------------------------------------------------------------------------
# Olympus full memory: 3 scales, gated fusion, consolidation
# ---------------------------------------------------------------------------


class OlympusMemory(nn.Module):
    """
    Olympus multi-resolution memory: fast (token), mid (chunk), slow (P chunks).
    Uses ISF, curvature-aware step sizes, optional contrastive loss, and consolidation.
    """

    def __init__(
        self,
        dim: int,
        chunk_size: int = 32,
        slow_interval_chunks: int = 8,
        # Fast / Mid / Slow MLP depths and hidden mults (paper: 1,2,3 and 2,2,4)
        fast_layers: int = 1,
        fast_hidden_mult: int = 2,
        mid_layers: int = 2,
        mid_hidden_mult: int = 2,
        slow_layers: int = 3,
        slow_hidden_mult: int = 4,
        # Inner-loop
        theta_fast: float = 0.01,
        theta_mid: float = 0.005,
        theta_slow: float = 0.001,
        alpha_base: float = 0.01,
        eta_momentum: float = 0.9,
        # ISF & curvature
        gamma_isf: float = 0.99,
        rho_curv: float = 0.95,
        epsilon_curv: float = 1e-8,
        beta_isf: float = 0.5,
        # Contrastive
        lambda_contr: float = 0.2,
        contrastive_temp: float = 0.07,
        neg_buffer_size: int = 16,
        # Consolidation
        consolidate_R: int = 32,
        consolidate_zeta: float = 0.001,
        tau_c_percentile: float = 0.9,
        # Gate
        ctx_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.P = slow_interval_chunks
        self.theta_fast = theta_fast
        self.theta_mid = theta_mid
        self.theta_slow = theta_slow
        self.alpha_base = alpha_base
        self.eta_momentum = eta_momentum
        self.gamma_isf = gamma_isf
        self.rho_curv = rho_curv
        self.epsilon_curv = epsilon_curv
        self.beta_isf = beta_isf
        self.lambda_contr = lambda_contr
        self.contrastive_temp = contrastive_temp
        self.neg_buffer_size = neg_buffer_size
        self.consolidate_R = min(chunk_size, consolidate_R)
        self.consolidate_zeta = consolidate_zeta
        self.tau_c_percentile = tau_c_percentile

        self.fast_mlp = MemoryMLP(dim, fast_layers, fast_hidden_mult)
        self.mid_mlp = MemoryMLP(dim, mid_layers, mid_hidden_mult)
        self.slow_mlp = MemoryMLP(dim, slow_layers, slow_hidden_mult)

        ctx_dim = ctx_dim or dim
        self.gate = nn.Linear(dim + ctx_dim, 3)

        self._fast_state: Optional[ScaleState] = None
        self._mid_state: Optional[ScaleState] = None
        self._slow_state: Optional[ScaleState] = None
        self._chunk_step = 0
        self._slow_chunk_counter = 0
        self._consolidation_scores: list[float] = []

    def _ensure_states(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._fast_state is None:
            self._fast_state = ScaleState(
                self.fast_mlp, device, dtype,
                gamma=self.gamma_isf, rho=self.rho_curv, epsilon=self.epsilon_curv,
            )
        if self._mid_state is None:
            self._mid_state = ScaleState(
                self.mid_mlp, device, dtype,
                gamma=self.gamma_isf, rho=self.rho_curv, epsilon=self.epsilon_curv,
            )
        if self._slow_state is None:
            self._slow_state = ScaleState(
                self.slow_mlp, device, dtype,
                gamma=self.gamma_isf, rho=self.rho_curv, epsilon=self.epsilon_curv,
            )

    def retrieve(self, q: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Gated fusion: g = softmax(W_gate [q; ctx]), y = sum_k g_k M_k*(q)."""
        with torch.no_grad():
            r_fast = self.fast_mlp(q)
            r_mid = self.mid_mlp(q)
            r_slow = self.slow_mlp(q)
        ctx = ctx if ctx is not None else q
        gate_in = torch.cat([q, ctx], dim=-1)
        if gate_in.dim() == 1:
            gate_in = gate_in.unsqueeze(0)
        g = F.softmax(self.gate(gate_in), dim=-1)  # (1, 3) or (B, 3)
        if g.dim() == 1:
            g = g.unsqueeze(0)
        r_fast = r_fast.unsqueeze(0) if r_fast.dim() == 1 else r_fast
        r_mid = r_mid.unsqueeze(0) if r_mid.dim() == 1 else r_mid
        r_slow = r_slow.unsqueeze(0) if r_slow.dim() == 1 else r_slow
        y = g[:, 0:1] * r_fast + g[:, 1:2] * r_mid + g[:, 2:3] * r_slow
        return y.squeeze(0) if y.size(0) == 1 else y

    def forward_fast_with_loss(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        value_buffer: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward fast memory with loss; update fast scale (ISF + curvature + momentum).
        Returns (retrieval_for_loss, total_loss, L_recon, L_contr).
        """
        device = k.device
        dtype = k.dtype
        self._ensure_states(device, dtype)

        out = self.fast_mlp(k)
        total, L_recon, L_contr = total_associative_loss(
            out, v, value_buffer,
            lambda_contr=self.lambda_contr,
            temperature=self.contrastive_temp,
        )
        self.fast_mlp.zero_grad(set_to_none=True)
        total.backward()
        grads = get_grad_list(self.fast_mlp)

        alpha_list = self._fast_state.isf_state.update(
            grads, self.alpha_base, self.beta_isf
        )
        step_scales = self._fast_state.isf_state.step_scales()
        self._fast_state.S = apply_surprise_update(
            self.fast_mlp,
            self._fast_state.S,
            self.eta_momentum,
            step_scales,
            self.theta_fast,
            grads,
        )
        apply_weight_update(self.fast_mlp, alpha_list, self._fast_state.S)

        with torch.no_grad():
            retrieval = self.fast_mlp(q)
        return retrieval, total.detach(), L_recon.detach(), L_contr.detach()

    def step_chunk(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        segment_mean: torch.Tensor,
        do_consolidate_fast_mid: bool = True,
    ) -> None:
        """
        End-of-chunk: update mid from chunk loss; optionally consolidate fast->mid.
        keys/values/queries: (C, D) for chunk size C.
        """
        device = keys.device
        dtype = keys.dtype
        self._ensure_states(device, dtype)

        # Mid update: one gradient step on chunk loss
        chunk_loss = 0.0
        for i in range(keys.size(0)):
            out = self.mid_mlp(keys[i])
            chunk_loss = chunk_loss + F.mse_loss(out, values[i])
        chunk_loss = chunk_loss / keys.size(0)
        self.mid_mlp.zero_grad(set_to_none=True)
        chunk_loss.backward()
        grads = get_grad_list(self.mid_mlp)
        alpha_list = self._mid_state.isf_state.update(
            grads, self.alpha_base, self.beta_isf
        )
        step_scales = self._mid_state.isf_state.step_scales()
        self._mid_state.S = apply_surprise_update(
            self.mid_mlp,
            self._mid_state.S,
            self.eta_momentum,
            step_scales,
            self.theta_mid,
            grads,
        )
        apply_weight_update(self.mid_mlp, alpha_list, self._mid_state.S)

        # Consolidation fast -> mid
        if do_consolidate_fast_mid:
            self._consolidate(self.fast_mlp, self.mid_mlp, queries)

        self._chunk_step += 1
        self._slow_chunk_counter += 1

        # Every P chunks: update slow (consolidate mid -> slow)
        if self._slow_chunk_counter >= self.P:
            self._consolidate(self.mid_mlp, self.slow_mlp, queries)
            self._slow_chunk_counter = 0

    def _consolidate(
        self,
        src_mlp: MemoryMLP,
        tgt_mlp: MemoryMLP,
        queries: torch.Tensor,
    ) -> None:
        """One distillation step: minimize sum_r || tgt(q_r) - src*(q_r) ||^2."""
        R = min(queries.size(0), self.consolidate_R)
        indices = torch.randperm(queries.size(0), device=queries.device)[:R]
        q_r = queries[indices]
        with torch.no_grad():
            v_hat = src_mlp(q_r)
        pred = tgt_mlp(q_r)
        loss = F.mse_loss(pred, v_hat)
        tgt_mlp.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for p in tgt_mlp.parameters():
                if p.grad is not None:
                    p.data.sub_(p.grad, alpha=self.consolidate_zeta)


# ---------------------------------------------------------------------------
# Olympus-MAC: Memory-as-Context with causal attention
# ---------------------------------------------------------------------------


class OlympusMAC(nn.Module):
    """
    Olympus-MAC: persistent tokens + gated Olympus retrieval (h_t) + segment,
    then causal attention. Simplified: single attention layer over [prefix | h_t | segment].
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        max_len: int = 512,
        num_persistent: int = 4,
        chunk_size: int = 32,
        **olympus_kw,
    ):
        super().__init__()
        self.dim = dim
        self.num_persistent = num_persistent
        self.chunk_size = chunk_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)
        self.olympus = OlympusMemory(dim=dim, chunk_size=chunk_size, ctx_dim=dim, **olympus_kw)
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_memory_loss: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """
        input_ids: (B, T). Optionally run in chunks; here we do one forward with a single
        retrieval token per segment for simplicity.
        """
        B, T = input_ids.shape
        x = self.embed(input_ids) + self.pos_embed[:, :T]
        device = x.device
        dtype = x.dtype

        # Persistent prefix
        p = self.embed.weight.new_zeros(1, self.num_persistent, self.dim)
        p = p + self.pos_embed[:, : self.num_persistent]

        # Segment-level: one retrieval per chunk (simplified: one retrieval for full seq)
        q = self.W_Q(x[:, -1])
        k = self.W_K(x[:, -1])
        v = self.W_V(x[:, -1])
        ctx = x.mean(dim=1)
        h_t, mem_loss, lr, lc = self.olympus.forward_fast_with_loss(k, v, q, None)
        h_t = h_t.unsqueeze(1)

        # Build context: [p | h_t | x]
        seq = torch.cat([p.expand(B, -1, -1), h_t.expand(B, -1, -1), x], dim=1)
        # Causal mask
        L = seq.size(1)
        mask = torch.triu(torch.ones(L, L, device=device, dtype=dtype) * float("-inf"), diagonal=1)
        for layer in self.attn_layers:
            seq = layer(seq, mask=mask)
        seq = self.ln(seq)
        logits = self.head(seq[:, self.num_persistent + 1 :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            if return_memory_loss and mem_loss is not None:
                loss = loss + 0.1 * mem_loss

        info = None
        if return_memory_loss:
            info = {"mem_loss": mem_loss.item(), "L_recon": lr.item(), "L_contr": lc.item()}
        return logits, loss, info
