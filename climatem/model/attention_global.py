"""
Attention-based global transition for the hierarchical model (thesis, opt-in).

Drop-in replacement for ``TransitionModelGlobal`` in ``tsdcd_latent_hierarchy_raw.py``: same call signature
``forward(z, i, k, y_co2=None, y_ch4=None)``, same ``logvar`` parameter, same output (the mean of the next global
latent). What changes is *how* the inputs are combined:

- every scalar input becomes a token: the global latent at each of the ``tau`` past steps, CO2 at each of its lag
  samples, CH4 at each of its lag samples (Shan's loader gives 6 response-interval samples per forcing), plus, when
  ``data_params.global_memory_years`` is set, one exponential forcing-memory value per timescale (``mem10y`` ...);
- one learned query per output latent attends over the tokens (single head, softmax);
- the attention weights are the implicit causal graph: one weight per (variable, lag), readable as effect size and lag
  response (variant A);
- optionally a learned gate per (output, token) multiplies the attention logits (added as ``log gate``), with Gumbel
  noise during training; an L1 penalty on the gates (``gate_penalty``) lets the trainer enforce sparsity (variant B).

Selected from the config with ``model_params.global_transition = "attention"``; the default ``"mlp"`` keeps Shan's model
unchanged.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shan's RESPONSE_INTERVALS, oldest first
FORCING_LAG_LABELS = ["lag60", "lag12-59", "lag5-11", "lag2-4", "lag1", "lag0"]


def _mlp(num_layers: int, num_hidden: int, num_input: int, num_output: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    n_in = num_input
    for _ in range(num_layers):
        layers += [nn.Linear(n_in, num_hidden), nn.LeakyReLU()]
        n_in = num_hidden
    layers.append(nn.Linear(n_in, num_output))
    return nn.Sequential(*layers)


class AttentionTransitionGlobal(nn.Module):
    """Attention over (variable, lag) tokens -> mean of the next global latent, one query per output latent."""

    def __init__(
        self,
        d: int,
        d_z: int,
        tau: int,
        num_layers: int,
        num_hidden: int,
        num_output: int = 2,
        d_y_co2: int = 0,
        d_y_ch4: int = 0,
        attention_dim: int = 16,
        gate: bool = False,
        gate_init: float = 3.0,
        n_forcing_lags: Optional[int] = None,
        gate_anchor: bool = False,
        memory_labels: Optional[List[str]] = None,  # thesis: labels of the forcing-memory channels after the 6 lags
    ):
        super().__init__()
        self.d = d
        self.d_z = d_z
        self.tau = tau
        self.num_output = 1  # same as TransitionModelGlobal: the variance is a separate learned logvar
        self.logvar = nn.Parameter(torch.ones(d, d_z) * -4)
        self.use_co2 = d_y_co2 > 0
        self.use_ch4 = d_y_ch4 > 0
        self.use_exogenous = self.use_co2 or self.use_ch4
        self.n_forcing_lags = n_forcing_lags if n_forcing_lags is not None else tau + 1  # Shan: tau + 1 samples
        self.memory_labels = list(memory_labels) if memory_labels else []
        self.attention_dim = attention_dim
        self.use_gate = gate

        # token layout, oldest first inside each variable block
        self.token_labels: List[str] = [f"z2 t-{tau - j}" for j in range(tau)]
        self.n_z_tokens = tau
        if self.use_co2:
            self.token_labels += [f"CO2 {lab}" for lab in self._lag_labels()]
        if self.use_ch4:
            self.token_labels += [f"CH4 {lab}" for lab in self._lag_labels()]
        self.n_tokens = len(self.token_labels)
        self.variable_of_token = torch.tensor(
            [0] * tau
            + ([1] * self.n_forcing_lags if self.use_co2 else [])
            + ([2] * self.n_forcing_lags if self.use_ch4 else [])
        )

        n_out = d * d_z
        # scalar value -> embedding, one per variable kind (z2, CO2, CH4) so each keeps its own scale
        self.value_embed = nn.ModuleList(nn.Linear(1, attention_dim) for _ in range(3))
        self.pos_embed = nn.Parameter(torch.randn(self.n_tokens, attention_dim) * 0.1)
        self.key = nn.Linear(attention_dim, attention_dim, bias=False)
        self.value = nn.Linear(attention_dim, attention_dim, bias=False)
        self.query = nn.Parameter(torch.randn(n_out, attention_dim) * 0.1)
        self.head = nn.ModuleList(_mlp(num_layers, num_hidden, attention_dim, self.num_output) for _ in range(n_out))
        self.use_anchor = gate and gate_anchor
        if gate:
            # sigmoid(3) = 0.95: every edge open at the start, like PICABU's mask init
            self.gate_logits = nn.Parameter(torch.full((n_out, self.n_tokens), float(gate_init)))
            # anchor: the self-link token (z2 at t-1) keeps gate = 1 and is not penalised. Softmax is shift-invariant,
            # so without an anchor the L1 shrinks ALL gates together for free (run 11867039); cf. Mask2Cause's γI.
            anchor = torch.zeros(self.n_tokens, dtype=torch.bool)
            anchor[self.n_z_tokens - 1] = True
            self.register_buffer("anchor_mask", anchor)
        self.last_attention: Optional[torch.Tensor] = None  # (b, n_out, n_tokens) of the last forward pass
        self._attn_batch: List[Optional[torch.Tensor]] = [None] * n_out

    def _lag_labels(self) -> List[str]:
        if self.n_forcing_lags == len(FORCING_LAG_LABELS) + len(self.memory_labels):
            return list(FORCING_LAG_LABELS) + self.memory_labels
        return [f"lag{self.n_forcing_lags - 1 - j}" for j in range(self.n_forcing_lags)]

    # ------------------------------------------------------------------ gates
    def gate(self, sample: bool) -> torch.Tensor:
        """(n_out, n_tokens) in (0, 1); Gumbel-logistic noise when sampling (training)."""
        logits = self.gate_logits
        if sample:
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            logits = logits + torch.log(u) - torch.log(1 - u)
        g = torch.sigmoid(logits)
        if self.use_anchor:
            g = torch.where(self.anchor_mask, torch.ones_like(g), g)
        return g

    def gate_penalty(self) -> torch.Tensor:
        """Mean gate value: an L1 edge penalty in [0, 1]; 0 when no gate."""
        if not self.use_gate:
            return torch.zeros((), device=self.logvar.device)
        g = torch.sigmoid(self.gate_logits)
        if self.use_anchor:
            g = g[:, ~self.anchor_mask]
        return g.mean()

    # ---------------------------------------------------------------- forward
    def tokens(self, z: torch.Tensor, y_co2: Optional[torch.Tensor], y_ch4: Optional[torch.Tensor]) -> torch.Tensor:
        """z: (b, tau, d_z, 1) history of ONE variable's global latents; y_*: (b, n_lags, 1).

        Returns (b, n_tokens, D).
        """
        b = z.size(0)
        # (b, tau, D); with d_z > 1 only the first global latent is tokenised (d_z_global = 1 is the tested case)
        parts = [self.value_embed[0](z.reshape(b, self.tau, -1)[:, :, :1])]
        if self.use_co2:
            parts.append(self.value_embed[1](y_co2.reshape(b, -1, 1)))
        if self.use_ch4:
            parts.append(self.value_embed[2](y_ch4.reshape(b, -1, 1)))
        tok = torch.cat(parts, dim=1)
        if tok.size(1) != self.n_tokens:
            raise ValueError(f"got {tok.size(1)} tokens, expected {self.n_tokens} ({self.token_labels})")
        return tok + self.pos_embed

    def forward(self, z, i, k, y_co2=None, y_ch4=None):
        """
        Mean of N(z2_t | tokens) for variable i, latent k.

        Same contract as TransitionModelGlobal.forward.
        """
        out_idx = i * self.d_z + k
        tok = self.tokens(z, y_co2 if self.use_co2 else None, y_ch4 if self.use_ch4 else None)  # (b, T, D)
        keys = self.key(tok)
        vals = self.value(tok)
        logits = keys @ self.query[out_idx] / math.sqrt(self.attention_dim)  # (b, T)
        if self.use_gate:
            g = self.gate(sample=self.training)[out_idx]  # (T,)
            logits = logits + torch.log(g + 1e-6)
        alpha = F.softmax(logits, dim=-1)  # (b, T): the implicit graph row for this output
        ctx = (alpha.unsqueeze(-1) * vals).sum(dim=1)  # (b, D)
        self._attn_batch[out_idx] = alpha.detach()
        if all(a is not None for a in self._attn_batch):
            self.last_attention = torch.stack(self._attn_batch, dim=1)  # (b, n_out, T)
        return self.head[out_idx](ctx)

    # ---------------------------------------------------------------- readout
    @torch.no_grad()
    def attention_matrix(self) -> Optional[torch.Tensor]:
        """Batch-mean attention of the last forward pass: (n_out, n_tokens). None before the first pass."""
        if self.last_attention is None:
            return None
        return self.last_attention.mean(dim=0).cpu()

    @torch.no_grad()
    def gate_matrix(self) -> Optional[torch.Tensor]:
        if not self.use_gate:
            return None
        return self.gate(sample=False).detach().cpu()
