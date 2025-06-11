"""
loss.py

This module implements two actor objectives:

1. GRPOLoss      – the *original* GRPO loss used by Shao et al. (2024)
                   (symmetric clipping + optional KL regularisation).

2. AsymGRPOLoss  – the *final‑paper* objective described in the task prompt:
                   · no KL term,
                   · asymmetric clipping (“Clip‑Higher”),
                   · exact token‑level normalisation
                     1 / (∑_i |o_i|)  across the whole mini‑batch.

Utility functions `approx_kl_divergence` and `masked_mean`
are kept in the same file so they can be reused by experiments.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from replay_buffer import Experience


# ───────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ───────────────────────────────────────────────────────────────────────────────
def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Monte‑Carlo approximation of KL divergence, k3 estimator.
    Reference: http://joschu.net/blog/kl-approx.html
    """
    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    return log_ratio.exp() - log_ratio - 1


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: Union[int, None] = None,
) -> torch.Tensor:
    """
    Mean with an optional boolean mask.

    * If `mask is None`, ordinary `.mean()` is used.
    * If `dim is None`, the mean is taken over **all elements**.
    * The function is central to implementing the paper’s global
      token‑level normalisation:

          1 / (Σ_i |o_i|)  Σ_{i,t} loss_{i,t}

      when called with  `dim=None` and `mask = action_mask`.
    """
    if mask is None:
        return tensor.mean() if dim is None else tensor.mean(dim)

    if dim is None:
        return (tensor * mask).sum() / mask.sum()

    return (tensor * mask).sum(dim) / mask.sum(dim)

class GRPOLoss(nn.Module):
    """
    GRPO loss as in Shao et al. (2024):

        L = - E   [ min(ρ, clip(ρ,1-ε,1+ε)) Â ] + β·KL

    where     ρ = exp(log π_θ − log π_θ_old)
    """

    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    # Returns:  (scalar loss, mean KL)  – signature kept unchanged
    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        old_log_probs = experience.action_log_probs
        log_probs_ref = experience.log_probs_ref
        action_mask = experience.action_mask
        advantages = experience.advantages

        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        token_loss = -torch.min(surr1, surr2) + self.kl_weight * kl

        loss = masked_mean(token_loss, action_mask, dim=-1).mean()
        return loss, kl.mean()


class MistralGRPOLoss(nn.Module):
    r"""
    Implements the *final* GRPO objective supplied in the prompt:

        J(θ) =  E   [ min(ρ, clip(ρ, 1-ε_low, 1+ε_high)) · Â_norm ]
                q,o

    −  KL term has been **removed** (β = 0).
    −  Clipping is asymmetric (ε_high > ε_low) → promotes exploration.
    −  Loss is normalised across **all tokens** in the mini‑batch,
       matching the factor 1 / (Σ_i |o_i|).

    The two‑stage advantage normalisation is performed in `train.py`
    (baseline subtraction per group, then mean‑0/std‑1 per mini‑batch).
    """

    def __init__(self, eps_low: float = 0.20, eps_high: float = 0.28) -> None:
        super().__init__()
        assert eps_high > eps_low, "Need ε_high > ε_low for Clip‑Higher."
        self.eps_low = eps_low
        self.eps_high = eps_high

    # Returns: scalar loss only (no KL)
    def forward(
        self,
        log_probs: torch.Tensor,      # log π_θ (current policy)
        experience: Experience,       # batch sampled under π_θ_old
    ) -> torch.Tensor:

        old_log_probs = experience.action_log_probs       # log π_θ_old
        action_mask = experience.action_mask
        advantages = experience.advantages                # already normed

        # Importance‑sampling ratio ρ
        ratio = (log_probs - old_log_probs).exp()

        # asymmetric clipping
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.eps_low, 1 + self.eps_high) * advantages
        token_loss = -torch.min(surr1, surr2)

        # Global token average (implements Σ_{i,t} / Σ_i |o_i|)
        loss = masked_mean(token_loss, action_mask, dim=None)
        return loss