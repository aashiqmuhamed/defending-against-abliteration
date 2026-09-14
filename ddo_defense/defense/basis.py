"""Candidate decoy directions: an orthonormal basis around the refusal direction.

:func:`build_decoy_basis` returns a ``[d_model, k]`` orthonormal matrix whose
first column is the normalised refusal direction and whose remaining ``k-1``
columns are random directions orthogonal to it.  That is the initialisation the
method uses: sample from a normal distribution, project onto the orthogonal
complement of the refusal direction, orthonormalise.

:mod:`ddo_defense.defense.surgery` writes a chosen column into the MLP weights,
and :mod:`ddo_defense.defense.optimizer` takes column 1 as the warm start for the
direction it then optimises.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _normalize(v: Tensor, eps: float = 1e-8) -> Tensor:
    return v / (v.norm() + eps)


def build_decoy_basis(
    r: Tensor,
    k: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build an orthonormal basis ``U`` in ``R^{d_model x k}``.

    ``U[:, 0]`` is the normalised refusal direction ``r``; the remaining ``k-1``
    columns are random and orthogonal to it and to each other.

    Parameters
    ----------
    r : Tensor [d_model]
        Refusal direction for one layer.
    k : int
        Total basis size: one refusal direction plus ``k-1`` decoy candidates.
    seed : int
        Seed for the random columns, so a layer's basis is reproducible.
    dtype : torch.dtype
        Working precision.

    Returns
    -------
    U : Tensor [d_model, k]
        Orthonormal, with ``U[:, 0]`` along ``r``.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if r.ndim != 1:
        raise ValueError(f"r must be 1-D [d_model], got shape {tuple(r.shape)}")
    if k > r.shape[0]:
        # QR would silently return d_model columns, so the documented shape
        # would not hold and a caller slicing U[:, 1:] would get fewer decoys
        # than it asked for.
        raise ValueError(
            f"k={k} exceeds d_model={r.shape[0]}; there cannot be that many "
            f"mutually orthogonal directions."
        )

    device = r.device
    r_hat = _normalize(r.to(device=device, dtype=dtype))
    aux = _random_basis(r_hat, k - 1, seed=seed, device=device, dtype=dtype)

    U = torch.cat([r_hat[:, None], aux], dim=1)

    # The columns are already orthonormal by construction, so this is a guard
    # against a near-degenerate r leaving a measurable residual, not a routine
    # step.
    eye = torch.eye(k, device=device, dtype=dtype)
    if (U.T @ U - eye).abs().max().item() > 1e-4:
        Q, _ = torch.linalg.qr(U)
        U = Q[:, :k]

    return U


def _random_basis(
    r_hat: Tensor,
    n: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return *n* random orthonormal vectors orthogonal to *r_hat*."""
    d_model = r_hat.shape[0]
    # Drawn on CPU and moved, so the same seed gives the same basis whether the
    # caller's direction lives on CPU or GPU.
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(d_model, n, generator=g, dtype=dtype).to(device)
    G = G - torch.outer(r_hat, r_hat) @ G
    Q, _ = torch.linalg.qr(G)
    return Q[:, :n]
