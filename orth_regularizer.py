# orth_regularizer.py
import torch
import torch.nn.functional as F

def orth_loss(protos: torch.Tensor, *, offdiag_only: bool = True, normalize: bool = True) -> torch.Tensor:
    assert protos.ndim == 2, f"protos shape must be [K,D], got {tuple(protos.shape)}"
    X = F.normalize(protos, dim=-1) if normalize else protos  
    G = X @ X.t()  # [K,K]

    if offdiag_only:
        K = G.size(0)
        off = G - torch.eye(K, device=G.device, dtype=G.dtype)
        loss = (off ** 2).sum() / max(1, K * (K - 1))  
    else:
        loss = (G ** 2).mean()

    return loss
