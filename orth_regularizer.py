# orth_regularizer.py
import torch
import torch.nn.functional as F

def orth_loss(protos: torch.Tensor, *, offdiag_only: bool = True, normalize: bool = True) -> torch.Tensor:
    """
    protos: [K, D]
    余弦-Gram + 仅非对角 + 规模归一化，量级稳定、可对比。

    返回值：标量（未乘 lambda），典型量级 ~ 0~几十（取决于 K、D 与训练阶段）
    """
    assert protos.ndim == 2, f"protos shape must be [K,D], got {tuple(protos.shape)}"
    X = F.normalize(protos, dim=-1) if normalize else protos  # 先行向量归一化
    G = X @ X.t()  # [K,K]

    if offdiag_only:
        K = G.size(0)
        # 仅惩罚非对角
        off = G - torch.eye(K, device=G.device, dtype=G.dtype)
        loss = (off ** 2).sum() / max(1, K * (K - 1))  # 规模归一化
    else:
        # 全 Gram 惩罚（包含对角），再做元素数归一化
        loss = (G ** 2).mean()

    return loss