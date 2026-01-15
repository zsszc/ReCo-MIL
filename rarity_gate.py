# rarity_gate.py
import torch
import torch.nn.functional as F
from typing import Optional

def _sim(feat, protos, method='l2'):
    if method == 'l2':
        return -torch.cdist(feat, protos)  # [N,K]
    elif method == 'cosine':
        return F.normalize(feat, dim=-1) @ F.normalize(protos, dim=-1).t()
    else:
        return feat @ protos.t()

@torch.no_grad()
def rarity_gate(features: torch.Tensor,
                prototypes: torch.Tensor,
                top_p: float = 0.10,
                alpha: float = 0.5,
                method: str = 'l2',
                in_embed: bool = False) -> torch.Tensor:
    """
    返回 [N] 的 gate，离群/稀有实例*放大* (1+alpha)。
    默认在 raw 空间做最近原型距离；若 in_embed=True，表示传入的 feat/proto 已在同一投影空间。
    """
    if top_p <= 0 or alpha <= 0:
        return torch.ones(features.size(0), device=features.device, dtype=features.dtype)

    N = features.size(0)
    S = _sim(features, prototypes, method=method)  # [N,K]
    # 以“最近原型的相似度”作为代表值（低者更离群）
    best = S.max(dim=1).values  # [N]
    # 选最不相似的 top_p（即 best 最小的那部分）
    k = max(1, int(N * top_p))
    thr = torch.kthvalue(best, k).values
    mask = best <= thr
    gate = torch.ones_like(best)
    gate[mask] = 1.0 + alpha
    return gate