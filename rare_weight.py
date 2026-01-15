# rare_weight.py
import os
import numpy as np
import torch

def load_rare_weights(freq_path: str, beta: float, device: torch.device, K_expect: int = None):
    if beta <= 0 or not freq_path or not os.path.isfile(freq_path):
        return None
    freq = np.load(freq_path)  # [K]
    if K_expect is not None and len(freq) != K_expect:
        raise ValueError(f"freq len {len(freq)} != expected K {K_expect}")
    eps = 1.0
    w = (freq.astype(np.float32) + eps) ** (-beta)
    w = w / (w.mean() + 1e-6)
    return torch.from_numpy(w).to(device)
