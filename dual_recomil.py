# dual_recomil.py
import torch
import torch.nn as nn
from typing import Optional, Literal, Dict, Any

from recomil_model import ReCoMIL


class DualReCoMIL(nn.Module):
    def __init__(self,
                 # ----- high -----
                 dim_in_high: int = 768,
                 embed_dim_high: int = 512,
                 num_protos_high: int = 128,
                 proto_init_path_high: Optional[str] = None,
                 similarity_method_high: str = 'l2',
                 tau_high: float = 1.0,
                 route_topk_high: int = 0,
                 use_graph_reducer_high: bool = False,
                 graph_thresh_high: float = 0.2,
                 # ----- low -----
                 dim_in_low: int = 768,
                 embed_dim_low: int = 512,
                 num_protos_low: int = 128,
                 proto_init_path_low: Optional[str] = None,
                 similarity_method_low: str = 'l2',
                 tau_low: float = 1.0,
                 route_topk_low: int = 0,
                 use_graph_reducer_low: bool = False,
                 graph_thresh_low: float = 0.2,
                 # ----- common -----
                 num_classes: int = 2,
                 num_enhancers: int = 3,
                 drop: float = 0.25,
                 hard: bool = False,
                 # ----- fusion (new, but backwards-compatible) -----
                 fusion_mode: Literal['learned', 'confidence', 'fixed'] = 'learned',
                 alpha_init: float = 0.5,
                 freeze_alpha: bool = False):

        super().__init__()

        # high branch
        self.high = ReCoMIL(dim_in_high, embed_dim_high, num_protos_high, num_classes,
                            proto_init_path_high, num_enhancers, drop, hard,
                            similarity_method_high, tau_high, route_topk_high,
                            use_graph_reducer_high, graph_thresh_high)

        # low branch
        self.low = ReCoMIL(dim_in_low, embed_dim_low, num_protos_low, num_classes,
                           proto_init_path_low, num_enhancers, drop, hard,
                           similarity_method_low, tau_low, route_topk_low,
                           use_graph_reducer_low, graph_thresh_low)

        # fusion
        self.fusion_mode: str = fusion_mode
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        if freeze_alpha:
            self.alpha.requires_grad_(False)

    # --------- small helpers ---------
    def set_tau(self, tau_high: Optional[float] = None, tau_low: Optional[float] = None):
        if tau_high is not None:
            self.high.set_tau(tau_high)
        if tau_low is not None:
            self.low.set_tau(tau_low)

    def get_projected_protos(self):
        with torch.no_grad():
            return self.high.get_projected_protos(), self.low.get_projected_protos()

    @staticmethod
    def _confidence_weight(p_h: torch.Tensor, p_l: torch.Tensor) -> torch.Tensor:
        conf_h = p_h.max(dim=1, keepdim=True).values  # [B,1]
        conf_l = p_l.max(dim=1, keepdim=True).values  # [B,1]
        w = conf_h / (conf_h + conf_l + 1e-8)
        return w.clamp(0.0, 1.0)

    # --------- forward ---------
    def forward(self,
                data_high: torch.Tensor,
                data_low: torch.Tensor,
                instance_gate_high: Optional[torch.Tensor] = None,
                instance_gate_low: Optional[torch.Tensor] = None,
                proto_weights_high: Optional[torch.Tensor] = None,
                proto_weights_low: Optional[torch.Tensor] = None) -> Dict[str, Any]:

        oh = self.high(data=data_high,
                       instance_gate=instance_gate_high,
                       proto_weights=proto_weights_high)
        ol = self.low(data=data_low,
                      instance_gate=instance_gate_low,
                      proto_weights=proto_weights_low)

        logit_h = oh["logits"]
        logit_l = ol["logits"]

        device = logit_h.device
        logit_l = logit_l.to(device)
        alpha = self.alpha.to(device)

        if self.fusion_mode == 'learned':
            wh = torch.sigmoid(alpha).view(1, 1).expand(logit_h.size(0), 1)
        elif self.fusion_mode == 'confidence':
            p_h = torch.softmax(logit_h, dim=1)
            p_l = torch.softmax(logit_l, dim=1)
            wh = self._confidence_weight(p_h, p_l).to(device)  # [B,1]
        else:  # 'fixed'
            with torch.no_grad():
                wh = torch.tensor(float(alpha), device=device).sigmoid().view(1, 1).expand(logit_h.size(0), 1)

        logits = wh * logit_h + (1.0 - wh) * logit_l
        Y_prob = torch.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        return {
            "logits": logits,
            "Y_prob": Y_prob,
            "Y_hat": Y_hat,
            "logits_high": logit_h,
            "logits_low": logit_l,
            "w_high": wh,
            "A_high": oh.get("A"), 
            "A_low": ol.get("A")
        }
