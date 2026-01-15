# recomil_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ---------- Small blocks ----------

class GatedAttention(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=1, drop=0.25):
        super().__init__()
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Dropout(drop) if drop > 0 else nn.Identity()
        )
        self.attention_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Sigmoid(),
            nn.Dropout(drop) if drop > 0 else nn.Identity()
        )
        self.attention_scorer = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):  # [M,C]
        t = self.feature_transform(features)
        g = self.attention_gate(features)
        logits = self.attention_scorer(t * g)  # [M,1]
        return logits, features


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, out_features)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class ProtoReducer(Mlp):
    """MLP-based reducer along prototype dimension"""
    def forward(self, protos):  # [K,C]
        return super().forward(protos.transpose(0, 1)).transpose(0, 1)


class GraphProtoReducer(nn.Module):
    def __init__(self, in_K: int, out_K: int, feat_dim: int, thresh: float = 0.2, drop: float = 0.0):
        super().__init__()
        self.in_K, self.out_K, self.feat_dim = in_K, out_K, feat_dim
        self.thresh = float(thresh)
        self.lin_reduce = nn.Linear(in_K, out_K, bias=False)
        nn.init.kaiming_uniform_(self.lin_reduce.weight, a=5**0.5)
        self.post = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(),
                                  nn.Dropout(drop) if drop>0 else nn.Identity())

    @torch.no_grad()
    def _adjacency(self, X: torch.Tensor):
        Xn = F.normalize(X, dim=-1)
        A = torch.clamp(Xn @ Xn.t(), 0.0, 1.0)
        if self.thresh > 0:
            A = torch.where(A >= self.thresh, A, torch.zeros_like(A))
        I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        A = A + I
        d = torch.clamp(A.sum(-1), min=1e-6)
        D_inv_sqrt = torch.diag(torch.pow(d, -0.5))
        return D_inv_sqrt @ A @ D_inv_sqrt

    def forward(self, protos):
        A_hat = self._adjacency(protos.detach())
        msg = A_hat @ protos
        msg = self.post(msg)
        out = self.lin_reduce(msg.transpose(0,1)).transpose(0,1)
        return out


# ---------- ReCoMIL (single branch) ----------

class ReCoMIL(nn.Module):
    def __init__(self,
                 dim_in=768,
                 embed_dim=512,
                 num_protos=128,
                 num_classes=2,
                 proto_init_path: Optional[str]=None,
                 num_enhancers=3,
                 drop=0.25,
                 hard=False,
                 similarity_method='l2',
                 tau: float = 1.0,
                 route_topk: int = 0,
                 use_graph_reducer: bool = False,
                 graph_thresh: float = 0.2):
        super().__init__()
        self.embedding_dim = embed_dim
        self.num_protos = num_protos
        self.hard_assignment = hard
        self.similarity_method = similarity_method
        self.tau = float(tau)
        self.route_topk = int(route_topk)
        self.use_graph_reducer = bool(use_graph_reducer)
        self.graph_thresh = float(graph_thresh)

        # Init Protos
        if proto_init_path:
            # 推理时如果不传path，会随机初始化，但load_state_dict会覆盖它，所以无所谓
            try:
                obj = torch.load(proto_init_path, map_location="cpu")
                if isinstance(obj, torch.Tensor):
                    initial = obj
                elif hasattr(obj, "keys"):
                    arr = obj.get("protos", obj.get("centers", obj.get("cluster_centers")))
                    initial = arr if isinstance(arr, torch.Tensor) else torch.from_numpy(arr)
                else:
                    initial = torch.randn(num_protos, dim_in)
            except:
                initial = torch.randn(num_protos, dim_in)
            
            # 尺寸校验
            if initial.shape[0] != num_protos: 
                initial = torch.randn(num_protos, dim_in)
                
            self.prototypes = nn.Parameter(initial.float(), requires_grad=True)
        else:
            self.prototypes = nn.Parameter(torch.randn(num_protos, dim_in), requires_grad=True)

        self.patch_proj = nn.Sequential(nn.Linear(dim_in, embed_dim), nn.LeakyReLU(inplace=True))
        self.proto_proj = nn.Sequential(nn.Linear(dim_in, embed_dim), nn.LeakyReLU(inplace=True))

        self._dyn_K = [num_protos // (2 ** i) for i in range(num_enhancers + 1)]
        self.enhancers = nn.ModuleList([Mlp(embed_dim, embed_dim, embed_dim, nn.ReLU, drop) for _ in range(num_enhancers)])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_enhancers)])
        self.reducers = nn.ModuleList()
        for i in range(num_enhancers):
            inK, outK = self._dyn_K[i], self._dyn_K[i+1]
            if self.use_graph_reducer:
                self.reducers.append(GraphProtoReducer(inK, outK, embed_dim, thresh=self.graph_thresh, drop=drop))
            else:
                self.reducers.append(ProtoReducer(in_features=inK, hidden_features=inK, out_features=outK, drop=drop))

        self.proc = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                                  nn.Dropout(drop) if drop>0 else nn.Identity())
        self.aggr_norm = nn.LayerNorm(embed_dim)
        self.attn = GatedAttention(embed_dim, embed_dim, num_classes=1, drop=drop)

        self.head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                                  nn.Dropout(drop) if drop>0 else nn.Identity())
        self.cls = nn.Linear(embed_dim, num_classes)
        self.similarity_scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self._warned_w_mismatch = False

    def set_tau(self, tau: float):
        self.tau = float(tau)

    def get_projected_protos(self) -> torch.Tensor:
        with torch.no_grad():
            return self.proto_proj(self.prototypes.detach())

    def _similarity(self, X, Y):
        if self.similarity_method == 'l2':
            return -torch.cdist(X, Y)
        elif self.similarity_method == 'cosine':
            return F.normalize(X, dim=-1) @ F.normalize(Y, dim=-1).t()
        else:
            return X @ Y.t()

    def _straight_through_softmax(self, sim, hard: bool = True, dim: int = -1, topk: int = 0, tau: float = 1.0):
        orig_dtype = sim.dtype
        S = sim.float()
        if topk and topk > 0:
            k = min(topk, S.size(dim))
            _, topi = torch.topk(S, k=k, dim=dim)
            mask = torch.zeros_like(S, dtype=torch.bool).scatter_(dim, topi, True)
            S = S.masked_fill(~mask, float("-inf"))
        denom = max(1e-6, tau * float(self.similarity_scale.item()))
        y_soft32 = F.softmax(S / denom, dim=dim)
        if hard:
            idx = y_soft32.max(dim, keepdim=True)[1]
            y_hard32 = torch.zeros_like(y_soft32).scatter_(dim, idx, 1.0)
            y_out32 = y_hard32 - y_soft32.detach() + y_soft32
        else:
            y_out32 = y_soft32
        return y_out32.to(orig_dtype)

    def _reduce_weights_by_linear(self, w: torch.Tensor, reducer: GraphProtoReducer) -> torch.Tensor:
        W = reducer.lin_reduce.weight.detach()
        A = torch.relu(W)
        row_sum = A.sum(dim=1, keepdim=True).clamp_min(1e-6)
        M = A / row_sum
        w = w.view(-1)
        w_new = M @ w
        return w_new

    def _reduce_weights_by_interp(self, w: torch.Tensor, outK: int) -> torch.Tensor:
        w = w.view(1, 1, -1)
        w_new = F.interpolate(w, size=outK, mode="linear", align_corners=False).view(-1)
        return w_new

    def _maybe_align_weights(self, C: torch.Tensor, w: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if w is None: return None
        Kc = C.size(0)
        w = w.view(-1).to(C.device, dtype=C.dtype)
        if w.numel() == Kc: return w
        if not self._warned_w_mismatch:
            print(f"[ReCo] proto_weights size {w.numel()} != current K {Kc}; applying fallback.")
            self._warned_w_mismatch = True
        if w.numel() > Kc: return w[:Kc]
        return self._reduce_weights_by_interp(w, Kc)

    def _context(self, P, C, proto_weights=None):
        sim = self._similarity(P, C)
        A = self._straight_through_softmax(sim, hard=self.hard_assignment, dim=1,
                                           topk=self.route_topk, tau=self.tau)
        if proto_weights is not None:
            w = self._maybe_align_weights(C, proto_weights)
            A = A * w.unsqueeze(0)
            A = A / (A.sum(dim=1, keepdim=True) + 1e-6)
        ctx = A @ C
        return ctx

    def forward(self,
                data: torch.Tensor,
                instance_gate: Optional[torch.Tensor]=None,
                proto_weights: Optional[torch.Tensor]=None):
        X = data.float()
        if X.ndim == 3: X = X.squeeze(0)
        if instance_gate is not None:
            g = instance_gate.float()
            if g.ndim == 2: g = g.squeeze(0)
            if g.numel() == X.size(0):
                X = X * g.unsqueeze(-1)

        P = self.patch_proj(X)
        C = self.proto_proj(self.prototypes)
        cur_w = None if proto_weights is None else proto_weights.view(-1).to(C.device, dtype=C.dtype)

        for i, enh in enumerate(self.enhancers):
            ctx = self._context(P, C, proto_weights=cur_w)
            P = self.norms[i](P + ctx)
            P = P + enh(P)
            reducer = self.reducers[i]
            Kin = C.size(0)
            C = reducer(C)
            Kout = C.size(0)
            if cur_w is not None and Kout != Kin:
                if isinstance(reducer, GraphProtoReducer):
                    cur_w = self._reduce_weights_by_linear(cur_w, reducer)
                else:
                    cur_w = self._reduce_weights_by_interp(cur_w, Kout)

        Z = torch.cat([P, C], dim=0)
        Z = self.proc(Z)
        Z = self.aggr_norm(Z)

        attn_scores, _ = self.attn(Z)
        attn_w = F.softmax(attn_scores.transpose(0,1), dim=1) # [1, N+K]
        rep = attn_w @ Z

        feat = self.head(rep).squeeze(0)
        logits = self.cls(feat).unsqueeze(0)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        # --- KEY CHANGE for Heatmap ---
        # 截取前 N 个作为 Patch 的注意力
        num_patches = P.size(0)
        patch_attn = attn_w[:, :num_patches] # [1, N]

        return {
            "logits": logits, 
            "Y_prob": Y_prob, 
            "Y_hat": Y_hat, 
            "proj_protos": C,
            "A": patch_attn # <--- Return patch attention
        }