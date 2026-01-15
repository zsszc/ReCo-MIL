# train_recomil_v3.py
#!/usr/bin/env python3
import os, argparse, time, math, random
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from recomil_model import ReCoMIL
from dual_recomil import DualReCoMIL
from rarity_gate import rarity_gate
from orth_regularizer import orth_loss
from rare_weight import load_rare_weights

from reco_dataset import (
    read_slide_list,
    read_label_csv,
    infer_label_from_name,
    SlideBagDataset,
    C16MultiScaleDataset, 
    collate_bag,
    collate_bag_pair,
)

try:
    import wandb
except Exception as e:
    raise RuntimeError("Please `pip install wandb` and login first.") from e

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

# ---------------- Utils ----------------
def set_seed(seed: int = 2025):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, device, num_classes=2, split_name="val", multi_scale=False):
    model.eval()
    ys, ps, hats = [], [], []
    total, correct = 0, 0

    if loader is None:
        return {"acc": float("nan"), "auc": float("nan")}

    for batch in loader:
        if multi_scale:
            (xh, xl), yb, sid = batch
            xh = xh.to(device, non_blocking=True)
            xl = xl.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).view(-1)
            out = model(data_high=xh, data_low=xl)
        else:
            xb, yb, sid = batch
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).view(-1)
            out = model(data=xb)

        logits = out["logits"]
        prob = torch.softmax(logits, dim=-1)
        hat = prob.argmax(dim=-1)

        ys.append(yb.detach().cpu().numpy())
        ps.append(prob[:, 1].detach().cpu().numpy())
        hats.append(hat.detach().cpu().numpy())
        total += yb.numel()
        correct += (hat.view(-1) == yb.view(-1)).sum().item()

    if len(ys) == 0:
        return {"acc": 0.0, "auc": float("nan")}

    y_true = np.concatenate(ys, axis=0)
    y_score = np.concatenate(ps, axis=0)
    y_hat = np.concatenate(hats, axis=0)
    acc = correct / max(1, total)

    auc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        pass

    try:
        cm_plot = wandb.plot.confusion_matrix(
            probs=None, y_true=y_true.tolist(), preds=y_hat.tolist(),
            class_names=[str(i) for i in range(num_classes)]
        )
        wandb.log({f"{split_name}/confusion_matrix": cm_plot})
        if num_classes == 2 and not math.isnan(auc):
            roc_plot = wandb.plot.roc_curve(
                y_true=y_true.tolist(),
                y_probas=np.vstack([1.0 - y_score, y_score]).T.tolist(),
                labels=[0, 1],
            )
            wandb.log({f"{split_name}/roc_curve": roc_plot})
    except Exception:
        pass

    wandb.log({f"{split_name}/acc": acc, f"{split_name}/auc": auc})
    return {"acc": acc, "auc": auc}


# ---------------- Train ----------------
def main():
    ap = argparse.ArgumentParser("ReCo-MIL V3 training (multi/single-scale, W&B)")
    # data
    ap.add_argument("--root_dir_high", required=True)
    ap.add_argument("--root_dir_low", default=None)
    ap.add_argument("--train_list", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--val_split", type=float, default=0.10)

    ap.add_argument("--no_val", action="store_true",
                    help="不划分验证集：全部 train_list 用于训练；保存策略按 --save_policy")
    ap.add_argument("--save_policy", default="joint",
                    choices=["val_auc", "val_acc", "train_loss", "last","joint"],
                    help="选优/保存策略；no_val 推荐 train_loss 或 last")

    # model (high)
    ap.add_argument("--feat_dim_high", type=int, default=768)
    ap.add_argument("--embed_dim_high", type=int, default=512)
    ap.add_argument("--num_protos_high", type=int, default=128)
    ap.add_argument("--proto_init_path_high", required=True)
    ap.add_argument("--similarity_method_high", type=str, default="l2", choices=["l2", "dot", "cosine"])
    ap.add_argument("--tau_start_high", type=float, default=1.2)
    ap.add_argument("--tau_end_high", type=float, default=0.7)
    ap.add_argument("--tau_epochs_high", type=int, default=35)
    ap.add_argument("--route_topk_high", type=int, default=3)
    ap.add_argument("--use_graph_reducer_high", action="store_true")
    ap.add_argument("--graph_thresh_high", type=float, default=0.15)
    ap.add_argument("--freq_path_high", default=None)        
    ap.add_argument("--rare_beta_high", type=float, default=0.3)
    ap.add_argument("--freeze_protos_epochs_high", type=int, default=12)
    ap.add_argument("--outlier_top_p_high", type=float, default=0.015)
    ap.add_argument("--outlier_alpha_high", type=float, default=0.20)
    ap.add_argument("--gate_in_embed_high", action="store_true")

    # model (low)
    ap.add_argument("--feat_dim_low", type=int, default=768)
    ap.add_argument("--embed_dim_low", type=int, default=512)
    ap.add_argument("--num_protos_low", type=int, default=128)
    ap.add_argument("--proto_init_path_low", default=None)
    ap.add_argument("--similarity_method_low", type=str, default="l2", choices=["l2", "dot", "cosine"])
    ap.add_argument("--tau_start_low", type=float, default=1.2)
    ap.add_argument("--tau_end_low", type=float, default=0.7)
    ap.add_argument("--tau_epochs_low", type=int, default=35)
    ap.add_argument("--route_topk_low", type=int, default=3)
    ap.add_argument("--use_graph_reducer_low", action="store_true")
    ap.add_argument("--graph_thresh_low", type=float, default=0.15)
    ap.add_argument("--freq_path_low", default=None)
    ap.add_argument("--rare_beta_low", type=float, default=0.3)
    ap.add_argument("--freeze_protos_epochs_low", type=int, default=12)
    ap.add_argument("--outlier_top_p_low", type=float, default=0.015)
    ap.add_argument("--outlier_alpha_low", type=float, default=0.2)
    ap.add_argument("--gate_in_embed_low", action="store_true")

    # train
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--warmup_epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--label_smoothing", type=float, default=0.03)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--class_weight", action="store_true")

    # regularizers
    ap.add_argument("--lambda_ortho_high", type=float, default=0.02)
    ap.add_argument("--lambda_ortho_low", type=float, default=0.02)
    ap.add_argument("--ortho_on_projected", action="store_true")

    # io
    ap.add_argument("--save_dir", default="./checkpoints_v3")
    ap.add_argument("--save_name", default="recomil_best_v3.pt")

    # wandb
    ap.add_argument("--wandb_project", required=True)
    ap.add_argument("--wandb_entity", default='898091583-huzhou')
    ap.add_argument("--wandb_group", default="recomil_v3")
    ap.add_argument("--name_prefix", default="ReCoMIL-v3")
    ap.add_argument("--tags", nargs="*", default=["v3"])
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--watch", action="store_true")

    ap.add_argument("--print_every", type=int, default=100,
                    help=">0 则每 N 步打印一次 batch 级别 CE/ortho/total；默认 1")

    args = ap.parse_args()
    set_seed(args.seed)

    multi_scale = args.root_dir_low is not None and args.proto_init_path_low is not None

    run_name = f"{args.name_prefix}-ms{int(multi_scale)}-K{args.num_protos_high}{'' if not multi_scale else f'/K{args.num_protos_low}'}-val{args.val_split}-seed{args.seed}-{datetime.now().strftime('%m%d_%H%M%S')}"
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
               name=run_name, mode=args.wandb_mode, tags=args.tags, config=vars(args))
    wandb.define_metric("epoch")
    for split in ["train", "val", "test"]:
        wandb.define_metric(f"{split}/*", step_metric="epoch")

    # ---------- data ----------
    all_ids = read_slide_list(args.train_list)
    y_all = [infer_label_from_name(s) for s in all_ids]

    if args.no_val:
        tr_ids, va_ids = all_ids, []
    else:
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=args.seed)
            tr_idx, va_idx = next(sss.split(all_ids, y_all))
            tr_ids = [all_ids[i] for i in tr_idx]
            va_ids = [all_ids[i] for i in va_idx]
        except Exception:
            random.Random(args.seed).shuffle(all_ids)
            n_val = max(1, int(len(all_ids) * args.val_split))
            va_ids, tr_ids = all_ids[:n_val], all_ids[n_val:]

    test_map = read_label_csv(args.test_csv)
    test_ids = list(test_map.keys())

    def pos_rate(ids: List[str]):
        ys = [infer_label_from_name(s) for s in ids]
        return float(np.mean(ys)) if len(ys) else 0.0

    wandb.log({
        "data/train_size": len(tr_ids),
        "data/val_size": len(va_ids),
        "data/train_pos_rate": pos_rate(tr_ids),
        "data/val_pos_rate": pos_rate(va_ids) if len(va_ids) else float("nan")
    }, step=0)

    if multi_scale:
        # ----------------- [MODIFIED for C16] -----------------
        ds_tr = C16MultiScaleDataset(
            root_dir=args.root_dir_high, 
            slide_ids=tr_ids, 
            label_map=None,
            feat_dim_high=args.feat_dim_high, 
            feat_dim_low=args.feat_dim_low
        )
        
        ds_te = C16MultiScaleDataset(
            root_dir=args.root_dir_high, 
            slide_ids=test_ids, 
            label_map=test_map,
            feat_dim_high=args.feat_dim_high, 
            feat_dim_low=args.feat_dim_low
        )
        
        if not args.no_val:
            ds_va = C16MultiScaleDataset(
                root_dir=args.root_dir_high, 
                slide_ids=va_ids, 
                label_map=None,
                feat_dim_high=args.feat_dim_high, 
                feat_dim_low=args.feat_dim_low
            )
        # ------------------------------------------------------

        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                           pin_memory=True, collate_fn=collate_bag_pair)
        dl_te = DataLoader(ds_te, batch_size=1, shuffle=False, num_workers=args.workers,
                           pin_memory=True, collate_fn=collate_bag_pair)
        
        if args.no_val:
            dl_va = None
        else:
            dl_va = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=args.workers,
                               pin_memory=True, collate_fn=collate_bag_pair)


    # ---------- model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if multi_scale:
        model = DualReCoMIL(
            # high
            dim_in_high=args.feat_dim_high, embed_dim_high=args.embed_dim_high, num_protos_high=args.num_protos_high,
            proto_init_path_high=args.proto_init_path_high, similarity_method_high=args.similarity_method_high,
            tau_high=args.tau_start_high, route_topk_high=args.route_topk_high,
            use_graph_reducer_high=args.use_graph_reducer_high, graph_thresh_high=args.graph_thresh_high,
            # low
            dim_in_low=args.feat_dim_low, embed_dim_low=args.embed_dim_low, num_protos_low=args.num_protos_low,
            proto_init_path_low=args.proto_init_path_low, similarity_method_low=args.similarity_method_low,
            tau_low=args.tau_start_low, route_topk_low=args.route_topk_low,
            use_graph_reducer_low=args.use_graph_reducer_low, graph_thresh_low=args.graph_thresh_low,
            # common
            num_classes=args.num_classes, num_enhancers=3, drop=0.25, hard=False
        ).to(device)
    else:
        model = ReCoMIL(
            dim_in=args.feat_dim_high, embed_dim=args.embed_dim_high, num_protos=args.num_protos_high,
            num_classes=args.num_classes, proto_init_path=args.proto_init_path_high,
            similarity_method=args.similarity_method_high, tau=args.tau_start_high,
            route_topk=args.route_topk_high, use_graph_reducer=args.use_graph_reducer_high,
            graph_thresh=args.graph_thresh_high
        ).to(device)

    if args.watch:
        wandb.watch(model, log="gradients", log_freq=50, log_graph=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler('cuda', enabled=args.fp16)

    # class weights
    class_w = None
    if args.class_weight:
        pr = pos_rate(tr_ids)
        w0 = 0.5 / max(1e-6, 1.0 - pr)
        w1 = 0.5 / max(1e-6, pr if pr > 0 else 1e-6)
        class_w = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    w_high = load_rare_weights(args.freq_path_high, args.rare_beta_high, device, args.num_protos_high)
    w_low  = load_rare_weights(args.freq_path_low , args.rare_beta_low , device, args.num_protos_low) if multi_scale else None

    # cosine + warmup
    def lr_factor(epoch):
        if epoch <= args.warmup_epochs:
            return epoch / max(1, args.warmup_epochs)
        t = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * t))

    # tau schedule
    def tau_now(ep, start, end, tot):
        if tot <= 0: return start
        t = min(1.0, ep / max(1, tot))
        return start + (end - start) * t

    # freeze prototypes
    def set_proto_trainable(on_high: bool, on_low: bool):
        if multi_scale:
            if hasattr(model.high, "prototypes"):
                model.high.prototypes.requires_grad = on_high
            if hasattr(model.low, "prototypes"):
                model.low.prototypes.requires_grad = on_low
        else:
            if hasattr(model, "prototypes"):
                model.prototypes.requires_grad = on_high

    # ---------- train loop ----------
    os.makedirs(args.save_dir, exist_ok=True)
    best_score = -1e9
    best_path = os.path.join(args.save_dir, args.save_name)
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time()
        # lr
        for pg in opt.param_groups:
            pg["lr"] = args.lr * lr_factor(epoch)

        # tau
        if multi_scale:
            model.set_tau(
                tau_high=tau_now(epoch, args.tau_start_high, args.tau_end_high, args.tau_epochs_high),
                tau_low =tau_now(epoch, args.tau_start_low , args.tau_end_low , args.tau_epochs_low )
            )
        else:
            model.set_tau(tau_now(epoch, args.tau_start_high, args.tau_end_high, args.tau_epochs_high))

        # freeze protos
        if multi_scale:
            on_h = epoch > args.freeze_protos_epochs_high
            on_l = epoch > args.freeze_protos_epochs_low
            set_proto_trainable(on_h, on_l)
        else:
            set_proto_trainable(epoch > args.freeze_protos_epochs_high, False)

        ce_sum, ortho_h_sum, ortho_l_sum, total_sum, num_steps = 0.0, 0.0, 0.0, 0.0, 0

        # one epoch
        for batch in dl_tr:
            if multi_scale:
                (xh, xl), yb, sid = batch
                xh = xh.to(device, non_blocking=True)
                xl = xl.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).view(-1)

                # rarity gate
                if args.outlier_top_p_high > 0 and args.outlier_alpha_high > 0:
                    if args.gate_in_embed_high:
                        with torch.no_grad():
                            ph = model.high.patch_proj(xh.squeeze(0))
                            ch = model.high.proto_proj(model.high.prototypes)
                        gate_h = rarity_gate(ph, ch, args.outlier_top_p_high, args.outlier_alpha_high,
                                             method=args.similarity_method_high, in_embed=True).unsqueeze(0)
                    else:
                        with torch.no_grad():
                            gate_h = rarity_gate(xh.squeeze(0), model.high.prototypes.detach(),
                                                 args.outlier_top_p_high, args.outlier_alpha_high,
                                                 method=args.similarity_method_high, in_embed=False).unsqueeze(0)
                else:
                    gate_h = None

                if args.outlier_top_p_low > 0 and args.outlier_alpha_low > 0:
                    if args.gate_in_embed_low:
                        with torch.no_grad():
                            pl = model.low.patch_proj(xl.squeeze(0))
                            cl = model.low.proto_proj(model.low.prototypes)
                        gate_l = rarity_gate(pl, cl, args.outlier_top_p_low, args.outlier_alpha_low,
                                             method=args.similarity_method_low, in_embed=True).unsqueeze(0)
                    else:
                        with torch.no_grad():
                            gate_l = rarity_gate(xl.squeeze(0), model.low.prototypes.detach(),
                                                 args.outlier_top_p_low, args.outlier_alpha_low,
                                                 method=args.similarity_method_low, in_embed=False).unsqueeze(0)
                else:
                    gate_l = None

                opt.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=args.fp16):
                    out = model(
                        data_high=xh, data_low=xl,
                        instance_gate_high=gate_h, instance_gate_low=gate_l,
                        proto_weights_high=w_high, proto_weights_low=w_low
                    )
                    logits = out["logits"]
                    ce = F.cross_entropy(logits, yb, label_smoothing=args.label_smoothing, weight=class_w)

                    ortho_h_term = torch.tensor(0.0, device=device)
                    ortho_l_term = torch.tensor(0.0, device=device)
                    if args.lambda_ortho_high > 0:
                        ph_proj = model.high.get_projected_protos() if args.ortho_on_projected else model.high.prototypes
                        ortho_h_term = args.lambda_ortho_high * orth_loss(ph_proj)
                    if args.lambda_ortho_low > 0:
                        pl_proj = model.low.get_projected_protos() if args.ortho_on_projected else model.low.prototypes
                        ortho_l_term = args.lambda_ortho_low * orth_loss(pl_proj)

                    loss = ce + ortho_h_term + ortho_l_term

                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update()

                num_steps += 1
                ce_sum      += float(ce.detach().cpu())
                ortho_h_sum += float(ortho_h_term.detach().cpu())
                ortho_l_sum += float(ortho_l_term.detach().cpu())
                total_sum   += float(loss.detach().cpu())

                if args.print_every > 0 and (num_steps % args.print_every == 0):
                    print(f"  step {num_steps:4d} | CE {ce_sum/num_steps:.4f} | "
                          f"ortho_h {ortho_h_sum/num_steps:.4f} | ortho_l {ortho_l_sum/num_steps:.4f} | "
                          f"total {total_sum/num_steps:.4f}")

            else:
                xb, yb, sid = batch
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).view(-1)

                # rarity gate
                if args.outlier_top_p_high > 0 and args.outlier_alpha_high > 0:
                    if args.gate_in_embed_high:
                        with torch.no_grad():
                            p = model.patch_proj(xb.squeeze(0))
                            c = model.proto_proj(model.prototypes)
                        gate = rarity_gate(p, c, args.outlier_top_p_high, args.outlier_alpha_high,
                                           method=args.similarity_method_high, in_embed=True).unsqueeze(0)
                    else:
                        with torch.no_grad():
                            gate = rarity_gate(xb.squeeze(0), model.prototypes.detach(),
                                               args.outlier_top_p_high, args.outlier_alpha_high,
                                               method=args.similarity_method_high, in_embed=False).unsqueeze(0)
                else:
                    gate = None

                opt.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=args.fp16):
                    out = model(data=xb, instance_gate=gate, proto_weights=w_high)
                    logits = out["logits"]
                    ce = F.cross_entropy(logits, yb, label_smoothing=args.label_smoothing, weight=class_w)

                    ortho_h_term = torch.tensor(0.0, device=device)
                    if args.lambda_ortho_high > 0:
                        ph_proj = model.get_projected_protos() if args.ortho_on_projected else model.prototypes
                        ortho_h_term = args.lambda_ortho_high * orth_loss(ph_proj)

                    loss = ce + ortho_h_term

                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update()

                num_steps   += 1
                ce_sum      += float(ce.detach().cpu())
                ortho_h_sum += float(ortho_h_term.detach().cpu())
                total_sum   += float(loss.detach().cpu())

                if args.print_every > 0 and (num_steps % args.print_every == 0):
                    print(f"  step {num_steps:4d} | CE {ce_sum/num_steps:.4f} | "
                          f"ortho_h {ortho_h_sum/num_steps:.4f} | total {total_sum/num_steps:.4f}")

        val_metrics = evaluate(model, dl_va, device, args.num_classes, "val", multi_scale=multi_scale)
        dt = time.time() - t0

        avg_ce  = ce_sum / max(1, num_steps)
        avg_oh  = ortho_h_sum / max(1, num_steps)
        avg_ol  = ortho_l_sum / max(1, num_steps) if multi_scale else 0.0
        avg_tot = total_sum / max(1, num_steps)
        cur_lr  = opt.param_groups[0]["lr"]

        if multi_scale:
            tau_h = model.high.tau; tau_l = model.low.tau
            w_high_alpha = torch.sigmoid(model.alpha).item()
            print(f"Epoch {epoch:03d} | CE {avg_ce:.4f} | ortho_h {avg_oh:.4f} | ortho_l {avg_ol:.4f} | "
                  f"total {avg_tot:.4f} | val_acc {val_metrics['acc']:.4f} | val_auc {val_metrics['auc']:.4f} | "
                  f"lr {cur_lr:.2e} | tau_h {tau_h:.3f} | tau_l {tau_l:.3f} | w_high {w_high_alpha:.2f} | time {dt:.1f}s")
        else:
            tau = model.tau
            print(f"Epoch {epoch:03d} | CE {avg_ce:.4f} | ortho {avg_oh:.4f} | total {avg_tot:.4f} | "
                  f"val_acc {val_metrics['acc']:.4f} | val_auc {val_metrics['auc']:.4f} | "
                  f"lr {cur_lr:.2e} | tau {tau:.3f} | time {dt:.1f}s")

        log_dict = {
            "epoch": epoch,
            "train/ce": avg_ce,
            "train/ortho_high": avg_oh,
            "train/loss_total": avg_tot,
            "train/lr": cur_lr,
            "time/epoch_sec": dt,
        }
        if multi_scale:
            log_dict["train/ortho_low"] = avg_ol
            log_dict["train/tau_high"] = model.high.tau
            log_dict["train/tau_low"]  = model.low.tau
            log_dict["train/w_high"]   = torch.sigmoid(model.alpha).item()
        else:
            log_dict["train/tau"] = model.tau
        wandb.log(log_dict)

        if args.no_val:
            if args.save_policy == "train_loss":
                score = -avg_tot               
            elif args.save_policy == "last":
                score = epoch                  
            elif args.save_policy == "val_acc":
                score = float("-inf")          
            else:  # "val_auc"
                score = float("-inf")         
        else:
            if args.save_policy == "val_acc":
                score = val_metrics["acc"]
            elif args.save_policy == "joint":
                auc_val = val_metrics["auc"] if not math.isnan(val_metrics["auc"]) else 0.0
                acc_val = val_metrics["acc"] 
                score = (auc_val + acc_val) / 2.0
            else:  # 默认 val_auc
                score = val_metrics["auc"] if not math.isnan(val_metrics["auc"]) else val_metrics["acc"]

        improved = score > best_score + args.min_delta
        if improved:
            best_score = score; epochs_no_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "best": best_score, "args": vars(args)}, best_path)
            try:
                art = wandb.Artifact(name=f"{wandb.run.name}-best", type="model")
                art.add_file(best_path); wandb.log_artifact(art)
            except Exception:
                pass
        else:
            epochs_no_improve += 1
            if not (args.no_val and args.save_policy == "last"):
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch} (best={best_score:.4f})")
                    break

    # final test
    state = torch.load(best_path, map_location="cpu")
    model.load_state_dict(state["model_state"], strict=False)
    test_metrics = evaluate(model, dl_te, device, args.num_classes, "test", multi_scale=multi_scale)
    wandb.summary["best_score"] = best_score
    wandb.summary["save_policy"] = args.save_policy
    wandb.summary["test/acc"] = test_metrics["acc"]
    wandb.summary["test/auc"] = test_metrics["auc"]
    wandb.finish()


if __name__ == "__main__":
    main()
