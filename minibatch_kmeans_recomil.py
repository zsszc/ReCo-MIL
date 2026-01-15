# minibatch_kmeans_recomil.py
import os, argparse, random, sys, pickle, csv
import numpy as np

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def load_ids(ids_file):
    """旧有的 txt 读取逻辑 (C17/Legacy用)"""
    if ids_file is None:
        return None
    ids = []
    with open(ids_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            s = os.path.basename(s.rstrip("/"))
            ids.append(s)
    return set(ids)

def read_tcga_csv(csv_path):
    """
    读取 TCGA CSV 文件获取 ID 和 类别 (LUAD/LUSC)
    Returns: dict {slide_id: label_str}
    """
    info = {}
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2: continue
            sid = row[0].strip()
            label = row[1].strip().upper() # 转大写方便判断
            
            # 跳过表头
            if sid.lower() in ["slide_id", "id", "case_id", "filename"]: continue
            
            info[sid] = label
    return info

def load_pkl_features(path, feat_dim):
    """
    读取 pkl 文件: List[Dict{'feature': np.array, ...}]
    """
    if not os.path.isfile(path):
        return None
    
    try:
        with open(path, 'rb') as f:
            data_list = pickle.load(f)
    except Exception as e:
        print(f"[Warn] Failed to load pickle {path}: {e}")
        return None

    feats = []
    for item in data_list:
        if 'feature' in item:
            feats.append(item['feature'])
    
    if not feats:
        return None

    arr = np.stack(feats) # [N, D]
    
    # 简单的维度检查
    if arr.ndim != 2:
        return None
        
    # 如果维度不匹配 (例如 moco是2048但这就要求768)，这里最好做个警告或过滤
    if arr.shape[1] != feat_dim:
        print(f"[Warn] Dim mismatch in {os.path.basename(path)}: {arr.shape[1]} vs {feat_dim}")
        return None
        
    return arr.astype(np.float32)

def iter_slide_arrays(root_dir, allow_ids=None):
    # 旧有的 npy 遍历逻辑
    for name in sorted(os.listdir(root_dir)):
        if allow_ids is not None and name not in allow_ids:
            continue
        p = os.path.join(root_dir, name, "features.npy")
        if os.path.isfile(p):
            yield name, p

def sample_arrays(args):
    """
    根据 args 配置加载数据
    """
    xs = []
    total_used = 0
    
    root_dir = args.root_dir
    feat_dim = args.feat_dim
    sample_ratio = args.sample_ratio
    max_per_slide = args.max_per_slide
    scale = args.scale
    
    # ================= TCGA Mode =================
    if args.dataset == "tcga":
        print(f"[clustering] TCGA Mode. Scale: {scale}")
        print(f"  - LUAD Dir: {args.luad_dir}")
        print(f"  - LUSC Dir: {args.lusc_dir}")
        
        # 1. 读取 CSV
        slide_map = read_tcga_csv(args.train_csv)
        print(f"[clustering] Loaded {len(slide_map)} slides from CSV.")
        
        valid_count = 0
        ids_list = sorted(list(slide_map.keys()))
        
        for sid in ids_list:
            label = slide_map[sid]
            
            # 2. 确定文件夹
            # 逻辑：如果标签含LUAD或为0 -> luad_dir；含LUSC或为1 -> lusc_dir
            target_dir = None
            if "LUAD" in label or label == "0":
                target_dir = args.luad_dir
            elif "LUSC" in label or label == "1":
                target_dir = args.lusc_dir
            else:
                # 无法判断类别的跳过
                continue
                
            # 3. 构造文件名 (ID + _high/low_feature_ctrans.pkl)
            fname = f"{sid}_{scale}_feature_ctrans.pkl"
            full_path = os.path.join(target_dir, fname)
            
            # 4. 加载
            arr = load_pkl_features(full_path, feat_dim)
            if arr is None:
                # print(f"File not found or invalid: {full_path}")
                continue
                
            # 5. 采样
            if sample_ratio < 1.0:
                n = arr.shape[0]
                m = max(1, int(n * sample_ratio))
                idx = np.random.choice(n, m, replace=False)
                arr = arr[idx]
            if max_per_slide is not None and max_per_slide > 0 and arr.shape[0] > max_per_slide:
                idx = np.random.choice(arr.shape[0], max_per_slide, replace=False)
                arr = arr[idx]
                
            xs.append(arr)
            total_used += arr.shape[0]
            valid_count += 1
            
        print(f"[clustering] Successfully loaded {valid_count} / {len(ids_list)} slides.")

    # ================= C17 Mode =================
    elif args.dataset == "c17":
        print(f"[clustering] C17 Mode. Scanning {root_dir}...")
        allow_ids = load_ids(args.ids_file) # C17用txt
        if allow_ids is None:
            raise ValueError("In C17 mode, --ids_file (train.txt) is required.")
            
        # 预扫描
        file_map = {}
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".pkl"):
                    file_map[f] = os.path.join(root, f)
        
        valid_count = 0
        for sid in sorted(list(allow_ids)):
            filename = f"{sid}_{scale}_feature_ctrans.pkl"
            if filename in file_map:
                p = file_map[filename]
            else:
                continue
            
            arr = load_pkl_features(p, feat_dim)
            if arr is None: continue
            
            # 采样
            if sample_ratio < 1.0:
                n = arr.shape[0]
                m = max(1, int(n * sample_ratio))
                idx = np.random.choice(n, m, replace=False)
                arr = arr[idx]
            if max_per_slide is not None and max_per_slide > 0 and arr.shape[0] > max_per_slide:
                idx = np.random.choice(arr.shape[0], max_per_slide, replace=False)
                arr = arr[idx]
                
            xs.append(arr)
            total_used += arr.shape[0]
            valid_count += 1
        print(f"[clustering] Valid slides: {valid_count}")

    # ================= Legacy NPY Mode =================
    else:
        print(f"[clustering] Standard NPY Mode. Reading from {root_dir}")
        allow_ids = load_ids(args.ids_file)
        for _, p in iter_slide_arrays(root_dir, allow_ids):
            arr = np.load(p)
            if arr.ndim != 2 or arr.shape[1] != feat_dim:
                continue
            if sample_ratio < 1.0:
                n = arr.shape[0]
                m = max(1, int(n * sample_ratio))
                idx = np.random.choice(n, m, replace=False)
                arr = arr[idx]
            if max_per_slide is not None and max_per_slide > 0 and arr.shape[0] > max_per_slide:
                idx = np.random.choice(arr.shape[0], max_per_slide, replace=False)
                arr = arr[idx]
            xs.append(arr.astype(np.float32))
            total_used += arr.shape[0]

    if not xs:
        raise RuntimeError("No features found. Check path, dataset mode, or CSV/IDs.")
    
    X = np.vstack(xs)
    print(f"[clustering] patches_used={total_used}, final_stack={X.shape}", flush=True)
    return X

def main():
    ap = argparse.ArgumentParser("MiniBatch-KMeans for ReCo-MIL prototypes")
    
    # 通用参数
    ap.add_argument("--dataset", type=str, default="tcga", choices=["c17", "tcga", "legacy"], 
                    help="Dataset mode: 'c17' (pkl search), 'tcga' (luad/lusc folders), 'legacy' (npy)")
    ap.add_argument("--feat_dim", type=int, default=768)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--sample_ratio", type=float, default=1.0)
    ap.add_argument("--max_per_slide", type=int, default=0)
    ap.add_argument("--save_protos", required=True)
    ap.add_argument("--save_freq", default=None)
    
    # 尺度 (for TCGA & C17)
    ap.add_argument("--scale", type=str, default="high", choices=["high", "low"])
    
    # TCGA 专用参数
    ap.add_argument("--train_csv", default=None, help="Path to TCGA train csv")
    ap.add_argument("--luad_dir", default="/home/Public/WSI/TCGA-LUAD", help="TCGA-LUAD feature dir")
    ap.add_argument("--lusc_dir", default="/home/Public/WSI/TCGA-LUSC", help="TCGA-LUSC feature dir")
    
    # C17 / Legacy 专用参数
    ap.add_argument("--root_dir", default=None, help="Root dir for C17/Legacy")
    ap.add_argument("--ids_file", default=None, help="txt file for C17/Legacy")

    # 兼容旧代码的 flag (可选，如果用 dataset 参数则不需要)
    ap.add_argument("--c17_mode", action="store_true", help="Deprecated, use --dataset c17")

    args = ap.parse_args()
    set_seed(args.seed)

    # 兼容性处理：如果传了 --c17_mode 但没设 dataset，自动设为 c17
    if args.c17_mode and args.dataset == "legacy": 
        args.dataset = "c17"

    # 执行数据加载
    X = sample_arrays(args)

    from sklearn.cluster import MiniBatchKMeans
    print(f"[clustering] Starting MiniBatchKMeans (k={args.k})...")
    km = MiniBatchKMeans(
        n_clusters=args.k, random_state=args.seed,
        batch_size=args.batch_size, max_iter=args.max_iter, verbose=0
    )
    km.fit(X)
    centers = km.cluster_centers_.astype(np.float32)

    labels = km.predict(X)
    freq = np.bincount(labels, minlength=args.k).astype(np.int64)

    import torch
    torch.save(torch.from_numpy(centers), args.save_protos)
    if args.save_freq:
        np.save(args.save_freq, freq)

    print(f"[+] prototypes -> {args.save_protos} shape={centers.shape}")
    if args.save_freq:
        print(f"[+] freq -> {args.save_freq} sum={int(freq.sum())}")

if __name__ == "__main__":
    main()