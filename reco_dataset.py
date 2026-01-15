# reco_dataset.py
import os, csv, numpy as np, torch
import pickle
from torch.utils.data import Dataset

# --------- Read List / Label Tools ---------
def _basename(x: str) -> str:
    x = x.strip()
    if not x: return x
    x = os.path.basename(x)
    return x

def read_slide_list(path: str):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sid = _basename(line)
            if sid:
                ids.append(sid)
    return ids

# [NEW] C17 Specific Label Loading Tool
def load_c17_labels(txt_path: str, npy_path: str):
    """
    Read txt (filenames) and npy (labels), mapping them via index into a dict
    """
    names = read_slide_list(txt_path)
    labels = np.load(npy_path) 
    
    if len(names) != len(labels):
        print(f"[Warning] txt count ({len(names)}) != npy count ({len(labels)})")
        min_len = min(len(names), len(labels))
        names = names[:min_len]
        labels = labels[:min_len]

    mp = {}
    for name, lab in zip(names, labels):
        mp[name] = int(lab)
    return mp

# [NEW] TCGA (LUAD/LUSC) Specific Reader
def read_tcga_csv(csv_path: str):
    """
    Read TCGA CSV file.
    Assumed format:
    Column 0: Slide ID (e.g. TCGA-18-3406-...)
    Column 1: Label String (TCGA-LUAD or TCGA-LUSC) or int
    
    Returns:
        mp: {slide_id: label_int} (LUAD=0, LUSC=1)
    """
    mp = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2: continue
            sid = row[0].strip()
            label_str = row[1].strip()
            
            # Skip common headers
            if sid.lower() in ["slide_id", "id", "case_id", "filename", "file_name"]: continue
            
            # Map labels: TCGA-LUAD -> 0, TCGA-LUSC -> 1
            # Compatible with numeric labels
            if "LUAD" in label_str.upper():
                mp[sid] = 0
            elif "LUSC" in label_str.upper():
                mp[sid] = 1
            elif label_str == "0":
                mp[sid] = 0
            elif label_str == "1":
                mp[sid] = 1
            else:
                # Fallback for unknown labels (e.g., Normal), assuming only LUAD/LUSC for now
                pass
    return mp

def infer_label_from_name(sid: str) -> int:
    # Legacy logic retained as fallback
    s = sid.lower()
    if "tumor" in s or "cancer" in s or "positive" in s or "pos" in s:
        return 1
    if "normal" in s or "negative" in s or "neg" in s:
        return 0
    return 0

def read_label_csv(csv_path: str):
    # Legacy logic retained
    mp = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            if len(row) == 1: row = row[0].split()
            if len(row) < 2: continue
            sid, lab = row[0].strip(), row[1].strip()
            if sid.lower() in ("slide","wsi","id","name","case"): continue
            l = lab.strip().lower()
            if l in ("1","tumor","positive","pos"): mp[_basename(sid)] = 1
            elif l in ("0","normal","negative","neg"): mp[_basename(sid)] = 0
            else: raise ValueError(f"Unknown label value in CSV: {lab}")
    return mp

# --------- I/O Tools ---------

def _load_bag(root_dir: str, sid: str, feat_dim: int):
    # Legacy logic retained (for generic npy)
    p1 = os.path.join(root_dir, sid, "features.npy")
    p2 = os.path.join(root_dir, f"{sid}.npy")
    if os.path.isfile(p1): arr = np.load(p1)
    elif os.path.isfile(p2): arr = np.load(p2)
    else: raise FileNotFoundError(f"features not found for {sid}")
    if arr.ndim != 2 or arr.shape[1] != feat_dim:
        raise ValueError(f"{sid} feature shape {arr.shape} != (N,{feat_dim})")
    return torch.from_numpy(arr).float()

# [NEW] C17 / TCGA Generic PKL Reader
def _load_c17_pkl(path: str, feat_dim: int):
    """
    Read .pkl file, structure: list[dict], dict key='feature'
    Compatible with C17 and TCGA data formats
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Feature file not found: {path}")
        
    with open(path, 'rb') as f:
        data_list = pickle.load(f) # List[Dict]

    # Extract feature field
    feats = []
    for item in data_list:
        if 'feature' in item:
            feats.append(item['feature'])
        else:
            continue
            
    if len(feats) == 0:
        print(f"[Warning] Empty features in {os.path.basename(path)}")
        return torch.zeros((1, feat_dim), dtype=torch.float32)

    arr = np.stack(feats) # [N, feat_dim]
    
    if arr.ndim != 2 or arr.shape[1] != feat_dim:
        # Simple check for dimension mismatch (e.g. moco is 2048, ctrans is 768)
        raise ValueError(f"Feature shape mismatch in {path}: {arr.shape} != (N, {feat_dim})")
        
    return torch.from_numpy(arr).float()

# --------- Single Scale Dataset ---------
class SlideBagDataset(Dataset):
    def __init__(self, root_dir: str, slide_ids, label_map=None, feat_dim: int = 768):
        self.root_dir = root_dir
        self.slide_ids = list(slide_ids)
        self.label_map = label_map or {}
        self.feat_dim = feat_dim
    def __len__(self): return len(self.slide_ids)
    def __getitem__(self, idx):
        sid = self.slide_ids[idx]
        xb = _load_bag(self.root_dir, sid, self.feat_dim)
        y = self.label_map.get(sid, infer_label_from_name(sid))
        y = torch.tensor([y], dtype=torch.long)
        return xb, y, sid

# [New] C16 Specific Multi-Scale Dataset
class C16MultiScaleDataset(Dataset):
    def __init__(self, root_dir: str, slide_ids, label_map=None,
                 feat_dim_high: int = 768, feat_dim_low: int = 768):
        self.ids = list(slide_ids)
        self.root = root_dir
        self.label_map = label_map or {}
        self.fdh = feat_dim_high
        self.fdl = feat_dim_low
        self.path_index_high = {}
        self.path_index_low = {}
        self._build_index()

    def _build_index(self):
        sub_dirs = [
            os.path.join(self.root, "train", "tumor"),
            os.path.join(self.root, "train", "normal"),
            os.path.join(self.root, "test")
        ]
        print(f"Scanning C16 directories for features...")
        for d in sub_dirs:
            if not os.path.exists(d): continue
            for fname in os.listdir(d):
                if not fname.endswith(".pkl"): continue
                full_path = os.path.join(d, fname)
                if "_high_feature_ctrans.pkl" in fname:
                    sid = fname.replace("_high_feature_ctrans.pkl", "")
                    self.path_index_high[sid] = full_path
                elif "_low_feature_ctrans.pkl" in fname:
                    sid = fname.replace("_low_feature_ctrans.pkl", "")
                    self.path_index_low[sid] = full_path

    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        sid = self.ids[idx]
        path_h = self.path_index_high.get(sid)
        path_l = self.path_index_low.get(sid)
        if not path_h or not path_l:
            raise FileNotFoundError(f"Cannot find high/low features for {sid} in {self.root}")

        xh = _load_c17_pkl(path_h, self.fdh)
        xl = _load_c17_pkl(path_l, self.fdl)
        
        if sid in self.label_map:
            y_val = self.label_map[sid]
        else:
            y_val = infer_label_from_name(sid)
            
        y = torch.tensor([y_val], dtype=torch.long)
        return (xh, xl), y, sid

# --------- C17 Specific Multi-Scale Dataset ---------
class Camelyon17MultiScaleDataset(Dataset):
    def __init__(self, root_dir_high: str, root_dir_low: str, 
                 slide_ids, label_map=None,
                 feat_dim_high: int = 768, feat_dim_low: int = 768):
        self.ids = list(slide_ids)
        self.root_h = root_dir_high
        self.root_l = root_dir_low
        self.label_map = label_map or {}
        self.fdh = feat_dim_high
        self.fdl = feat_dim_low
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        sid = self.ids[idx]
        name_h = f"{sid}_high_feature_ctrans.pkl"
        path_h = os.path.join(self.root_h, name_h)
        
        name_l = f"{sid}_low_feature_ctrans.pkl"
        path_l = os.path.join(self.root_l, name_l)
        
        xh = _load_c17_pkl(path_h, self.fdh)
        xl = _load_c17_pkl(path_l, self.fdl)
        
        y_val = self.label_map.get(sid, 0)
        y = torch.tensor([y_val], dtype=torch.long)
        
        return (xh, xl), y, sid

# --------- [NEW] TCGA (LUAD/LUSC) Specific Multi-Scale Dataset ---------
class TCGAMultiScaleDataset(Dataset):
    def __init__(self, luad_dir: str, lusc_dir: str, 
                 slide_ids: list, label_map: dict,
                 feat_dim_high: int = 768, feat_dim_low: int = 768):
        """
        Automatically find files in luad_dir or lusc_dir based on Label (0=LUAD, 1=LUSC)
        """
        self.ids = list(slide_ids)
        self.label_map = label_map
        self.luad_dir = luad_dir
        self.lusc_dir = lusc_dir
        self.fdh = feat_dim_high
        self.fdl = feat_dim_low
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        sid = self.ids[idx]
        y_val = self.label_map.get(sid)
        
        if y_val is None:
            # If no label, raise error or print warning
            raise ValueError(f"Label not found for {sid} in TCGA map")

        # 0 -> LUAD, 1 -> LUSC
        root_dir = self.luad_dir if y_val == 0 else self.lusc_dir
        
        # Construct filename
        name_h = f"{sid}_high_feature_ctrans.pkl"
        name_l = f"{sid}_low_feature_ctrans.pkl"
        
        path_h = os.path.join(root_dir, name_h)
        path_l = os.path.join(root_dir, name_l)
        
        # Reuse _load_c17_pkl
        xh = _load_c17_pkl(path_h, self.fdh)
        xl = _load_c17_pkl(path_l, self.fdl)
        
        y = torch.tensor([y_val], dtype=torch.long)
        
        return (xh, xl), y, sid

# --------- Collate Functions ---------
def collate_bag(batch):
    assert len(batch) == 1, "Recommended batch_size=1 for MIL"
    xb, y, sid = batch[0]
    xb = xb.unsqueeze(0)
    return xb, y, sid

def collate_bag_pair(batch):
    assert len(batch) == 1, "Recommended batch_size=1 for MIL"
    (xh, xl), y, sid = batch[0]
    xh = xh.unsqueeze(0)
    xl = xl.unsqueeze(0)
    return (xh, xl), y, sid
