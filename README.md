# ReCo-MIL: Rare-enhanced Contextual Multiple Instance Learning

[![Framework Architecture](https://img.shields.io/badge/Framework-ReCo--MIL-blue)](https://github.com/zsszc/ReCo-MIL)

**ReCo-MIL** is a Multiple Instance Learning (MIL) framework designed for Whole Slide Image (WSI) classification. It introduces a **Prototype-based Context Recalibration** mechanism to capture global contextual information and a **Rarity Gate** to handle rare but diagnostic instances (e.g., small tumor regions).

The framework supports **Dual-Scale (High/Low Magnification)** integration to mimic the pathologist's workflow of zooming in and out.

<div align="center">
  <img src="./assets/完整流程图.png" alt="ReCo-MIL Framework Architecture" width="800"/>
  <br>
  <em>Figure 1: Overview of the ReCo-MIL Framework Architecture.</em>
</div>

## 🌟 Key Features

* **Context-Aware Recalibration**: Uses learnable prototypes to recalibrate instance features based on their semantic context.
* **Dual-Scale Architecture**: Simultaneously processes High-Mag (e.g., 20x) and Low-Mag (e.g., 5x/10x) features with a fusion mechanism (Learned/Confidence-based).
* **Rarity Gate**: A specialized module that amplifies the weights of "outlier" instances (features distant from prototypes) to prevent rare classes from being dominated by background noise.
* **Orthogonal Regularization**: Enforces diversity among prototypes to cover the feature space effectively.
* **Graph/MLP Reducers**: Options to dynamically reduce the prototype search space.

## 🛠️ Requirements

**Dependencies:**
* Python 3.8+
* PyTorch 2.0+ (Recommended for AMP support)
* WandB (Weights & Biases for logging)
* Scikit-learn
* Numpy

**Installation:**
```bash
pip install -r requirements.txt

```

## 📂 Project Structure

```text
.
├── train_recomil_v3.py         # Main training script (Single/Dual scale)
├── recomil_model.py            # Core ReCoMIL model definition
├── dual_recomil.py             # Dual-branch wrapper for multi-scale fusion
├── minibatch_kmeans_recomil.py # Prototype initialization script
├── reco_dataset.py             # Dataset loaders (C16, C17, TCGA, Legacy)
├── rarity_gate.py              # Rarity gating logic
├── rare_weight.py              # Rare prototype weighting logic
├── orth_regularizer.py         # Orthogonal loss for prototypes
└── requirements.txt

```

## 🚀 Workflow

### 1. Data Preparation

The code expects WSI features to be pre-extracted (e.g., using CTransPath or ResNet) and saved as `.pkl` or `.npy` files.

* **Format**: List of dictionaries or Arrays of shape `[N, Feature_Dim]`.
* **Naming**: Files should follow `{slide_id}_high_feature_ctrans.pkl` (or similar) as defined in `reco_dataset.py`.

### 2. Prototype Initialization

Before training, run K-Means on your training set to initialize the prototypes. This stabilizes convergence.

```bash
python minibatch_kmeans_recomil.py \
  --dataset c17 \
  --root_dir /path/to/features \
  --ids_file /path/to/train_list.txt \
  --save_protos ./init_protos_high.pt \
  --scale high \
  --k 128

```

*(Repeat for 'low' scale if using Dual-Scale mode)*

### 3. Training

#### Dual-Scale Training (Recommended)

Train the model using both High and Low magnification features.

```bash
python train_recomil_v3.py \
  --wandb_project "ReCo-MIL-Project" \
  --root_dir_high /path/to/c16_data/ \
  --root_dir_low /path/to/c16_data/ \
  --train_list ./splits/train_list.txt \
  --test_csv ./splits/test_labels.csv \
  --proto_init_path_high ./init_protos_high.pt \
  --proto_init_path_low ./init_protos_low.pt \
  --feat_dim_high 768 \
  --feat_dim_low 768 \
  --num_protos_high 128 \
  --num_protos_low 128 \
  --save_dir ./checkpoints \
  --batch_size 1 \
  --lr 1e-4 \
  --use_graph_reducer_high \
  --outlier_top_p_high 0.015 \
  --outlier_alpha_high 0.2

```

#### Single-Scale Training

To train only on one scale, simply omit the `--root_dir_low` and `--proto_init_path_low` arguments.

```bash
python train_recomil_v3.py \
  --wandb_project "ReCo-MIL-Single" \
  --root_dir_high /path/to/features/ \
  --train_list ./train.txt \
  --test_csv ./test.csv \
  --proto_init_path_high ./init_protos.pt \
  ...

```

## ⚙️ Key Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--num_protos_high` | Number of prototypes (clusters) for high mag. | 128 |
| `--similarity_method_high` | Distance metric for proto-instance sim (`l2`, `cosine`). | `l2` |
| `--outlier_top_p_high` | Percentage of instances considered "rare" outliers. | 0.015 |
| `--outlier_alpha_high` | Amplification factor for rare instances. | 0.2 |
| `--lambda_ortho_high` | Weight for Orthogonal Regularization loss. | 0.02 |
| `--save_policy` | Metric to determine best model (`val_auc`, `joint`, `train_loss`). | `joint` |

## 📊 Logging & Visualization

The training script automatically logs metrics to **Weights & Biases (WandB)**:

* Training Loss (CE, Orthogonal, Total)
* Validation/Test AUC and Accuracy
* Confusion Matrices
* ROC Curves

Ensure you are logged in to WandB before running:

```bash
wandb login

```

## 📝 Citation

This paper has been submitted to **ICPR 2026**.
The full citation will be available here after acceptance.

If you find this code useful for your research, please star this repository to stay updated!

```
