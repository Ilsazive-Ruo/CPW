import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# =========================
# Basic config
# =========================
DATA_DIR = "CLSData2"
RESULT_DIR = "ML_Results_GroupSplit_Ablation"
os.makedirs(RESULT_DIR, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42
FIXED_THRESHOLD = 0.5

USE_GPU = torch.cuda.is_available()

PATHWAY_POOLING_MODE = "mean"

SPLIT_SCHEMES = [
    "random",
    "pathway_group",
    "drug_group",
]


ABLATION_MODES = [
    "original",
    "mask_molecule",
    "mask_pathway",
    "mask_both",
]

MODEL_NAMES = [
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
    "DNN_MeanPooling",
]

# DNN config
DNN_EPOCHS = 15
DNN_BATCH_SIZE = 2048
DNN_LR = 1e-3
DNN_WEIGHT_DECAY = 1e-4

# Learnable pooling DNN config
LP_EPOCHS = 15
LP_BATCH_SIZE = 2048
LP_LR = 1e-3
LP_WEIGHT_DECAY = 1e-4
LP_DROPOUT = 0.2
FREEZE_SCGPT = True


# =========================
# Metrics
# =========================
def safe_divide(a, b):
    return float(a / b) if b != 0 else np.nan


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "AUROC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "AUPRC": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision_PPV": precision_score(y_true, y_pred, zero_division=0),
        "Recall_Sensitivity_TPR": recall_score(y_true, y_pred, zero_division=0),
        "Specificity_TNR": safe_divide(tn, tn + fp),
        "NPV": safe_divide(tn, tn + fn),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen_Kappa": cohen_kappa_score(y_true, y_pred),
        "Brier_Score": brier_score_loss(y_true, y_prob),
        "Log_Loss": log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Prevalence": float(np.mean(y_true)),
        "Predicted_Positive_Rate": float(np.mean(y_pred)),
    }

    return metrics


def check_group_leakage(groups, train_idx, val_idx, split_name, fold):
    if groups is None:
        return

    train_groups = set(np.asarray(groups)[train_idx])
    val_groups = set(np.asarray(groups)[val_idx])
    overlap = train_groups.intersection(val_groups)

    if len(overlap) > 0:
        raise RuntimeError(
            f"Group leakage detected in split={split_name}, fold={fold}: "
            f"{len(overlap)} overlapping groups."
        )


# =========================
# Data loading
# =========================
def get_first_existing_meta_field(meta, candidate_keys, required=False, field_name="group"):
    for key in candidate_keys:
        if key in meta:
            return np.asarray(meta[key]), key

    if required:
        raise KeyError(
            f"Cannot find {field_name} in metadata. Tried keys: {candidate_keys}"
        )

    return None, None


def load_cls_data():
    fps_path = os.path.join(DATA_DIR, "cls_morgan_fps.npy")
    labels_path = os.path.join(DATA_DIR, "cls_labels.npy")
    meta_path = os.path.join(DATA_DIR, "cls_gene_metadata.pkl")
    scgpt_path = os.path.join(DATA_DIR, "scgpt_matrix.pt")

    print("========== Loading CLSData ==========")

    fps = np.load(fps_path).astype(np.float32)
    y = np.load(labels_path).astype(np.int64)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    gene_ids_list = meta["gene_ids"]
    scgpt_matrix = torch.load(scgpt_path, map_location="cpu")

    if len(fps) != len(y) or len(y) != len(gene_ids_list):
        raise ValueError(
            f"Data length mismatch: fps={len(fps)}, labels={len(y)}, gene_ids={len(gene_ids_list)}"
        )

    pathway_groups, pathway_key = get_first_existing_meta_field(
        meta,
        candidate_keys=[
            "pathway_ids",
            "pathway_id",
            "pathway_names",
            "pathway_name",
            "pathways",
            "pathway",
            "term_ids",
            "term_id",
            "term_descriptions",
            "term_description",
        ],
        required=False,
        field_name="pathway group",
    )

    drug_groups, drug_key = get_first_existing_meta_field(
        meta,
        candidate_keys=[
            "drug_ids",
            "drug_id",
            "drug_names",
            "drug_name",
            "drugs",
            "drug",
            "molecule_ids",
            "molecule_id",
            "molecule_names",
            "molecule_name",
            "smiles",
            "SMILES",
            "compound_ids",
            "compound_id",
            "compound_names",
            "compound_name",
        ],
        required=False,
        field_name="drug group",
    )

    if pathway_groups is not None and len(pathway_groups) != len(y):
        raise ValueError(
            f"Pathway group length mismatch: {len(pathway_groups)} vs labels {len(y)}"
        )

    if drug_groups is not None and len(drug_groups) != len(y):
        raise ValueError(
            f"Drug group length mismatch: {len(drug_groups)} vs labels {len(y)}"
        )

    print(f"Samples: {len(y)}")
    print(f"Low label=0: {(y == 0).sum()}")
    print(f"High label=1: {(y == 1).sum()}")
    print(f"High ratio: {y.mean():.4f}")

    if pathway_groups is not None:
        print(f"Pathway group key: {pathway_key}")
        print(f"Unique pathway groups: {len(np.unique(pathway_groups))}")
    else:
        print("Pathway group key: NOT FOUND; pathway_group split will be skipped.")

    if drug_groups is not None:
        print(f"Drug group key: {drug_key}")
        print(f"Unique drug groups: {len(np.unique(drug_groups))}")
    else:
        print("Drug group key: NOT FOUND; drug_group split will be skipped.")

    print("Available metadata keys:")
    print(sorted(list(meta.keys())))
    print("=====================================")

    return fps, y, gene_ids_list, scgpt_matrix, pathway_groups, drug_groups, meta


# =========================
# Ablation helpers
# =========================
def apply_fps_ablation(fps, ablation_mode):
    if ablation_mode in ["mask_molecule", "mask_both"]:
        return np.zeros_like(fps, dtype=np.float32)
    return fps.astype(np.float32)


def should_mask_pathway(ablation_mode):
    return ablation_mode in ["mask_pathway", "mask_both"]


# =========================
# Fixed feature building for conventional baselines
# =========================
def build_pathway_features(gene_ids_list, scgpt_matrix, mode="mean", mask_pathway=False):

    print(
        f"--> Building fixed pathway features | pooling={mode} | mask_pathway={mask_pathway}"
    )

    if isinstance(scgpt_matrix, torch.Tensor):
        scgpt_np = scgpt_matrix.cpu().numpy().astype(np.float32)
    else:
        scgpt_np = np.asarray(scgpt_matrix, dtype=np.float32)

    embed_dim = scgpt_np.shape[1]

    if mode == "mean":
        out_dim = embed_dim
    elif mode == "mean_max_std":
        out_dim = embed_dim * 3
    else:
        raise ValueError(f"Unknown pathway pooling mode: {mode}")

    if mask_pathway:
        return np.zeros((len(gene_ids_list), out_dim), dtype=np.float32)

    features = []

    for gids in gene_ids_list:
        gids = [int(g) for g in gids if int(g) > 0 and int(g) < scgpt_np.shape[0]]

        if len(gids) == 0:
            embs = np.zeros((1, embed_dim), dtype=np.float32)
        else:
            embs = scgpt_np[gids].astype(np.float32)

        mean_emb = embs.mean(axis=0)

        if mode == "mean":
            feat = mean_emb
        elif mode == "mean_max_std":
            max_emb = embs.max(axis=0)
            std_emb = embs.std(axis=0)
            feat = np.concatenate([mean_emb, max_emb, std_emb], axis=0)

        features.append(feat)

    return np.vstack(features).astype(np.float32)


def build_fixed_baseline_features(fps, gene_ids_list, scgpt_matrix, ablation_mode):

    fps_used = apply_fps_ablation(fps, ablation_mode)
    mask_pathway = should_mask_pathway(ablation_mode)

    pathway_feat = build_pathway_features(
        gene_ids_list,
        scgpt_matrix,
        mode=PATHWAY_POOLING_MODE,
        mask_pathway=mask_pathway,
    )

    if mask_pathway:
        gene_count = np.zeros((len(gene_ids_list), 1), dtype=np.float32)
    else:
        gene_count = np.array(
            [np.log1p(len(gids)) for gids in gene_ids_list],
            dtype=np.float32,
        ).reshape(-1, 1)

    X = np.concatenate([fps_used, pathway_feat, gene_count], axis=1).astype(np.float32)

    print("========== Fixed feature summary ==========")
    print(f"Ablation mode: {ablation_mode}")
    print(f"Morgan FP dim: {fps.shape[1]} | masked={ablation_mode in ['mask_molecule', 'mask_both']}")
    print(f"Pathway feature dim: {pathway_feat.shape[1]} | masked={mask_pathway}")
    print(f"Gene count dim: 1 | masked={mask_pathway}")
    print(f"Final X shape: {X.shape}")
    print("===========================================")

    return X


# =========================
# Fixed-feature DNN baseline
# =========================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DNNClassifier(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_dnn_fold(X_train, y_train, X_val, y_val):
    device = torch.device("cuda" if USE_GPU else "cpu")

    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=DNN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=USE_GPU,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=DNN_BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=USE_GPU,
    )

    model = DNNClassifier(input_dim=X_train.shape[1]).to(device)

    class_counts = np.bincount(y_train, minlength=2)
    if np.any(class_counts == 0):
        raise RuntimeError(f"Training fold has an empty class: {class_counts}")

    class_weights = class_counts.sum() / (2.0 * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DNN_LR,
        weight_decay=DNN_WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=DNN_EPOCHS,
        eta_min=1e-5,
    )

    amp_scaler = torch.amp.GradScaler("cuda", enabled=USE_GPU)

    best_auprc = -np.inf
    best_probs = None

    for epoch in range(DNN_EPOCHS):
        model.train()
        running_loss = 0.0
        total_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_GPU):
                logits = model(xb)
                loss = criterion(logits, yb)

            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

            running_loss += loss.item() * yb.size(0)
            total_seen += yb.size(0)

        scheduler.step()

        model.eval()
        probs = []

        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_GPU):
                    logits = model(xb)

                prob = torch.softmax(logits.float(), dim=1)[:, 1]
                probs.extend(prob.cpu().numpy())

        probs = np.asarray(probs, dtype=np.float32)
        auprc = average_precision_score(y_val, probs) if len(np.unique(y_val)) == 2 else np.nan

        if np.isnan(auprc):
            score = -np.inf
        else:
            score = auprc

        if score > best_auprc:
            best_auprc = score
            best_probs = probs.copy()

        print(
            f"    DNN_MeanPooling Epoch [{epoch + 1}/{DNN_EPOCHS}] | "
            f"Loss={running_loss / total_seen:.4f} | Val AUPRC={auprc:.4f}"
        )

    return best_probs


# =========================
# Learnable pooling DNN
# =========================
class LearnablePoolingDataset(Dataset):
    def __init__(self, fps, labels, gene_ids_list, ablation_mode="original"):
        self.fps = fps.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.gene_ids_list = gene_ids_list
        self.ablation_mode = ablation_mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.ablation_mode in ["mask_molecule", "mask_both"]:
            fp_np = np.zeros_like(self.fps[idx], dtype=np.float32)
        else:
            fp_np = self.fps[idx]

        fp = torch.tensor(fp_np, dtype=torch.float32)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        gids = self.gene_ids_list[idx]
        if self.ablation_mode in ["mask_pathway", "mask_both"]:
            # Keep one padding token so the sequence is valid, and mask gene_count.
            gene_ids = torch.tensor([0], dtype=torch.long)
            gene_count = torch.tensor([0.0], dtype=torch.float32)
        else:
            gene_ids = torch.tensor(gids, dtype=torch.long)
            gene_count = torch.tensor([np.log1p(len(gids))], dtype=torch.float32)

        return fp, gene_ids, gene_count, label


def learnable_pooling_collate_fn(batch):
    fps, gene_ids_list, gene_counts, labels = zip(*batch)

    padded_gene_ids = pad_sequence(
        gene_ids_list,
        batch_first=True,
        padding_value=0,
    )

    lengths = torch.tensor([len(g) for g in gene_ids_list])
    max_len = padded_gene_ids.size(1)
    key_padding_mask = torch.arange(max_len)[None, :] >= lengths[:, None]


    return (
        torch.stack(fps),
        padded_gene_ids,
        key_padding_mask,
        torch.stack(gene_counts),
        torch.stack(labels),
    )

# =========================
# Sklearn baselines
# =========================
def get_xgboost_model(y_train):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("⚠️ xgboost is not installed. Skipping XGBoost.")
        return None

    n_pos = max(1, int((y_train == 1).sum()))
    n_neg = max(1, int((y_train == 0).sum()))
    scale_pos_weight = n_neg / n_pos

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        reg_alpha=0.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return model


def run_sklearn_model(model_name, model, X_train, y_train, X_val):
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X_val)
        prob = 1 / (1 + np.exp(-score))
    else:
        raise ValueError(f"{model_name} does not support probability prediction.")

    return prob.astype(np.float32)


# =========================
# Split helpers
# =========================
def make_splitter(split_scheme, groups=None):
    if split_scheme == "random":
        splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        return splitter, None

    if split_scheme in ["pathway_group", "drug_group"]:
        if groups is None:
            raise ValueError(f"groups cannot be None for {split_scheme}")

        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        return splitter, np.asarray(groups)

    raise ValueError(f"Unknown split_scheme: {split_scheme}")


def iter_splits(split_scheme, X, y, pathway_groups, drug_groups):
    if split_scheme == "random":
        splitter, _ = make_splitter(split_scheme)
        for train_idx, val_idx in splitter.split(X, y):
            yield train_idx, val_idx, None

    elif split_scheme == "pathway_group":
        if pathway_groups is None:
            print("⚠️ pathway_groups not found. Skipping pathway_group split.")
            return
        splitter, groups = make_splitter(split_scheme, pathway_groups)
        for train_idx, val_idx in splitter.split(X, y, groups=groups):
            yield train_idx, val_idx, groups

    elif split_scheme == "drug_group":
        if drug_groups is None:
            print("⚠️ drug_groups not found. Skipping drug_group split.")
            return
        splitter, groups = make_splitter(split_scheme, drug_groups)
        for train_idx, val_idx in splitter.split(X, y, groups=groups):
            yield train_idx, val_idx, groups

    else:
        raise ValueError(f"Unknown split_scheme: {split_scheme}")


# =========================
# Main comparison
# =========================
def run_all_experiments():
    fps, y, gene_ids_list, scgpt_matrix, pathway_groups, drug_groups, meta = load_cls_data()

    fold_metric_rows = []
    oof_metric_rows = []
    oof_prediction_frames = []
    group_assignment_rows = []

    completed_runs = []

    for ablation_mode in ABLATION_MODES:
        print("\n" + "#" * 90)
        print(f"# Ablation mode: {ablation_mode}")
        print("#" * 90)

        X = build_fixed_baseline_features(
            fps=fps,
            gene_ids_list=gene_ids_list,
            scgpt_matrix=scgpt_matrix,
            ablation_mode=ablation_mode,
        )

        for split_scheme in SPLIT_SCHEMES:
            print("\n" + "=" * 90)
            print(f"Split scheme: {split_scheme} | Ablation: {ablation_mode}")
            print("=" * 90)

            active_model_names = []
            oof_probs = {}
            fold_ids = np.zeros(len(y), dtype=np.int64)

            for model_name in MODEL_NAMES:
                oof_probs[model_name] = np.full(len(y), np.nan, dtype=np.float32)

            split_generator = list(iter_splits(split_scheme, X, y, pathway_groups, drug_groups))
            if len(split_generator) == 0:
                continue

            for fold, (train_idx, val_idx, active_groups) in enumerate(split_generator, start=1):
                print(f"\n================ {split_scheme} | {ablation_mode} | Fold {fold}/{N_SPLITS} ================")

                check_group_leakage(active_groups, train_idx, val_idx, split_scheme, fold)

                X_train_raw, X_val_raw = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                fold_ids[val_idx] = fold

                print(f"Train: {len(y_train)} | Val: {len(y_val)}")
                print(f"Train high ratio: {y_train.mean():.4f} | Val high ratio: {y_val.mean():.4f}")

                if active_groups is not None:
                    print(f"Train groups: {len(np.unique(active_groups[train_idx]))}")
                    print(f"Val groups: {len(np.unique(active_groups[val_idx]))}")

                    for g in np.unique(active_groups[val_idx]):
                        group_assignment_rows.append({
                            "Split_Scheme": split_scheme,
                            "Ablation": ablation_mode,
                            "Fold": fold,
                            "Group": g,
                        })

                if len(np.unique(y_train)) < 2:
                    print(f"⚠️ Fold {fold} training labels contain only one class. Skipping this fold.")
                    continue

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
                X_val_scaled = scaler.transform(X_val_raw).astype(np.float32)

                # -------------------------
                # Logistic Regression
                # -------------------------
                if "LogisticRegression" in MODEL_NAMES:
                    print("-> Training Logistic Regression...")
                    lr_model = LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        solver="saga",
                        max_iter=3000,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    )

                    lr_prob = run_sklearn_model(
                        "LogisticRegression",
                        lr_model,
                        X_train_scaled,
                        y_train,
                        X_val_scaled,
                    )

                    oof_probs["LogisticRegression"][val_idx] = lr_prob
                    active_model_names.append("LogisticRegression")

                    metrics = compute_metrics(y_val, lr_prob, FIXED_THRESHOLD)
                    metrics.update({
                        "Split_Scheme": split_scheme,
                        "Ablation": ablation_mode,
                        "Model": "LogisticRegression",
                        "Fold": fold,
                        "Train_N": len(train_idx),
                        "Val_N": len(val_idx),
                    })
                    fold_metric_rows.append(metrics)

                    print(f"   LR AUROC={metrics['AUROC']:.4f} | AUPRC={metrics['AUPRC']:.4f} | MCC={metrics['MCC']:.4f}")

                # -------------------------
                # Random Forest
                # -------------------------
                if "RandomForest" in MODEL_NAMES:
                    print("-> Training Random Forest...")
                    rf_model = RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    )

                    rf_prob = run_sklearn_model(
                        "RandomForest",
                        rf_model,
                        X_train_raw,
                        y_train,
                        X_val_raw,
                    )

                    oof_probs["RandomForest"][val_idx] = rf_prob
                    active_model_names.append("RandomForest")

                    metrics = compute_metrics(y_val, rf_prob, FIXED_THRESHOLD)
                    metrics.update({
                        "Split_Scheme": split_scheme,
                        "Ablation": ablation_mode,
                        "Model": "RandomForest",
                        "Fold": fold,
                        "Train_N": len(train_idx),
                        "Val_N": len(val_idx),
                    })
                    fold_metric_rows.append(metrics)

                    print(f"   RF AUROC={metrics['AUROC']:.4f} | AUPRC={metrics['AUPRC']:.4f} | MCC={metrics['MCC']:.4f}")

                # -------------------------
                # XGBoost
                # -------------------------
                if "XGBoost" in MODEL_NAMES:
                    print("-> Training XGBoost...")
                    xgb_model = get_xgboost_model(y_train)

                    if xgb_model is not None:
                        xgb_prob = run_sklearn_model(
                            "XGBoost",
                            xgb_model,
                            X_train_raw,
                            y_train,
                            X_val_raw,
                        )

                        oof_probs["XGBoost"][val_idx] = xgb_prob
                        active_model_names.append("XGBoost")

                        metrics = compute_metrics(y_val, xgb_prob, FIXED_THRESHOLD)
                        metrics.update({
                            "Split_Scheme": split_scheme,
                            "Ablation": ablation_mode,
                            "Model": "XGBoost",
                            "Fold": fold,
                            "Train_N": len(train_idx),
                            "Val_N": len(val_idx),
                        })
                        fold_metric_rows.append(metrics)

                        print(f"   XGB AUROC={metrics['AUROC']:.4f} | AUPRC={metrics['AUPRC']:.4f} | MCC={metrics['MCC']:.4f}")

                # -------------------------
                # DNN mean pooling
                # -------------------------
                if "DNN_MeanPooling" in MODEL_NAMES:
                    print("-> Training DNN_MeanPooling baseline...")
                    dnn_prob = train_dnn_fold(
                        X_train_scaled,
                        y_train,
                        X_val_scaled,
                        y_val,
                    )

                    oof_probs["DNN_MeanPooling"][val_idx] = dnn_prob
                    active_model_names.append("DNN_MeanPooling")

                    metrics = compute_metrics(y_val, dnn_prob, FIXED_THRESHOLD)
                    metrics.update({
                        "Split_Scheme": split_scheme,
                        "Ablation": ablation_mode,
                        "Model": "DNN_MeanPooling",
                        "Fold": fold,
                        "Train_N": len(train_idx),
                        "Val_N": len(val_idx),
                    })
                    fold_metric_rows.append(metrics)

                    print(f"   DNN_Mean AUROC={metrics['AUROC']:.4f} | AUPRC={metrics['AUPRC']:.4f} | MCC={metrics['MCC']:.4f}")

            # =========================
            # OOF metrics for this split + ablation
            # =========================
            pred_df = pd.DataFrame({
                "Split_Scheme": split_scheme,
                "Ablation": ablation_mode,
                "Fold": fold_ids,
                "True_Label": y,
            })

            if pathway_groups is not None:
                pred_df["Pathway_Group"] = pathway_groups
            if drug_groups is not None:
                pred_df["Drug_Group"] = drug_groups

            for model_name in active_model_names:
                prob = oof_probs[model_name]
                valid_mask = ~np.isnan(prob)

                if valid_mask.sum() == 0:
                    continue

                metrics = compute_metrics(y[valid_mask], prob[valid_mask], FIXED_THRESHOLD)
                metrics.update({
                    "Split_Scheme": split_scheme,
                    "Ablation": ablation_mode,
                    "Model": model_name,
                    "OOF_N": int(valid_mask.sum()),
                })
                oof_metric_rows.append(metrics)

                pred_df[f"{model_name}_Prob_High"] = prob
                pred_df[f"{model_name}_Pred_Label_0p5"] = (prob >= FIXED_THRESHOLD).astype(float)

            oof_prediction_frames.append(pred_df)
            completed_runs.append({
                "Split_Scheme": split_scheme,
                "Ablation": ablation_mode,
                "Completed_Models": active_model_names,
            })

    # =========================
    # Save fold metrics
    # =========================
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    if not fold_metrics_df.empty:
        fold_metrics_df = fold_metrics_df.sort_values(["Split_Scheme", "Ablation", "Model", "Fold"])

    fold_metrics_df.to_csv(
        os.path.join(RESULT_DIR, "group_ablation_fold_metrics.csv"),
        index=False,
    )

    # =========================
    # OOF metrics
    # =========================
    oof_metrics_df = pd.DataFrame(oof_metric_rows)
    if not oof_metrics_df.empty:
        oof_metrics_df = oof_metrics_df.sort_values(["Split_Scheme", "Ablation", "Model"])

    oof_metrics_df.to_csv(
        os.path.join(RESULT_DIR, "group_ablation_oof_metrics.csv"),
        index=False,
    )

    # =========================
    # Mean ± std across folds
    # =========================
    if not fold_metrics_df.empty:
        numeric_cols = fold_metrics_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "Fold"]

        summary_mean = fold_metrics_df.groupby(["Split_Scheme", "Ablation", "Model"])[numeric_cols].mean()
        summary_std = fold_metrics_df.groupby(["Split_Scheme", "Ablation", "Model"])[numeric_cols].std()

        summary_df = summary_mean.copy()
        for col in numeric_cols:
            summary_df[f"{col}_std"] = summary_std[col]

        summary_df = summary_df.reset_index()
    else:
        summary_df = pd.DataFrame()

    summary_df.to_csv(
        os.path.join(RESULT_DIR, "group_ablation_summary_mean_std.csv"),
        index=False,
    )

    # =========================
    # OOF predictions
    # =========================
    if len(oof_prediction_frames) > 0:
        pred_all_df = pd.concat(oof_prediction_frames, axis=0, ignore_index=True)
    else:
        pred_all_df = pd.DataFrame()

    pred_all_df.to_csv(
        os.path.join(RESULT_DIR, "group_ablation_oof_predictions_long.csv"),
        index=False,
    )

    # =========================
    # Group assignment
    # =========================
    group_assignment_df = pd.DataFrame(group_assignment_rows)
    group_assignment_df.to_csv(
        os.path.join(RESULT_DIR, "group_fold_assignment.csv"),
        index=False,
    )

    # =========================
    # Save config
    # =========================
    config = {
        "data_dir": DATA_DIR,
        "result_dir": RESULT_DIR,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "fixed_threshold": FIXED_THRESHOLD,
        "split_schemes": SPLIT_SCHEMES,
        "ablation_modes": ABLATION_MODES,
        "pathway_pooling_mode_for_fixed_baselines": PATHWAY_POOLING_MODE,
        "fixed_feature_definition": "Morgan FP + pooled scGPT pathway embedding + log1p(gene_count)",
        "ablation_definition": {
            "original": "Morgan + pathway + gene_count",
            "mask_molecule": "zero Morgan + pathway + gene_count",
            "mask_pathway": "Morgan + zero pathway + zero gene_count",
            "mask_both": "zero Morgan + zero pathway + zero gene_count",
        },
        "learnable_pooling_definition": "Morgan FP + learnable weighted pooling over scGPT gene embeddings + log1p(gene_count)",
        "models": MODEL_NAMES,
        "dnn_epochs": DNN_EPOCHS,
        "dnn_batch_size": DNN_BATCH_SIZE,
        "dnn_lr": DNN_LR,
        "dnn_weight_decay": DNN_WEIGHT_DECAY,
        "learnable_pooling_epochs": LP_EPOCHS,
        "learnable_pooling_batch_size": LP_BATCH_SIZE,
        "learnable_pooling_lr": LP_LR,
        "learnable_pooling_weight_decay": LP_WEIGHT_DECAY,
        "learnable_pooling_dropout": LP_DROPOUT,
        "freeze_scgpt": FREEZE_SCGPT,
        "use_gpu": USE_GPU,
        "completed_runs": completed_runs,
    }

    with open(
        os.path.join(RESULT_DIR, "group_ablation_config.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print("\n✅ Group split + feature-mask ablation experiments finished.")
    print("Results saved to:", RESULT_DIR)
    print("Main files:")
    print("1. group_ablation_fold_metrics.csv")
    print("2. group_ablation_oof_metrics.csv")
    print("3. group_ablation_summary_mean_std.csv")
    print("4. group_ablation_oof_predictions_long.csv")
    print("5. group_fold_assignment.csv")
    print("6. group_ablation_config.json")


if __name__ == "__main__":
    run_all_experiments()
