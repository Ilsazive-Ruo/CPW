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
from sklearn.model_selection import StratifiedKFold
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
from tqdm import tqdm

from QFormer import QFormerClassifier

warnings.filterwarnings("ignore")


# =========================
# Basic config
# =========================
DATA_DIR = "CLSData2"
RESULT_DIR = "QFormer_CV_FeatureMask_Results"
os.makedirs(RESULT_DIR, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42
FIXED_THRESHOLD = 0.5

ABLATION_MODES = [
    "original",
    "mask_molecule",
    "mask_pathway_genes",
    "mask_both",
    "shuffle_molecule",
    "randomize_pathway_genes_same_length",
]

# Q-Former parameters
EMBED_DIM = 512
NUM_HEADS = 8
NUM_QUERIES = 8
NUM_QFORMER_LAYERS = 2
DROPOUT_RATE = 0.1
NUM_CLASSES = 2
FREEZE_GENE_EMBEDDING = True
USE_MEAN_RESIDUAL = True

# Training parameters
BATCH_SIZE_TRAIN = 2048
BATCH_SIZE_VAL = 4096
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ETA_MIN = 1e-5
NUM_WORKERS = 8

USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")


# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


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

    return {
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


# =========================
# Dataset with feature masks
# =========================
class QFormerAblationDataset(Dataset):
    def __init__(
        self,
        fps_path,
        labels_path,
        gene_ids,
        scgpt_num_embeddings,
        ablation_mode="original",
        seed=42,
    ):
        self.fps_path = fps_path
        self.labels_path = labels_path
        self.gene_ids = gene_ids
        self.scgpt_num_embeddings = int(scgpt_num_embeddings)
        self.ablation_mode = ablation_mode
        self.seed = seed

        self.labels = np.load(labels_path, mmap_mode="c")
        self.fps = None

        if len(self.labels) != len(self.gene_ids):
            raise ValueError(
                f"labels length {len(self.labels)} != gene_ids length {len(self.gene_ids)}"
            )

        if self.ablation_mode not in ABLATION_MODES:
            raise ValueError(f"Unknown ablation mode: {self.ablation_mode}")

        rng = np.random.default_rng(seed)
        self.shuffle_indices = np.arange(len(self.labels))
        rng.shuffle(self.shuffle_indices)

    def __len__(self):
        return len(self.labels)

    def _random_gene_ids_same_length(self, idx, length):
        # Deterministic per sample. Valid gene ids are assumed to be 1..num_embeddings-1.
        # 0 is padding_idx and is not sampled here.
        if length <= 0:
            return [0]

        rng = np.random.default_rng(self.seed + int(idx) * 1009)
        max_gene_id = self.scgpt_num_embeddings - 1
        if max_gene_id <= 1:
            return [0]

        replace = length > max_gene_id
        return rng.choice(
            np.arange(1, max_gene_id + 1),
            size=length,
            replace=replace,
        ).astype(np.int64).tolist()

    def __getitem__(self, idx):
        if self.fps is None:
            self.fps = np.load(self.fps_path, mmap_mode="c")

        # -------- molecule branch --------
        if self.ablation_mode in ["mask_molecule", "mask_both"]:
            fp_np = np.zeros_like(self.fps[idx], dtype=np.float32)
        elif self.ablation_mode == "shuffle_molecule":
            fp_np = self.fps[self.shuffle_indices[idx]].astype(np.float32)
        else:
            fp_np = self.fps[idx].astype(np.float32)

        fp = torch.from_numpy(fp_np).float()

        # -------- pathway branch --------
        gids = self.gene_ids[idx]
        if self.ablation_mode in ["mask_pathway_genes", "mask_both"]:
            gene_id = torch.tensor([0], dtype=torch.long)
            gene_count = torch.tensor([0.0], dtype=torch.float32)
        elif self.ablation_mode == "randomize_pathway_genes_same_length":
            rand_gids = self._random_gene_ids_same_length(idx, len(gids))
            gene_id = torch.tensor(rand_gids, dtype=torch.long)
            gene_count = torch.tensor([np.log1p(len(gids))], dtype=torch.float32)
        else:
            gene_id = torch.tensor(gids, dtype=torch.long)
            gene_count = torch.tensor([np.log1p(len(gids))], dtype=torch.float32)

        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        return fp, gene_id, gene_count, label


def qformer_collate_fn(batch):
    fps, gene_ids_list, gene_counts, labels = zip(*batch)

    padded_gene_ids = pad_sequence(
        gene_ids_list,
        batch_first=True,
        padding_value=0,
    )

    lengths = torch.tensor([len(g) for g in gene_ids_list], dtype=torch.long)
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
# Data loading
# =========================
def load_data():
    fps_path = os.path.join(DATA_DIR, "cls_morgan_fps.npy")
    labels_path = os.path.join(DATA_DIR, "cls_labels.npy")
    gene_meta_path = os.path.join(DATA_DIR, "cls_gene_metadata.pkl")
    scgpt_path = os.path.join(DATA_DIR, "scgpt_matrix.pt")
    threshold_path = os.path.join(DATA_DIR, "classification_thresholds.json")

    with open(gene_meta_path, "rb") as f:
        meta = pickle.load(f)

    labels = np.load(labels_path).astype(np.int64)
    scgpt_matrix = torch.load(scgpt_path, map_location="cpu")

    gene_ids = meta["gene_ids"]
    if len(labels) != len(gene_ids):
        raise ValueError(f"labels length {len(labels)} != gene_ids length {len(gene_ids)}")

    threshold_info = None
    if os.path.exists(threshold_path):
        with open(threshold_path, "r", encoding="utf-8") as f:
            threshold_info = json.load(f)

    print("========== 数据集信息 ==========")
    print(f"Data dir      : {DATA_DIR}")
    print(f"Total samples : {len(labels)}")
    print(f"Low  label=0  : {(labels == 0).sum()}")
    print(f"High label=1  : {(labels == 1).sum()}")
    print(f"High ratio    : {labels.mean():.4f}")
    print(f"scGPT shape   : {tuple(scgpt_matrix.shape)}")
    print("================================")

    return fps_path, labels_path, gene_ids, scgpt_matrix, labels, threshold_info


# =========================
# Model / train / eval
# =========================
def build_model(scgpt_matrix):
    raw_model = QFormerClassifier(
        pretrained_scgpt_matrix=scgpt_matrix,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_queries=NUM_QUERIES,
        num_qformer_layers=NUM_QFORMER_LAYERS,
        dropout_rate=DROPOUT_RATE,
        num_classes=NUM_CLASSES,
        freeze_gene_embedding=FREEZE_GENE_EMBEDDING,
        use_mean_residual=USE_MEAN_RESIDUAL,
    ).to(DEVICE)

    try:
        model = torch.compile(raw_model)
        compiled = True
    except Exception:
        model = raw_model
        compiled = False

    return raw_model, model, compiled


def evaluate_model(model, val_loader):
    model.eval()
    probs = []
    labels_all = []

    with torch.no_grad():
        for fps, gene_ids, mask, gene_counts, batch_labels in val_loader:
            fps = fps.to(DEVICE, non_blocking=True)
            gene_ids = gene_ids.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            gene_counts = gene_counts.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_GPU):
                logits, _ = model(fps, gene_ids, mask, gene_counts)

            prob = torch.softmax(logits.float(), dim=1)[:, 1]
            probs.extend(prob.cpu().numpy())
            labels_all.extend(batch_labels.numpy())

    return np.asarray(labels_all, dtype=np.int64), np.asarray(probs, dtype=np.float32)


def train_one_fold(
    fold_id,
    ablation_mode,
    train_idx,
    val_idx,
    fps_path,
    labels_path,
    gene_ids,
    scgpt_matrix,
    labels,
):
    dataset = QFormerAblationDataset(
        fps_path=fps_path,
        labels_path=labels_path,
        gene_ids=gene_ids,
        scgpt_num_embeddings=scgpt_matrix.shape[0],
        ablation_mode=ablation_mode,
        seed=RANDOM_STATE,
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_GPU,
        collate_fn=qformer_collate_fn,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_GPU,
        collate_fn=qformer_collate_fn,
    )

    raw_model, model, compiled = build_model(scgpt_matrix)
    if compiled:
        print("✅ torch.compile enabled")
    else:
        print("⚠️ torch.compile unavailable; using raw model")

    class_counts = np.bincount(labels[train_idx], minlength=2)
    if np.any(class_counts == 0):
        raise RuntimeError(f"Fold {fold_id} training labels contain empty class: {class_counts}")

    class_weights = class_counts.sum() / (2.0 * class_counts)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=ETA_MIN,
    )

    amp_scaler = torch.amp.GradScaler("cuda", enabled=USE_GPU)

    best_auprc = -np.inf
    best_metrics = None
    best_probs = None
    epoch_rows = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total_seen = 0

        pbar = tqdm(
            train_loader,
            desc=f"{ablation_mode} | Fold {fold_id} | Epoch {epoch + 1}/{EPOCHS}",
            leave=False,
        )

        for fps, gene_ids_b, mask, gene_counts, batch_labels in pbar:
            fps = fps.to(DEVICE, non_blocking=True)
            gene_ids_b = gene_ids_b.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            gene_counts = gene_counts.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_GPU):
                logits, _ = model(fps, gene_ids_b, mask, gene_counts)
                loss = criterion(logits, batch_labels)

            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

            bs = batch_labels.size(0)
            running_loss += loss.item() * bs
            total_seen += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        epoch_loss = running_loss / max(1, total_seen)

        y_val, val_probs = evaluate_model(model, val_loader)
        metrics = compute_metrics(y_val, val_probs, FIXED_THRESHOLD)
        metrics.update({
            "Ablation": ablation_mode,
            "Fold": fold_id,
            "Epoch": epoch + 1,
            "Train_Loss": epoch_loss,
            "LR": optimizer.param_groups[0]["lr"],
        })
        epoch_rows.append(metrics.copy())

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Loss={epoch_loss:.5f} | "
            f"AUROC={metrics['AUROC']:.4f} | "
            f"AUPRC={metrics['AUPRC']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"MCC={metrics['MCC']:.4f}"
        )

        score = metrics["AUPRC"] if not np.isnan(metrics["AUPRC"]) else -np.inf
        if score > best_auprc:
            best_auprc = score
            best_metrics = metrics.copy()
            best_probs = val_probs.copy()

    # Save fold model state_dict for original only, to avoid huge disk usage.
    if ablation_mode == "original":
        fold_model_path = os.path.join(
            RESULT_DIR,
            f"qformer_original_fold{fold_id}_best_last_state_dict.pt",
        )
        torch.save(raw_model.state_dict(), fold_model_path)
        best_metrics["Saved_Model_Path"] = fold_model_path

    return best_metrics, best_probs, pd.DataFrame(epoch_rows)


# =========================
# Main CV + ablation
# =========================
def run_qformer_cv_feature_mask():
    set_seed(RANDOM_STATE)

    fps_path, labels_path, gene_ids, scgpt_matrix, labels, threshold_info = load_data()

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_metric_rows = []
    epoch_metric_frames = []
    oof_metric_rows = []
    oof_prediction_frames = []

    split_list = list(skf.split(np.zeros(len(labels)), labels))

    for ablation_mode in ABLATION_MODES:
        print("\n" + "#" * 90)
        print(f"# QFormer ablation mode: {ablation_mode}")
        print("#" * 90)

        oof_probs = np.full(len(labels), np.nan, dtype=np.float32)
        oof_fold = np.zeros(len(labels), dtype=np.int64)

        for fold_id, (train_idx, val_idx) in enumerate(split_list, start=1):
            print("\n" + "=" * 90)
            print(f"Ablation: {ablation_mode} | Fold {fold_id}/{N_SPLITS}")
            print("=" * 90)
            print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")
            print(f"Train high ratio: {labels[train_idx].mean():.4f} | Val high ratio: {labels[val_idx].mean():.4f}")

            best_metrics, best_probs, epoch_df = train_one_fold(
                fold_id=fold_id,
                ablation_mode=ablation_mode,
                train_idx=train_idx,
                val_idx=val_idx,
                fps_path=fps_path,
                labels_path=labels_path,
                gene_ids=gene_ids,
                scgpt_matrix=scgpt_matrix,
                labels=labels,
            )

            oof_probs[val_idx] = best_probs
            oof_fold[val_idx] = fold_id

            best_metrics.update({
                "Ablation": ablation_mode,
                "Fold": fold_id,
                "Train_N": len(train_idx),
                "Val_N": len(val_idx),
                "Best_By": "AUPRC",
            })
            fold_metric_rows.append(best_metrics)
            epoch_metric_frames.append(epoch_df)

            print(
                f"🌟 Best Fold {fold_id} | {ablation_mode}: "
                f"AUROC={best_metrics['AUROC']:.4f} | "
                f"AUPRC={best_metrics['AUPRC']:.4f} | "
                f"F1={best_metrics['F1']:.4f} | "
                f"MCC={best_metrics['MCC']:.4f}"
            )

        valid_mask = ~np.isnan(oof_probs)
        oof_metrics = compute_metrics(labels[valid_mask], oof_probs[valid_mask], FIXED_THRESHOLD)
        oof_metrics.update({
            "Ablation": ablation_mode,
            "OOF_N": int(valid_mask.sum()),
        })
        oof_metric_rows.append(oof_metrics)

        pred_df = pd.DataFrame({
            "Ablation": ablation_mode,
            "Fold": oof_fold,
            "True_Label": labels,
            "Pred_Prob_High": oof_probs,
            "Pred_Label_0p5": (oof_probs >= FIXED_THRESHOLD).astype(int),
        })
        oof_prediction_frames.append(pred_df)

    # =========================
    # Save results
    # =========================
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    fold_metrics_df = fold_metrics_df.sort_values(["Ablation", "Fold"])
    fold_metrics_df.to_csv(
        os.path.join(RESULT_DIR, "qformer_feature_mask_fold_metrics.csv"),
        index=False,
    )

    oof_metrics_df = pd.DataFrame(oof_metric_rows)
    oof_metrics_df = oof_metrics_df.sort_values(["Ablation"])
    oof_metrics_df.to_csv(
        os.path.join(RESULT_DIR, "qformer_feature_mask_oof_metrics.csv"),
        index=False,
    )

    if len(epoch_metric_frames) > 0:
        epoch_metrics_df = pd.concat(epoch_metric_frames, axis=0, ignore_index=True)
    else:
        epoch_metrics_df = pd.DataFrame()
    epoch_metrics_df.to_csv(
        os.path.join(RESULT_DIR, "qformer_feature_mask_epoch_metrics.csv"),
        index=False,
    )

    pred_all_df = pd.concat(oof_prediction_frames, axis=0, ignore_index=True)
    pred_all_df.to_csv(
        os.path.join(RESULT_DIR, "qformer_feature_mask_oof_predictions.csv"),
        index=False,
    )

    numeric_cols = fold_metrics_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Fold"]

    summary_mean = fold_metrics_df.groupby("Ablation")[numeric_cols].mean()
    summary_std = fold_metrics_df.groupby("Ablation")[numeric_cols].std()

    summary_df = summary_mean.copy()
    for col in numeric_cols:
        summary_df[f"{col}_std"] = summary_std[col]
    summary_df = summary_df.reset_index()
    summary_df.to_csv(
        os.path.join(RESULT_DIR, "qformer_feature_mask_summary_mean_std.csv"),
        index=False,
    )

    config = {
        "task": "binary_classification",
        "data_dir": DATA_DIR,
        "result_dir": RESULT_DIR,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "fixed_threshold": FIXED_THRESHOLD,
        "ablation_modes": ABLATION_MODES,
        "ablation_definition": {
            "original": "normal Morgan FP + normal pathway genes + log1p(gene_count)",
            "mask_molecule": "zero Morgan FP + normal pathway genes + log1p(gene_count)",
            "mask_pathway_genes": "normal Morgan FP + pathway gene ids replaced by [0] + gene_count=0",
            "mask_both": "zero Morgan FP + pathway gene ids replaced by [0] + gene_count=0",
            "shuffle_molecule": "Morgan FP assigned from a globally shuffled sample index + normal pathway",
            "randomize_pathway_genes_same_length": "normal Morgan FP + random gene ids with same pathway length + original gene_count",
        },
        "model_class": "QFormerClassifier",
        "num_classes": NUM_CLASSES,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "num_queries": NUM_QUERIES,
        "num_qformer_layers": NUM_QFORMER_LAYERS,
        "dropout_rate": DROPOUT_RATE,
        "freeze_gene_embedding": FREEZE_GENE_EMBEDDING,
        "use_mean_residual": USE_MEAN_RESIDUAL,
        "epochs": EPOCHS,
        "batch_size_train": BATCH_SIZE_TRAIN,
        "batch_size_val": BATCH_SIZE_VAL,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "eta_min": ETA_MIN,
        "loss": "CrossEntropyLoss with class weights per fold",
        "use_gpu": USE_GPU,
    }
    if threshold_info is not None:
        config["classification_thresholds"] = threshold_info

    with open(
        os.path.join(RESULT_DIR, "qformer_feature_mask_config.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print("\n✅ QFormer 交叉验证 + 特征 mask 消融实验完成。")
    print(f"结果目录: {RESULT_DIR}")
    print("主要文件:")
    print("1. qformer_feature_mask_fold_metrics.csv")
    print("2. qformer_feature_mask_oof_metrics.csv")
    print("3. qformer_feature_mask_summary_mean_std.csv")
    print("4. qformer_feature_mask_epoch_metrics.csv")
    print("5. qformer_feature_mask_oof_predictions.csv")
    print("6. qformer_feature_mask_config.json")


if __name__ == "__main__":
    run_qformer_cv_feature_mask()
