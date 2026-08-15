import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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
OUTPUT_DIR = "Weights_QFormer_Full"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
FIXED_THRESHOLD = 0.5

EMBED_DIM = 512
NUM_HEADS = 8
NUM_QUERIES = 8
NUM_QFORMER_LAYERS = 2
DROPOUT_RATE = 0.1
NUM_CLASSES = 2
FREEZE_GENE_EMBEDDING = True
USE_MEAN_RESIDUAL = True

BATCH_SIZE_TRAIN = 2048
BATCH_SIZE_EVAL = 4096
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ETA_MIN = 1e-5
NUM_WORKERS = 8

BEST_METRIC = "AUPRC"  # options: "AUPRC", "AUROC", "Train_Loss"

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
# Dataset: original features only
# =========================
class QFormerFullDataset(Dataset):
    def __init__(self, fps_path, labels_path, gene_ids):
        self.fps_path = fps_path
        self.labels_path = labels_path
        self.gene_ids = gene_ids

        self.labels = np.load(labels_path, mmap_mode="c")
        self.fps = None

        if len(self.labels) != len(self.gene_ids):
            raise ValueError(
                f"labels length {len(self.labels)} != gene_ids length {len(self.gene_ids)}"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.fps is None:
            self.fps = np.load(self.fps_path, mmap_mode="c")

        fp = torch.from_numpy(self.fps[idx].astype(np.float32)).float()

        gids = self.gene_ids[idx]
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

    print("========== 全量训练数据 ==========")
    print(f"Data dir      : {DATA_DIR}")
    print(f"Total samples : {len(labels)}")
    print(f"Low  label=0  : {(labels == 0).sum()}")
    print(f"High label=1  : {(labels == 1).sum()}")
    print(f"High ratio    : {labels.mean():.4f}")
    print(f"scGPT shape   : {tuple(scgpt_matrix.shape)}")
    print("================================")

    return fps_path, labels_path, gene_meta_path, scgpt_path, gene_ids, scgpt_matrix, labels, threshold_info


# =========================
# Model / eval
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


@torch.no_grad()
def evaluate_model(model, eval_loader):
    model.eval()
    probs = []
    labels_all = []

    for fps, gene_ids, mask, gene_counts, batch_labels in tqdm(
        eval_loader,
        desc="Evaluating full training set",
        leave=False,
    ):
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


def get_score_for_best(metrics, train_loss):
    if BEST_METRIC == "Train_Loss":
        # Lower loss is better, so use negative loss as score.
        return -float(train_loss)

    value = metrics.get(BEST_METRIC, np.nan)
    if np.isnan(value):
        return -np.inf
    return float(value)


def save_checkpoint(
    path,
    raw_model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    metrics,
    class_counts,
    class_weights,
    config,
):
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": float(train_loss),
            "metrics": metrics,
            "class_counts": class_counts.tolist(),
            "class_weights": class_weights.tolist(),
            "config": config,
        },
        path,
    )


# =========================
# Full-data training
# =========================
def train_full_qformer_best():
    set_seed(RANDOM_STATE)

    (
        fps_path,
        labels_path,
        gene_meta_path,
        scgpt_path,
        gene_ids,
        scgpt_matrix,
        labels,
        threshold_info,
    ) = load_data()

    full_dataset = QFormerFullDataset(
        fps_path=fps_path,
        labels_path=labels_path,
        gene_ids=gene_ids,
    )

    train_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_GPU,
        collate_fn=qformer_collate_fn,
    )

    eval_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE_EVAL,
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

    class_counts = np.bincount(labels, minlength=2)
    if np.any(class_counts == 0):
        raise RuntimeError(f"Training labels contain empty class: {class_counts}")

    class_weights = class_counts.sum() / (2.0 * class_counts)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    print(
        f"Class weights: Low={class_weights[0]:.4f}, "
        f"High={class_weights[1]:.4f}"
    )

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

    config = {
        "task": "binary_classification",
        "training_mode": "full_data_training",
        "best_model_selection": (
            f"Selected by training-set {BEST_METRIC}; no independent validation set is used."
        ),
        "data_dir": DATA_DIR,
        "output_dir": OUTPUT_DIR,
        "fps_path": fps_path,
        "labels_path": labels_path,
        "gene_metadata_path": gene_meta_path,
        "scgpt_matrix_path": scgpt_path,
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
        "batch_size_eval": BATCH_SIZE_EVAL,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "eta_min": ETA_MIN,
        "loss": "CrossEntropyLoss with full-data class weights",
        "fixed_threshold": FIXED_THRESHOLD,
        "random_state": RANDOM_STATE,
        "use_gpu": USE_GPU,
        "compiled": compiled,
        "class_counts": {
            "low": int(class_counts[0]),
            "high": int(class_counts[1]),
        },
        "class_weights": {
            "low": float(class_weights[0]),
            "high": float(class_weights[1]),
        },
    }

    if threshold_info is not None:
        config["classification_thresholds"] = threshold_info

    best_score = -np.inf
    best_epoch = None
    best_metrics = None
    best_probs = None
    train_log_rows = []

    best_ckpt_path = os.path.join(OUTPUT_DIR, "best_qformer_full_checkpoint.pt")
    final_ckpt_path = os.path.join(OUTPUT_DIR, "final_qformer_full_checkpoint.pt")
    best_state_dict_path = os.path.join(OUTPUT_DIR, "best_qformer_full_state_dict.pt")
    final_state_dict_path = os.path.join(OUTPUT_DIR, "final_qformer_full_state_dict.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total_seen = 0

        pbar = tqdm(
            train_loader,
            desc=f"Full QFormer training | Epoch {epoch}/{EPOCHS}",
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

        y_train_eval, train_probs = evaluate_model(model, eval_loader)
        metrics = compute_metrics(y_train_eval, train_probs, FIXED_THRESHOLD)
        current_score = get_score_for_best(metrics, epoch_loss)

        row = {
            "Epoch": epoch,
            "Train_Loss": epoch_loss,
            "LR": optimizer.param_groups[0]["lr"],
            "Best_Metric": BEST_METRIC,
            "Best_Score_For_Selection": current_score,
        }
        row.update(metrics)
        train_log_rows.append(row)

        print(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"Loss={epoch_loss:.5f} | "
            f"AUROC={metrics['AUROC']:.4f} | "
            f"AUPRC={metrics['AUPRC']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"MCC={metrics['MCC']:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.6f}"
        )

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_metrics = metrics.copy()
            best_probs = train_probs.copy()

            save_checkpoint(
                path=best_ckpt_path,
                raw_model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=epoch_loss,
                metrics=metrics,
                class_counts=class_counts,
                class_weights=class_weights,
                config=config,
            )
            torch.save(raw_model.state_dict(), best_state_dict_path)
            print(f"🌟 Saved new best model at epoch {epoch} by {BEST_METRIC}: {best_score:.6f}")

    # Save final model after the last epoch.
    final_metrics = train_log_rows[-1].copy()
    save_checkpoint(
        path=final_ckpt_path,
        raw_model=raw_model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=EPOCHS,
        train_loss=float(final_metrics["Train_Loss"]),
        metrics={k: v for k, v in final_metrics.items() if k not in ["Epoch", "Train_Loss", "LR"]},
        class_counts=class_counts,
        class_weights=class_weights,
        config=config,
    )
    torch.save(raw_model.state_dict(), final_state_dict_path)

    # Save logs.
    train_log_df = pd.DataFrame(train_log_rows)
    train_log_path = os.path.join(OUTPUT_DIR, "full_qformer_training_log.csv")
    train_log_df.to_csv(train_log_path, index=False)

    # Save full-training predictions from best model epoch.
    pred_df = pd.DataFrame({
        "True_Label": labels,
        "Best_Epoch": int(best_epoch),
        "Pred_Prob_High": best_probs,
        "Pred_Label_0p5": (best_probs >= FIXED_THRESHOLD).astype(int),
    })
    pred_path = os.path.join(OUTPUT_DIR, "best_qformer_full_training_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    config["best_epoch"] = int(best_epoch)
    config["best_metric"] = BEST_METRIC
    config["best_score"] = float(best_score)
    config["best_training_metrics"] = {
        k: float(v) if isinstance(v, (float, np.floating)) else int(v)
        for k, v in best_metrics.items()
    }
    config["output_files"] = {
        "best_checkpoint": best_ckpt_path,
        "final_checkpoint": final_ckpt_path,
        "best_state_dict": best_state_dict_path,
        "final_state_dict": final_state_dict_path,
        "training_log": train_log_path,
        "best_training_predictions": pred_path,
    }

    config_path = os.path.join(OUTPUT_DIR, "full_qformer_training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print("\n✅ 全量 QFormer 训练完成")
    print(f"Best epoch      : {best_epoch}")
    print(f"Best by         : {BEST_METRIC}")
    print(f"Best score      : {best_score:.6f}")
    print(f"Best checkpoint : {best_ckpt_path}")
    print(f"Final checkpoint: {final_ckpt_path}")
    print(f"Training log    : {train_log_path}")
    print(f"Config          : {config_path}")


if __name__ == "__main__":
    train_full_qformer_best()
