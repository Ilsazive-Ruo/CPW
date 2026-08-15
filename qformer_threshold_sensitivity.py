import os
import re
import gc
import json
import pickle
import random
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score,
    average_precision_score, brier_score_loss, log_loss, confusion_matrix,
)

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

try:
    from QFormer import QFormerClassifier
except ImportError:
    from CPW_model_cls_qformer import QFormerClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

SOURCE_CSV = "Source/interaction_processed.csv"
REFERENCE_DIR = "CLSData2"

RESULT_DIR = "QFormer_Threshold_Sensitivity"
CACHE_DIR = os.path.join(RESULT_DIR, "cache")

TAIL_FRACTIONS = [0.10, 0.20, 0.30]

N_SPLITS = 5
RANDOM_STATE = 42
FIXED_THRESHOLD = 0.5

# Match manuscript QFormer
EMBED_DIM = 512
NUM_HEADS = 8
NUM_QUERIES = 8
NUM_QFORMER_LAYERS = 2
DROPOUT_RATE = 0.1
FREEZE_GENE_EMBEDDING = True
USE_MEAN_RESIDUAL = True

EPOCHS = 15
TRAIN_BATCH_SIZE = 2048
VAL_BATCH_SIZE = 4096
LR = 1e-3
WEIGHT_DECAY = 1e-4
ETA_MIN = 1e-5
NUM_WORKERS = 8

# Matches the existing CV workflow: best epoch in each fold by validation AUPRC.
SELECT_BEST_EPOCH_BY_AUPRC = True

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
GENE_SPLIT_REGEX = r"[;,|]+"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if USE_GPU:
        torch.backends.cudnn.benchmark = True


def norm_col(x):
    return str(x).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def find_col(columns, candidates):
    cmap = {norm_col(c): c for c in columns}
    for candidate in candidates:
        if norm_col(candidate) in cmap:
            return cmap[norm_col(candidate)]
    raise KeyError(f"Could not find any of {candidates}. Available columns: {list(columns)}")


def safe_div(a, b):
    return float(a / b) if b else np.nan


def metrics(y_true, prob):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= FIXED_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "AUROC": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
        "AUPRC": average_precision_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "Specificity": safe_div(tn, tn + fp),
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
        "Cohen_Kappa": cohen_kappa_score(y_true, pred),
        "Brier_Score": brier_score_loss(y_true, prob),
        "Log_Loss": log_loss(y_true, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def load_reference():
    meta_path = os.path.join(REFERENCE_DIR, "cls_gene_metadata.pkl")
    scgpt_path = os.path.join(REFERENCE_DIR, "scgpt_matrix.pt")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    vocab = meta.get("vocab", meta.get("gene_vocab"))
    if not isinstance(vocab, dict):
        raise KeyError("cls_gene_metadata.pkl must contain 'vocab' or 'gene_vocab'.")

    vocab = {str(k).strip(): int(v) for k, v in vocab.items()}
    vocab_upper = {str(k).strip().upper(): int(v) for k, v in vocab.items()}

    matrix = torch.load(scgpt_path, map_location="cpu")
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=torch.float32)

    return vocab, vocab_upper, matrix.float()


def parse_gene_set(text, vocab, vocab_upper):
    if pd.isna(text):
        return []
    genes = [g.strip() for g in re.split(GENE_SPLIT_REGEX, str(text)) if g.strip()]
    out, seen = [], set()

    for gene in genes:
        gid = vocab.get(gene, vocab_upper.get(gene.upper()))
        if gid is not None and int(gid) > 0 and int(gid) not in seen:
            out.append(int(gid))
            seen.add(int(gid))
    return out


# ============================================================
# PREPARE COMPACT 30%-TAIL CACHE
# ============================================================

def prepare(force=False):
    os.makedirs(CACHE_DIR, exist_ok=True)

    required = [
        "unique_morgan.npy", "unique_mol_valid.npy",
        "unique_pathway_gene_ids.pkl", "unique_path_valid.npy",
        "master_mol_idx.npy", "master_path_idx.npy",
        "master_p.npy", "master_source_row.npy", "master_valid.npy",
        "quantile_thresholds.csv",
    ]

    if not force and all(os.path.exists(os.path.join(CACHE_DIR, f)) for f in required):
        print("Cache already exists. Use --force_prepare to rebuild.")
        return

    if not os.path.exists(SOURCE_CSV):
        raise FileNotFoundError(SOURCE_CSV)

    header = pd.read_csv(SOURCE_CSV, nrows=0)
    smiles_col = find_col(header.columns, ["SMILES"])
    gene_col = find_col(header.columns, ["Gene_Set", "GeneSet", "Genes"])
    p_col = find_col(
        header.columns,
        ["CorrectedPValue", "Corrected_PValue", "corrected p value",
         "corrected_p_value", "FDR", "AdjustedPValue"],
    )

    print("Columns:")
    print("  SMILES      =", smiles_col)
    print("  Gene_Set    =", gene_col)
    print("  Corrected P =", p_col)

    df = pd.read_csv(SOURCE_CSV, usecols=[smiles_col, gene_col, p_col])
    df["_source_row"] = np.arange(len(df), dtype=np.int64)
    df["_p"] = pd.to_numeric(df[p_col], errors="coerce")

    p_valid_mask = df["_p"].notna() & np.isfinite(df["_p"]) & (df["_p"] >= 0)
    p_all = df.loc[p_valid_mask, "_p"].astype(float)

    q_rows = []
    for t in TAIL_FRACTIONS:
        q_low = float(p_all.quantile(t))
        q_high = float(p_all.quantile(1 - t))
        q_rows.append({
            "Tail_Fraction": t,
            "Tail_Percent": int(round(t * 100)),
            "Low_CorrectedP_Cutoff": q_low,
            "High_CorrectedP_Cutoff": q_high,
            "Raw_Positive_N": int((p_all <= q_low).sum()),
            "Raw_Negative_N": int((p_all >= q_high).sum()),
            "Raw_Total_Selected_N": int((p_all <= q_low).sum() + (p_all >= q_high).sum()),
        })

    qdf = pd.DataFrame(q_rows)
    qdf.to_csv(os.path.join(CACHE_DIR, "quantile_thresholds.csv"), index=False)
    print(qdf.to_string(index=False))

    # 30% is the largest selected set. 10% and 20% are nested inside it.
    max_t = max(TAIL_FRACTIONS)
    q30 = float(p_all.quantile(max_t))
    q70 = float(p_all.quantile(1 - max_t))

    master = df.loc[
        p_valid_mask & ((df["_p"] <= q30) | (df["_p"] >= q70)),
        [smiles_col, gene_col, "_p", "_source_row"],
    ].reset_index(drop=True)

    print(f"30%-tail master rows: {len(master):,}")

    vocab, vocab_upper, _ = load_reference()

    # ---------------- Unique molecules ----------------
    smiles_array = master[smiles_col].fillna("").astype(str).to_numpy()
    unique_smiles, master_mol_idx = np.unique(smiles_array, return_inverse=True)

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
    )

    unique_morgan = np.zeros((len(unique_smiles), MORGAN_BITS), dtype=np.uint8)
    unique_mol_valid = np.zeros(len(unique_smiles), dtype=bool)

    for i, smi in enumerate(tqdm(unique_smiles, desc="Morgan fingerprints")):
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            continue

        fp = generator.GetFingerprint(mol)
        arr = np.zeros(MORGAN_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        unique_morgan[i] = arr
        unique_mol_valid[i] = True

    # ---------------- Unique pathways ----------------
    gene_array = master[gene_col].fillna("").astype(str).to_numpy()
    unique_gene_sets, master_path_idx = np.unique(gene_array, return_inverse=True)

    unique_gene_ids = []
    unique_path_valid = np.zeros(len(unique_gene_sets), dtype=bool)

    for i, text in enumerate(tqdm(unique_gene_sets, desc="Pathway gene mapping")):
        gids = parse_gene_set(text, vocab, vocab_upper)
        if gids:
            unique_gene_ids.append(gids)
            unique_path_valid[i] = True
        else:
            unique_gene_ids.append([0])

    master_valid = (
        unique_mol_valid[master_mol_idx]
        & unique_path_valid[master_path_idx]
    )

    np.save(os.path.join(CACHE_DIR, "unique_morgan.npy"), unique_morgan)
    np.save(os.path.join(CACHE_DIR, "unique_mol_valid.npy"), unique_mol_valid)
    np.save(os.path.join(CACHE_DIR, "unique_path_valid.npy"), unique_path_valid)
    np.save(os.path.join(CACHE_DIR, "master_mol_idx.npy"), master_mol_idx.astype(np.int32))
    np.save(os.path.join(CACHE_DIR, "master_path_idx.npy"), master_path_idx.astype(np.int32))
    np.save(os.path.join(CACHE_DIR, "master_p.npy"), master["_p"].to_numpy(np.float64))
    np.save(os.path.join(CACHE_DIR, "master_source_row.npy"), master["_source_row"].to_numpy(np.int64))
    np.save(os.path.join(CACHE_DIR, "master_valid.npy"), master_valid)

    with open(os.path.join(CACHE_DIR, "unique_pathway_gene_ids.pkl"), "wb") as f:
        pickle.dump(unique_gene_ids, f, protocol=pickle.HIGHEST_PROTOCOL)

    prep_summary = {
        "Source_CSV": SOURCE_CSV,
        "Total_Source_Rows": int(len(df)),
        "Valid_Numeric_P_Rows": int(p_valid_mask.sum()),
        "Master_30pct_Rows": int(len(master)),
        "Master_Modelable_Rows": int(master_valid.sum()),
        "Unique_Molecules": int(len(unique_smiles)),
        "Valid_Unique_Molecules": int(unique_mol_valid.sum()),
        "Unique_Gene_Sets": int(len(unique_gene_sets)),
        "Valid_Unique_Gene_Sets": int(unique_path_valid.sum()),
    }
    pd.DataFrame([prep_summary]).to_csv(
        os.path.join(RESULT_DIR, "preparation_summary.csv"), index=False
    )

    print("Preparation finished.")
    print(prep_summary)


def load_cache():
    with open(os.path.join(CACHE_DIR, "unique_pathway_gene_ids.pkl"), "rb") as f:
        unique_gene_ids = pickle.load(f)

    return {
        "morgan": np.load(os.path.join(CACHE_DIR, "unique_morgan.npy"), mmap_mode="r"),
        "mol_idx": np.load(os.path.join(CACHE_DIR, "master_mol_idx.npy"), mmap_mode="r"),
        "path_idx": np.load(os.path.join(CACHE_DIR, "master_path_idx.npy"), mmap_mode="r"),
        "p": np.load(os.path.join(CACHE_DIR, "master_p.npy"), mmap_mode="r"),
        "source_row": np.load(os.path.join(CACHE_DIR, "master_source_row.npy"), mmap_mode="r"),
        "valid": np.load(os.path.join(CACHE_DIR, "master_valid.npy"), mmap_mode="r"),
        "gene_ids": unique_gene_ids,
        "qdf": pd.read_csv(os.path.join(CACHE_DIR, "quantile_thresholds.csv")),
    }


def make_threshold_dataset(cache, t):
    row = cache["qdf"].loc[np.isclose(cache["qdf"]["Tail_Fraction"], t)]
    if len(row) != 1:
        raise RuntimeError(f"Threshold {t} not found in cache.")
    row = row.iloc[0]

    q_low = float(row["Low_CorrectedP_Cutoff"])
    q_high = float(row["High_CorrectedP_Cutoff"])

    p = np.asarray(cache["p"])
    valid = np.asarray(cache["valid"], dtype=bool)

    pos = valid & (p <= q_low)
    neg = valid & (p >= q_high)
    selected = pos | neg

    master_indices = np.where(selected)[0].astype(np.int64)
    labels = np.where(pos[master_indices], 1, 0).astype(np.int64)

    return master_indices, labels, q_low, q_high


# ============================================================
# DATASET
# ============================================================

class SensitivityDataset(Dataset):
    def __init__(self, cache, master_indices, labels):
        self.morgan = cache["morgan"]
        self.mol_idx = cache["mol_idx"]
        self.path_idx = cache["path_idx"]
        self.gene_ids = cache["gene_ids"]
        self.master_indices = np.asarray(master_indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        master_idx = int(self.master_indices[idx])
        mol_i = int(self.mol_idx[master_idx])
        path_i = int(self.path_idx[master_idx])

        fp = torch.from_numpy(
            np.asarray(self.morgan[mol_i], dtype=np.float32)
        )

        gids = self.gene_ids[path_i]
        gene_ids = torch.tensor(gids, dtype=torch.long)
        gene_count = torch.tensor(
            [np.log1p(sum(int(g) > 0 for g in gids))],
            dtype=torch.float32,
        )
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        return fp, gene_ids, gene_count, label, idx


def collate_fn(batch):
    fps, gene_ids_list, gene_counts, labels, local_idx = zip(*batch)

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
        torch.tensor(local_idx, dtype=torch.long),
    )


# ============================================================
# TRAIN / EVALUATE
# ============================================================

def build_model(scgpt_matrix):
    return QFormerClassifier(
        pretrained_scgpt_matrix=scgpt_matrix,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_queries=NUM_QUERIES,
        num_qformer_layers=NUM_QFORMER_LAYERS,
        dropout_rate=DROPOUT_RATE,
        num_classes=2,
        freeze_gene_embedding=FREEZE_GENE_EMBEDDING,
        use_mean_residual=USE_MEAN_RESIDUAL,
    ).to(DEVICE)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probs, ys, idxs = [], [], []

    for fps, gene_ids, mask, gene_counts, labels, local_idx in loader:
        fps = fps.to(DEVICE, non_blocking=True)
        gene_ids = gene_ids.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)
        gene_counts = gene_counts.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=USE_GPU
        ):
            logits, _ = model(fps, gene_ids, mask, gene_counts)

        prob = torch.softmax(logits.float(), dim=1)[:, 1]

        probs.append(prob.cpu().numpy())
        ys.append(labels.numpy())
        idxs.append(local_idx.numpy())

    return (
        np.concatenate(ys).astype(np.int64),
        np.concatenate(probs).astype(np.float32),
        np.concatenate(idxs).astype(np.int64),
    )


def train_fold(dataset, labels, train_idx, val_idx, scgpt_matrix, tail_pct, fold, quick):
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=512 if quick else TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0 if quick else NUM_WORKERS,
        pin_memory=USE_GPU,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=1024 if quick else VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0 if quick else NUM_WORKERS,
        pin_memory=USE_GPU,
        collate_fn=collate_fn,
    )

    set_seed(RANDOM_STATE + tail_pct * 100 + fold)
    model = build_model(scgpt_matrix)

    class_counts = np.bincount(labels[train_idx], minlength=2)
    class_weights = class_counts.sum() / (2.0 * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    epochs = 2 if quick else EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=ETA_MIN
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_GPU)

    best_score = -np.inf
    best_epoch = 0
    best_y = best_prob = best_idx = None
    epoch_rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0

        for fps, gene_ids, mask, gene_counts, batch_y, _ in tqdm(
            train_loader,
            desc=f"{tail_pct}% fold {fold} epoch {epoch}/{epochs}",
            leave=False,
        ):
            fps = fps.to(DEVICE, non_blocking=True)
            gene_ids = gene_ids.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            gene_counts = gene_counts.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=USE_GPU
            ):
                logits, _ = model(fps, gene_ids, mask, gene_counts)
                loss = criterion(logits, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * batch_y.size(0)
            n_seen += batch_y.size(0)

        scheduler.step()

        y_val, prob_val, local_idx_val = evaluate(model, val_loader)
        auroc = roc_auc_score(y_val, prob_val)
        auprc = average_precision_score(y_val, prob_val)

        epoch_rows.append({
            "Tail_Percent": tail_pct,
            "Fold": fold,
            "Epoch": epoch,
            "Train_Loss": loss_sum / max(n_seen, 1),
            "Val_AUROC": auroc,
            "Val_AUPRC": auprc,
        })

        score = auprc if SELECT_BEST_EPOCH_BY_AUPRC else epoch
        if (
            (SELECT_BEST_EPOCH_BY_AUPRC and score > best_score)
            or (not SELECT_BEST_EPOCH_BY_AUPRC)
        ):
            best_score = score
            best_epoch = epoch
            best_y = y_val.copy()
            best_prob = prob_val.copy()
            best_idx = local_idx_val.copy()

        print(
            f"{tail_pct}% | fold {fold} | epoch {epoch}: "
            f"AUROC={auroc:.4f}, AUPRC={auprc:.4f}"
        )

    out = metrics(best_y, best_prob)
    out.update({
        "Tail_Percent": tail_pct,
        "Fold": fold,
        "Best_Epoch": best_epoch,
        "Train_N": len(train_idx),
        "Val_N": len(val_idx),
        "Train_Positive_N": int((labels[train_idx] == 1).sum()),
        "Train_Negative_N": int((labels[train_idx] == 0).sum()),
        "Val_Positive_N": int((labels[val_idx] == 1).sum()),
        "Val_Negative_N": int((labels[val_idx] == 0).sum()),
    })

    del model, train_loader, val_loader
    gc.collect()
    if USE_GPU:
        torch.cuda.empty_cache()

    return out, best_y, best_prob, best_idx, pd.DataFrame(epoch_rows)


# ============================================================
# MAIN SENSITIVITY RUN
# ============================================================

def run_training(quick=False):
    cache = load_cache()
    _, _, scgpt_matrix = load_reference()

    thresholds = [0.20] if quick else TAIL_FRACTIONS

    fold_rows = []
    oof_metric_rows = []
    epoch_frames = []
    dataset_rows = []

    for t in thresholds:
        tail_pct = int(round(t * 100))
        master_indices, labels, q_low, q_high = make_threshold_dataset(cache, t)

        dataset_rows.append({
            "Tail_Percent": tail_pct,
            "Low_CorrectedP_Cutoff": q_low,
            "High_CorrectedP_Cutoff": q_high,
            "Positive_N": int((labels == 1).sum()),
            "Negative_N": int((labels == 0).sum()),
            "Total_N": int(len(labels)),
            "Positive_Ratio": float(labels.mean()),
        })

        print("\n" + "=" * 90)
        print(
            f"Tail={tail_pct}% | N={len(labels):,} | "
            f"positive={(labels==1).sum():,} | negative={(labels==0).sum():,}"
        )
        print(f"Strong: corrected P <= {q_low:.8g}")
        print(f"Weak:   corrected P >= {q_high:.8g}")
        print("=" * 90)

        dataset = SensitivityDataset(cache, master_indices, labels)

        n_splits = 2 if quick else N_SPLITS
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        oof_prob = np.full(len(labels), np.nan, dtype=np.float32)
        oof_fold = np.zeros(len(labels), dtype=np.int16)

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(np.zeros(len(labels)), labels), start=1
        ):
            fold_metric, y_val, prob, local_idx, epoch_df = train_fold(
                dataset, labels, train_idx, val_idx,
                scgpt_matrix, tail_pct, fold, quick
            )

            fold_rows.append(fold_metric)
            epoch_frames.append(epoch_df)

            oof_prob[local_idx] = prob
            oof_fold[local_idx] = fold

        valid = ~np.isnan(oof_prob)
        oof_metric = metrics(labels[valid], oof_prob[valid])
        oof_metric.update({
            "Tail_Percent": tail_pct,
            "OOF_N": int(valid.sum()),
        })
        oof_metric_rows.append(oof_metric)

        pd.DataFrame({
            "Tail_Percent": tail_pct,
            "Local_Index": np.arange(len(labels)),
            "Fold": oof_fold,
            "True_Label": labels,
            "Pred_Prob_Strong": oof_prob,
            "Pred_Label_0p5": (oof_prob >= FIXED_THRESHOLD).astype(int),
            "Master_Index": master_indices,
            "Source_Row": np.asarray(cache["source_row"])[master_indices],
            "CorrectedPValue": np.asarray(cache["p"])[master_indices],
        }).to_csv(
            os.path.join(RESULT_DIR, f"oof_predictions_tail_{tail_pct}.csv"),
            index=False,
        )

        print(
            f"OOF {tail_pct}%: AUROC={oof_metric['AUROC']:.4f}, "
            f"AUPRC={oof_metric['AUPRC']:.4f}, "
            f"F1={oof_metric['F1']:.4f}, MCC={oof_metric['MCC']:.4f}"
        )

        del dataset
        gc.collect()

    dataset_df = pd.DataFrame(dataset_rows)
    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_metric_rows)
    epoch_df = pd.concat(epoch_frames, ignore_index=True) if epoch_frames else pd.DataFrame()

    dataset_df.to_csv(os.path.join(RESULT_DIR, "threshold_dataset_summary.csv"), index=False)
    fold_df.to_csv(os.path.join(RESULT_DIR, "threshold_fold_metrics.csv"), index=False)
    oof_df.to_csv(os.path.join(RESULT_DIR, "threshold_oof_metrics.csv"), index=False)
    epoch_df.to_csv(os.path.join(RESULT_DIR, "threshold_epoch_log.csv"), index=False)

    # Mean ± SD across folds
    report_metrics = [
        "AUROC", "AUPRC", "Accuracy", "Balanced_Accuracy",
        "Precision", "Recall", "Specificity", "F1", "MCC",
        "Cohen_Kappa", "Brier_Score", "Log_Loss", "Best_Epoch",
    ]

    summary_rows = []

    for tail_pct, g in fold_df.groupby("Tail_Percent"):
        row = {"Tail_Percent": int(tail_pct), "Fold_N": int(len(g))}
        for metric in report_metrics:
            vals = pd.to_numeric(g[metric], errors="coerce")
            row[f"{metric}_Mean"] = float(vals.mean())
            row[f"{metric}_SD"] = float(vals.std(ddof=1))
        summary_rows.append(row)

    summary = dataset_df.merge(
        pd.DataFrame(summary_rows),
        on="Tail_Percent",
        how="left",
    )

    summary = summary.merge(
        oof_df.add_prefix("OOF_").rename(columns={"OOF_Tail_Percent": "Tail_Percent"}),
        on="Tail_Percent",
        how="left",
    )

    summary.to_csv(
        os.path.join(RESULT_DIR, "threshold_sensitivity_summary.csv"),
        index=False,
    )

    # Plot-ready table
    plot_rows = []
    for _, row in summary.iterrows():
        for metric in ["AUROC", "AUPRC", "F1", "MCC"]:
            plot_rows.append({
                "Tail_Percent": int(row["Tail_Percent"]),
                "Metric": metric,
                "Mean": row[f"{metric}_Mean"],
                "SD": row[f"{metric}_SD"],
            })
    pd.DataFrame(plot_rows).to_csv(
        os.path.join(RESULT_DIR, "threshold_sensitivity_plot_data.csv"),
        index=False,
    )

    # Absolute and relative change versus the original 20% choice.
    if not quick and 20 in set(summary["Tail_Percent"]):
        ref = summary.loc[summary["Tail_Percent"] == 20].iloc[0]
        change_rows = []

        for _, row in summary.iterrows():
            out = {"Tail_Percent": int(row["Tail_Percent"])}
            for metric in ["AUROC", "AUPRC", "F1", "MCC"]:
                cur = float(row[f"{metric}_Mean"])
                base = float(ref[f"{metric}_Mean"])
                out[f"{metric}_Delta_vs_20pct"] = cur - base
                out[f"{metric}_Percent_Change_vs_20pct"] = (
                    100.0 * (cur - base) / abs(base) if base != 0 else np.nan
                )
            change_rows.append(out)

        pd.DataFrame(change_rows).to_csv(
            os.path.join(RESULT_DIR, "threshold_change_vs_20pct.csv"),
            index=False,
        )

    print("\nSensitivity analysis finished.")
    print(summary.to_string(index=False))
    print(f"\nResults saved in: {RESULT_DIR}")


# ============================================================
# CLI
# ============================================================

def save_config(stage, quick):
    os.makedirs(RESULT_DIR, exist_ok=True)

    cfg = {
        "source_csv": SOURCE_CSV,
        "reference_dir": REFERENCE_DIR,
        "tail_fractions": TAIL_FRACTIONS,
        "label_definition": {
            "positive_strong": "smallest corrected-P-value tail",
            "negative_weak": "largest corrected-P-value tail",
            "middle": "excluded",
        },
        "model": {
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_queries": NUM_QUERIES,
            "num_qformer_layers": NUM_QFORMER_LAYERS,
            "dropout_rate": DROPOUT_RATE,
            "freeze_gene_embedding": FREEZE_GENE_EMBEDDING,
            "use_mean_residual": USE_MEAN_RESIDUAL,
        },
        "cv": {
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "select_best_epoch_by_auprc": SELECT_BEST_EPOCH_BY_AUPRC,
        },
        "training": {
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "val_batch_size": VAL_BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "eta_min": ETA_MIN,
        },
        "stage": stage,
        "quick": quick,
        "device": str(DEVICE),
    }

    with open(
        os.path.join(RESULT_DIR, "threshold_sensitivity_config.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "train"],
        default="all",
    )
    parser.add_argument(
        "--force_prepare",
        action="store_true",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="20% only, 2 folds, 2 epochs.",
    )
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    set_seed(RANDOM_STATE)
    save_config(args.stage, args.quick)

    print("Device:", DEVICE)

    if args.stage in {"all", "prepare"}:
        prepare(force=args.force_prepare)

    if args.stage in {"all", "train"}:
        run_training(quick=args.quick)


if __name__ == "__main__":
    main()
