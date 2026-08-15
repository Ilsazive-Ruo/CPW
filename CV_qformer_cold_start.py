import os
import json
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
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
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from QFormer import QFormerClassifier

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================
DATA_DIR = "CLSData2"
RESULT_DIR = "QFormer_ColdStart_Results"

# preprocess_classification.py generated this file while preserving row order.
LABELED_CSV_PATH = os.path.join(
    DATA_DIR,
    "interaction_classification_labeled.csv",
)

N_SPLITS = 5
RANDOM_STATE = 42
FIXED_THRESHOLD = 0.5

# Use a fixed epoch number selected beforehand from the previous CV.
# Do not select epochs using the outer cold-start test fold.
EPOCHS = 15
BATCH_SIZE_TRAIN = 2048
BATCH_SIZE_TEST = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ETA_MIN = 1e-5
NUM_WORKERS = 8

# Q-Former architecture: keep identical to the original model.
EMBED_DIM = 512
NUM_HEADS = 8
NUM_QUERIES = 8
NUM_QFORMER_LAYERS = 2
DROPOUT_RATE = 0.1
NUM_CLASSES = 2
FREEZE_GENE_EMBEDDING = True
USE_MEAN_RESIDUAL = True

USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

SPLIT_SCHEMES = [
    "random",
    "drug_identity_cold",
    "scaffold_cold",
    "pathway_cold",
    "double_cold",
]

# For cold-start comparison alone, keep only original.
# Uncomment the masks to combine cold-start and feature-ablation analysis.
ABLATION_MODES = [
    "original",
    # "mask_molecule",
    # "mask_pathway_genes",
    # "mask_both",
]

SAVE_FOLD_MODELS = False
MIN_DOUBLE_COLD_TEST_SAMPLES = 20


# ============================================================
# Utilities
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def normalized_name(value):
    return (
        str(value).strip().lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def find_column(df, candidates, required=True):
    mapping = {normalized_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalized_name(candidate)
        if key in mapping:
            return mapping[key]

    if required:
        raise ValueError(
            f"Cannot find column from {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def safe_divide(a, b):
    return float(a / b) if b != 0 else np.nan


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "AUROC": (
            roc_auc_score(y_true, y_prob)
            if len(np.unique(y_true)) == 2 else np.nan
        ),
        "AUPRC": (
            average_precision_score(y_true, y_prob)
            if len(np.unique(y_true)) == 2 else np.nan
        ),
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
        "Log_Loss": log_loss(
            y_true,
            np.clip(y_prob, 1e-7, 1 - 1e-7),
            labels=[0, 1],
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Prevalence": float(np.mean(y_true)),
        "Predicted_Positive_Rate": float(np.mean(y_pred)),
    }


# ============================================================
# Molecular grouping
# ============================================================
def canonicalize_smiles(smiles):
    raw = "" if pd.isna(smiles) else str(smiles).strip()
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return f"INVALID::{raw}"
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def smiles_to_scaffold(smiles):
    """Bemis-Murcko scaffold used as the molecular cold-start group."""
    raw = "" if pd.isna(smiles) else str(smiles).strip()
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return f"INVALID::{raw}"

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False,
    )

    # Acyclic molecules have an empty Murcko scaffold. Keep them separated by
    # canonical structure rather than placing all acyclic compounds in one group.
    if scaffold == "":
        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=False,
        )
        return f"ACYCLIC::{canonical}"

    return scaffold


# ============================================================
# Data loading and row alignment
# ============================================================
def load_entities_from_labeled_csv(labels):
    if not os.path.exists(LABELED_CSV_PATH):
        raise FileNotFoundError(
            f"Missing {LABELED_CSV_PATH}. It is needed to recover SMILES and PathwayID."
        )

    df = pd.read_csv(LABELED_CSV_PATH)

    class_col = find_column(df, ["class_label", "ClassLabel", "label"])
    smiles_col = find_column(df, ["SMILES", "smiles"])
    pathway_id_col = find_column(
        df,
        ["PathwayID", "Pathway_ID", "pathway_id", "TermID", "term_id"],
        required=False,
    )
    pathway_name_col = find_column(
        df,
        ["PathwayName", "Pathway_Name", "pathway_name", "Matched_Pathway"],
        required=False,
    )
    drug_id_col = find_column(
        df,
        ["ChemicalID", "Chemical_ID", "chemical_id", "CID", "MoleculeID"],
        required=False,
    )
    drug_name_col = find_column(
        df,
        ["ChemicalName", "Chemical_Name", "chemical_name", "MoleculeName"],
        required=False,
    )

    cls_df = df[df[class_col].isin([0, 1])].copy().reset_index(drop=True)
    cls_df[class_col] = cls_df[class_col].astype(np.int64)

    if len(cls_df) != len(labels):
        raise ValueError(
            "Filtered labeled CSV and cls_labels.npy have different lengths: "
            f"{len(cls_df)} vs {len(labels)}"
        )

    if not np.array_equal(cls_df[class_col].to_numpy(), labels):
        mismatch = int(np.sum(cls_df[class_col].to_numpy() != labels))
        raise ValueError(
            f"Labeled CSV order does not match cls_labels.npy; mismatches={mismatch}"
        )

    smiles = cls_df[smiles_col].fillna("").astype(str).to_numpy()

    if pathway_id_col is not None:
        pathway = cls_df[pathway_id_col].fillna("").astype(str).to_numpy()
    elif pathway_name_col is not None:
        pathway = cls_df[pathway_name_col].fillna("").astype(str).to_numpy()
    else:
        raise ValueError("Neither PathwayID nor PathwayName is present.")

    if drug_id_col is not None:
        drug_identity = cls_df[drug_id_col].fillna("").astype(str).to_numpy()
    elif drug_name_col is not None:
        drug_identity = cls_df[drug_name_col].fillna("").astype(str).to_numpy()
    else:
        drug_identity = np.asarray(
            [canonicalize_smiles(s) for s in smiles], dtype=object
        )

    pathway = np.asarray([
        p if str(p).strip() else f"MISSING_PATHWAY::{i}"
        for i, p in enumerate(pathway)
    ], dtype=object)

    drug_identity = np.asarray([
        d if str(d).strip() else canonicalize_smiles(smiles[i])
        for i, d in enumerate(drug_identity)
    ], dtype=object)

    return smiles, drug_identity, pathway


def load_data():
    fps_path = os.path.join(DATA_DIR, "cls_morgan_fps.npy")
    labels_path = os.path.join(DATA_DIR, "cls_labels.npy")
    meta_path = os.path.join(DATA_DIR, "cls_gene_metadata.pkl")
    scgpt_path = os.path.join(DATA_DIR, "scgpt_matrix.pt")

    for path in [fps_path, labels_path, meta_path, scgpt_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    labels = np.load(labels_path).astype(np.int64)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    gene_ids = meta["gene_ids"]
    scgpt_matrix = torch.load(scgpt_path, map_location="cpu")

    if len(labels) != len(gene_ids):
        raise ValueError(
            f"labels={len(labels)} and gene_ids={len(gene_ids)} do not match"
        )

    # Prefer metadata fields if they were saved; otherwise recover them from the
    # labeled CSV generated by preprocessing.
    smiles = None
    pathway = None
    drug_identity = None

    for key in ["smiles", "SMILES", "canonical_smiles"]:
        if key in meta:
            smiles = np.asarray(meta[key], dtype=object)
            break

    for key in [
        "pathway_ids", "pathway_id", "pathway_names", "pathway_name",
        "pathways", "pathway", "term_ids", "term_id",
    ]:
        if key in meta:
            pathway = np.asarray(meta[key], dtype=object)
            break

    for key in [
        "drug_ids", "drug_id", "chemical_ids", "chemical_id",
        "molecule_ids", "molecule_id", "drug_names", "chemical_names",
    ]:
        if key in meta:
            drug_identity = np.asarray(meta[key], dtype=object)
            break

    if (
        smiles is None or pathway is None
        or len(smiles) != len(labels)
        or len(pathway) != len(labels)
    ):
        smiles_csv, drug_csv, pathway_csv = load_entities_from_labeled_csv(labels)
        smiles = smiles_csv
        pathway = pathway_csv
        if drug_identity is None or len(drug_identity) != len(labels):
            drug_identity = drug_csv

    if drug_identity is None or len(drug_identity) != len(labels):
        drug_identity = np.asarray(
            [canonicalize_smiles(s) for s in smiles], dtype=object
        )

    canonical_groups = np.asarray(
        [canonicalize_smiles(s) for s in smiles], dtype=object
    )
    scaffold_groups = np.asarray(
        [smiles_to_scaffold(s) for s in smiles], dtype=object
    )

    print("========== Dataset summary ==========")
    print(f"Samples              : {len(labels)}")
    print(f"Low / High           : {(labels == 0).sum()} / {(labels == 1).sum()}")
    print(f"Unique drug identities: {len(np.unique(drug_identity))}")
    print(f"Unique canonical drugs: {len(np.unique(canonical_groups))}")
    print(f"Unique scaffolds       : {len(np.unique(scaffold_groups))}")
    print(f"Unique pathways        : {len(np.unique(pathway))}")
    print(f"scGPT shape            : {tuple(scgpt_matrix.shape)}")
    print("=====================================")

    entity_df = pd.DataFrame({
        "Sample_Index": np.arange(len(labels)),
        "True_Label": labels,
        "SMILES": smiles,
        "Drug_Identity": drug_identity,
        "Canonical_SMILES_Group": canonical_groups,
        "Murcko_Scaffold_Group": scaffold_groups,
        "Pathway_Group": pathway,
    })

    return {
        "fps_path": fps_path,
        "labels_path": labels_path,
        "labels": labels,
        "gene_ids": gene_ids,
        "scgpt_matrix": scgpt_matrix,
        "drug_identity": np.asarray(drug_identity, dtype=object),
        "scaffold_groups": scaffold_groups,
        "pathway_groups": np.asarray(pathway, dtype=object),
        "entity_df": entity_df,
    }


# ============================================================
# Dataset
# ============================================================
class ColdStartDataset(Dataset):
    def __init__(self, fps_path, labels_path, gene_ids, ablation_mode="original"):
        valid = {
            "original", "mask_molecule", "mask_pathway_genes", "mask_both"
        }
        if ablation_mode not in valid:
            raise ValueError(f"Unknown ablation mode: {ablation_mode}")

        self.fps_path = fps_path
        self.labels = np.load(labels_path, mmap_mode="c")
        self.gene_ids = gene_ids
        self.ablation_mode = ablation_mode
        self.fps = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.fps is None:
            self.fps = np.load(self.fps_path, mmap_mode="c")

        if self.ablation_mode in {"mask_molecule", "mask_both"}:
            fp_np = np.zeros_like(self.fps[idx], dtype=np.float32)
        else:
            fp_np = np.asarray(self.fps[idx], dtype=np.float32)

        if self.ablation_mode in {"mask_pathway_genes", "mask_both"}:
            gene_id = torch.tensor([0], dtype=torch.long)
            gene_count = torch.tensor([0.0], dtype=torch.float32)
        else:
            gids = [int(g) for g in self.gene_ids[idx]]
            if len(gids) == 0:
                gids = [0]
            gene_id = torch.tensor(gids, dtype=torch.long)
            gene_count = torch.tensor([np.log1p(len(gids))], dtype=torch.float32)

        return (
            torch.from_numpy(fp_np).float(),
            gene_id,
            gene_count,
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
        )


def collate_fn(batch):
    fps, gene_ids_list, gene_counts, labels = zip(*batch)
    padded = pad_sequence(gene_ids_list, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(x) for x in gene_ids_list], dtype=torch.long)
    max_len = padded.size(1)
    mask = torch.arange(max_len)[None, :] >= lengths[:, None]

    return (
        torch.stack(fps),
        padded,
        mask,
        torch.stack(gene_counts),
        torch.stack(labels),
    )


# ============================================================
# Split construction
# ============================================================
def check_group_number(groups, name):
    n = len(np.unique(groups))
    if n < N_SPLITS:
        raise ValueError(
            f"{name} has only {n} groups, fewer than N_SPLITS={N_SPLITS}"
        )


def assign_group_folds(labels, groups, seed):
    check_group_number(groups, "group assignment")
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    fold_ids = np.full(len(labels), -1, dtype=np.int64)
    for fold, (_, test_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups=groups)
    ):
        fold_ids[test_idx] = fold

    if np.any(fold_ids < 0):
        raise RuntimeError("Some samples did not receive a fold assignment")
    return fold_ids


def build_splits(split_scheme, labels, drug_groups, scaffold_groups, pathway_groups):
    rows = []

    if split_scheme == "random":
        splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        iterator = splitter.split(np.zeros(len(labels)), labels)
        for fold, (train_idx, test_idx) in enumerate(iterator, start=1):
            rows.append((fold, train_idx, test_idx, np.array([], dtype=np.int64)))
        return rows

    group_dict = {
        "drug_identity_cold": drug_groups,
        "scaffold_cold": scaffold_groups,
        "pathway_cold": pathway_groups,
    }

    if split_scheme in group_dict:
        groups = group_dict[split_scheme]
        check_group_number(groups, split_scheme)
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        iterator = splitter.split(np.zeros(len(labels)), labels, groups=groups)
        for fold, (train_idx, test_idx) in enumerate(iterator, start=1):
            rows.append((fold, train_idx, test_idx, np.array([], dtype=np.int64)))
        return rows

    if split_scheme == "double_cold":
        scaffold_fold = assign_group_folds(
            labels, scaffold_groups, RANDOM_STATE
        )
        pathway_fold = assign_group_folds(
            labels, pathway_groups, RANDOM_STATE + 97
        )

        for k in range(N_SPLITS):
            test_mask = (scaffold_fold == k) & (pathway_fold == k)
            train_mask = (scaffold_fold != k) & (pathway_fold != k)
            excluded_mask = ~(train_mask | test_mask)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            excluded_idx = np.where(excluded_mask)[0]

            if len(test_idx) < MIN_DOUBLE_COLD_TEST_SAMPLES:
                print(
                    f"⚠️ double_cold fold {k + 1}: only {len(test_idx)} test samples"
                )

            rows.append((k + 1, train_idx, test_idx, excluded_idx))
        return rows

    raise ValueError(f"Unknown split scheme: {split_scheme}")


def overlap_count(values, train_idx, test_idx):
    return len(set(values[train_idx]).intersection(set(values[test_idx])))


def audit_split(split_scheme, train_idx, test_idx, drug_groups, scaffold_groups, pathway_groups):
    audit = {
        "Drug_Identity_Overlap_N": overlap_count(drug_groups, train_idx, test_idx),
        "Scaffold_Overlap_N": overlap_count(scaffold_groups, train_idx, test_idx),
        "Pathway_Overlap_N": overlap_count(pathway_groups, train_idx, test_idx),
    }

    if split_scheme == "drug_identity_cold" and audit["Drug_Identity_Overlap_N"] != 0:
        raise RuntimeError("Drug identity leakage detected")
    if split_scheme == "scaffold_cold" and audit["Scaffold_Overlap_N"] != 0:
        raise RuntimeError("Scaffold leakage detected")
    if split_scheme == "pathway_cold" and audit["Pathway_Overlap_N"] != 0:
        raise RuntimeError("Pathway leakage detected")
    if split_scheme == "double_cold":
        if audit["Scaffold_Overlap_N"] != 0:
            raise RuntimeError("Scaffold leakage in double cold-start")
        if audit["Pathway_Overlap_N"] != 0:
            raise RuntimeError("Pathway leakage in double cold-start")

    return audit


# ============================================================
# Model training
# ============================================================
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


def evaluate(model, loader):
    model.eval()
    y_all = []
    p_all = []

    with torch.no_grad():
        for fps, gene_ids, mask, gene_counts, labels in loader:
            fps = fps.to(DEVICE, non_blocking=True)
            gene_ids = gene_ids.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            gene_counts = gene_counts.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=USE_GPU
            ):
                logits, _ = model(fps, gene_ids, mask, gene_counts)

            prob = torch.softmax(logits.float(), dim=1)[:, 1]
            p_all.append(prob.cpu().numpy())
            y_all.append(labels.numpy())

    return (
        np.concatenate(y_all).astype(np.int64),
        np.concatenate(p_all).astype(np.float32),
    )


def train_fold(split_scheme, ablation, fold, train_idx, test_idx, dataset, labels, scgpt_matrix):
    train_counts = np.bincount(labels[train_idx], minlength=2)
    test_counts = np.bincount(labels[test_idx], minlength=2)

    if np.any(train_counts == 0):
        raise RuntimeError(
            f"Training set has an empty class: {train_counts.tolist()}"
        )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_GPU,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_GPU,
        collate_fn=collate_fn,
    )

    set_seed(RANDOM_STATE + fold * 1009)
    raw_model, model, compiled = build_model(scgpt_matrix)
    print(f"torch.compile: {'enabled' if compiled else 'disabled'}")

    class_weights = train_counts.sum() / (2.0 * train_counts)
    class_weights_t = torch.tensor(
        class_weights, dtype=torch.float32, device=DEVICE
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=ETA_MIN
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_GPU)

    epoch_rows = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total_seen = 0

        pbar = tqdm(
            train_loader,
            desc=(
                f"{split_scheme} | {ablation} | Fold {fold} | "
                f"Epoch {epoch + 1}/{EPOCHS}"
            ),
            leave=False,
        )

        for fps, gene_ids, mask, gene_counts, batch_labels in pbar:
            fps = fps.to(DEVICE, non_blocking=True)
            gene_ids = gene_ids.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            gene_counts = gene_counts.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=USE_GPU
            ):
                logits, _ = model(fps, gene_ids, mask, gene_counts)
                loss = criterion(logits, batch_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = batch_labels.size(0)
            running_loss += loss.item() * bs
            total_seen += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        epoch_loss = running_loss / max(1, total_seen)
        epoch_rows.append({
            "Split_Scheme": split_scheme,
            "Ablation": ablation,
            "Fold": fold,
            "Epoch": epoch + 1,
            "Train_Loss": epoch_loss,
            "LR": optimizer.param_groups[0]["lr"],
        })
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | Loss={epoch_loss:.5f} | "
            f"LR={optimizer.param_groups[0]['lr']:.7f}"
        )

    # Evaluate the outer test fold only once after fixed-epoch training.
    y_test, test_prob = evaluate(model, test_loader)
    metrics = compute_metrics(y_test, test_prob, FIXED_THRESHOLD)
    metrics.update({
        "Split_Scheme": split_scheme,
        "Ablation": ablation,
        "Fold": fold,
        "Train_N": int(len(train_idx)),
        "Test_N": int(len(test_idx)),
        "Train_Low_N": int(train_counts[0]),
        "Train_High_N": int(train_counts[1]),
        "Test_Low_N": int(test_counts[0]),
        "Test_High_N": int(test_counts[1]),
        "Train_High_Ratio": float(labels[train_idx].mean()),
        "Test_High_Ratio": float(labels[test_idx].mean()),
    })

    pred_df = pd.DataFrame({
        "Sample_Index": test_idx,
        "Split_Scheme": split_scheme,
        "Ablation": ablation,
        "Fold": fold,
        "True_Label": y_test,
        "Pred_Prob_High": test_prob,
        "Pred_Label_0p5": (test_prob >= FIXED_THRESHOLD).astype(np.int64),
    })

    if SAVE_FOLD_MODELS:
        model_dir = os.path.join(RESULT_DIR, "fold_models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(
            model_dir,
            f"qformer_{split_scheme}_{ablation}_fold{fold}.pt",
        )
        torch.save(raw_model.state_dict(), model_path)
        metrics["Saved_Model_Path"] = model_path

    del model, raw_model
    if USE_GPU:
        torch.cuda.empty_cache()

    return metrics, pred_df, pd.DataFrame(epoch_rows)


# ============================================================
# Main
# ============================================================
def run():
    os.makedirs(RESULT_DIR, exist_ok=True)
    set_seed(RANDOM_STATE)

    data = load_data()
    labels = data["labels"]
    drug_groups = data["drug_identity"]
    scaffold_groups = data["scaffold_groups"]
    pathway_groups = data["pathway_groups"]

    data["entity_df"].to_csv(
        os.path.join(RESULT_DIR, "sample_entity_groups.csv"),
        index=False,
    )

    fold_rows = []
    pred_frames = []
    epoch_frames = []
    audit_rows = []

    for ablation in ABLATION_MODES:
        dataset = ColdStartDataset(
            data["fps_path"],
            data["labels_path"],
            data["gene_ids"],
            ablation_mode=ablation,
        )

        for split_scheme in SPLIT_SCHEMES:
            print("\n" + "#" * 100)
            print(f"# Split={split_scheme} | Ablation={ablation}")
            print("#" * 100)

            splits = build_splits(
                split_scheme,
                labels,
                drug_groups,
                scaffold_groups,
                pathway_groups,
            )

            for fold, train_idx, test_idx, excluded_idx in splits:
                print("\n" + "=" * 100)
                print(f"{split_scheme} | {ablation} | Fold {fold}/{N_SPLITS}")
                print("=" * 100)
                print(f"Train={len(train_idx)} | Test={len(test_idx)} | Excluded={len(excluded_idx)}")

                if len(test_idx) == 0:
                    print("⚠️ Empty test set; skipped")
                    continue

                audit = audit_split(
                    split_scheme,
                    train_idx,
                    test_idx,
                    drug_groups,
                    scaffold_groups,
                    pathway_groups,
                )

                audit_rows.append({
                    "Split_Scheme": split_scheme,
                    "Ablation": ablation,
                    "Fold": fold,
                    "Train_N": len(train_idx),
                    "Test_N": len(test_idx),
                    "Excluded_N": len(excluded_idx),
                    "Train_Drug_N": len(np.unique(drug_groups[train_idx])),
                    "Test_Drug_N": len(np.unique(drug_groups[test_idx])),
                    "Train_Scaffold_N": len(np.unique(scaffold_groups[train_idx])),
                    "Test_Scaffold_N": len(np.unique(scaffold_groups[test_idx])),
                    "Train_Pathway_N": len(np.unique(pathway_groups[train_idx])),
                    "Test_Pathway_N": len(np.unique(pathway_groups[test_idx])),
                    **audit,
                })

                metrics, pred_df, epoch_df = train_fold(
                    split_scheme,
                    ablation,
                    fold,
                    train_idx,
                    test_idx,
                    dataset,
                    labels,
                    data["scgpt_matrix"],
                )
                metrics["Excluded_N"] = int(len(excluded_idx))
                metrics.update(audit)

                fold_rows.append(metrics)
                pred_frames.append(pred_df)
                epoch_frames.append(epoch_df)

                print(
                    f"AUROC={metrics['AUROC']:.4f} | "
                    f"AUPRC={metrics['AUPRC']:.4f} | "
                    f"F1={metrics['F1']:.4f} | MCC={metrics['MCC']:.4f}"
                )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(
        os.path.join(RESULT_DIR, "qformer_cold_start_fold_metrics.csv"),
        index=False,
    )

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(
        os.path.join(RESULT_DIR, "qformer_cold_start_split_audit.csv"),
        index=False,
    )

    if epoch_frames:
        pd.concat(epoch_frames, ignore_index=True).to_csv(
            os.path.join(RESULT_DIR, "qformer_cold_start_epoch_log.csv"),
            index=False,
        )

    if pred_frames:
        pred_df = pd.concat(pred_frames, ignore_index=True)
        pred_df = pred_df.merge(
            data["entity_df"],
            on=["Sample_Index", "True_Label"],
            how="left",
        )
    else:
        pred_df = pd.DataFrame()

    pred_df.to_csv(
        os.path.join(RESULT_DIR, "qformer_cold_start_predictions_long.csv"),
        index=False,
    )

    # Aggregate all held-out predictions for each split scheme.
    aggregate_rows = []
    if not pred_df.empty:
        for (split_scheme, ablation), sub in pred_df.groupby(
            ["Split_Scheme", "Ablation"]
        ):
            metrics = compute_metrics(
                sub["True_Label"].to_numpy(),
                sub["Pred_Prob_High"].to_numpy(),
                FIXED_THRESHOLD,
            )
            metrics.update({
                "Split_Scheme": split_scheme,
                "Ablation": ablation,
                "Evaluated_N": int(len(sub)),
                "Unique_Evaluated_N": int(sub["Sample_Index"].nunique()),
                "Dataset_N": int(len(labels)),
                "Coverage_Ratio": float(sub["Sample_Index"].nunique() / len(labels)),
            })
            aggregate_rows.append(metrics)

    pd.DataFrame(aggregate_rows).to_csv(
        os.path.join(RESULT_DIR, "qformer_cold_start_aggregate_metrics.csv"),
        index=False,
    )

    # Mean ± SD across folds.
    if not fold_df.empty:
        numeric_cols = fold_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "Fold"]
        mean_df = fold_df.groupby(["Split_Scheme", "Ablation"])[numeric_cols].mean()
        std_df = fold_df.groupby(["Split_Scheme", "Ablation"])[numeric_cols].std()
        summary_df = mean_df.copy()
        for col in numeric_cols:
            summary_df[f"{col}_std"] = std_df[col]
        summary_df = summary_df.reset_index()
    else:
        summary_df = pd.DataFrame()

    summary_df.to_csv(
        os.path.join(RESULT_DIR, "qformer_cold_start_summary_mean_std.csv"),
        index=False,
    )

    config = {
        "data_dir": DATA_DIR,
        "result_dir": RESULT_DIR,
        "labeled_csv_path": LABELED_CSV_PATH,
        "split_schemes": SPLIT_SCHEMES,
        "ablation_modes": ABLATION_MODES,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "fixed_epochs": EPOCHS,
        "fixed_threshold": FIXED_THRESHOLD,
        "outer_test_policy": (
            "The outer cold-start test fold is evaluated only once after fixed-epoch training."
        ),
        "double_cold_definition": {
            "train": "scaffold fold != k AND pathway fold != k",
            "test": "scaffold fold == k AND pathway fold == k",
            "excluded": "only one side belongs to fold k",
        },
        "model": {
            "class": "QFormerClassifier",
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_queries": NUM_QUERIES,
            "num_qformer_layers": NUM_QFORMER_LAYERS,
            "dropout_rate": DROPOUT_RATE,
            "freeze_gene_embedding": FREEZE_GENE_EMBEDDING,
            "use_mean_residual": USE_MEAN_RESIDUAL,
        },
    }

    with open(
        os.path.join(RESULT_DIR, "qformer_cold_start_config.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print("\n✅ Cold-start comparison completed")
    print(f"Results saved to: {RESULT_DIR}")


if __name__ == "__main__":
    run()
