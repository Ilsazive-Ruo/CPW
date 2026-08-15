import os
import json
import pickle
import argparse
import warnings
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

from QFormer import QFormerClassifier

warnings.filterwarnings("ignore")


# -------------------------
# Defaults matching your QFormer training code
# -------------------------
DEFAULT_DATA_DIR = "CLSData2"
DEFAULT_CHECKPOINT_PATH = "best_finetuned_qformer_checkpoint.pt"
DEFAULT_OUTPUT_DIR = "QFormer_Predictions"

DEFAULT_EMBED_DIM = 512
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_QUERIES = 8
DEFAULT_NUM_QFORMER_LAYERS = 2
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_NUM_CLASSES = 2
DEFAULT_FREEZE_GENE_EMBEDDING = True
DEFAULT_USE_MEAN_RESIDUAL = True

DEFAULT_BATCH_SIZE = 4096
DEFAULT_THRESHOLD = 0.5
DEFAULT_NUM_WORKERS = 0
DEFAULT_USE_PROVIDED_GENE_COUNT = True

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smilestofp(smiles: str) -> np.ndarray:
    """Convert one SMILES to Morgan fingerprint."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            fp_bit = morgan_gen.GetFingerprint(mol)
            arr = np.zeros((2048,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp_bit, arr)
            return arr.astype(np.float32)
    except Exception:
        pass
    return np.zeros(2048, dtype=np.float32)


def split_gene_string(gene_string: str) -> List[str]:
    """Support comma-separated and semicolon-separated gene strings."""
    if pd.isna(gene_string):
        return []
    text = str(gene_string).strip()
    if not text:
        return []

    # The new pathway CSV uses comma, while previous datasets used semicolon.
    # Support both, including mixed delimiters.
    text = text.replace(";", ",")
    genes = [g.strip() for g in text.split(",") if g.strip()]
    return genes


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def load_gene_vocab(data_dir: str) -> Dict[str, int]:
    meta_path = os.path.join(data_dir, "cls_gene_metadata.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cannot find metadata file: {meta_path}")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    if "vocab" not in meta:
        raise KeyError(
            f"{meta_path} does not contain key 'vocab'. "
            "Prediction needs the original training gene vocab to map pathway genes."
        )

    return meta["vocab"]


def load_scgpt_matrix(data_dir: str) -> torch.Tensor:
    scgpt_path = os.path.join(data_dir, "scgpt_matrix.pt")
    if not os.path.exists(scgpt_path):
        raise FileNotFoundError(f"Cannot find scGPT matrix: {scgpt_path}")
    return torch.load(scgpt_path, map_location="cpu")


def get_config_value(config: Dict[str, Any], keys: List[str], default: Any) -> Any:
    for k in keys:
        if k in config:
            return config[k]
    return default


def load_checkpoint_and_config(checkpoint_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Cannot find checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        config = ckpt.get("config", {})
    elif isinstance(ckpt, dict):
        # It may be a raw state_dict.
        state_dict = ckpt
        config = {}
    else:
        raise ValueError("Unsupported checkpoint format.")

    return state_dict, config


def build_model_from_checkpoint(
    scgpt_matrix: torch.Tensor,
    checkpoint_path: str,
    device: torch.device,
) -> QFormerClassifier:
    state_dict, config = load_checkpoint_and_config(checkpoint_path)

    embed_dim = int(get_config_value(config, ["embed_dim", "EMBED_DIM"], DEFAULT_EMBED_DIM))
    num_heads = int(get_config_value(config, ["num_heads", "NUM_HEADS"], DEFAULT_NUM_HEADS))
    num_queries = int(get_config_value(config, ["num_queries", "NUM_QUERIES"], DEFAULT_NUM_QUERIES))
    num_qformer_layers = int(get_config_value(config, ["num_qformer_layers", "NUM_QFORMER_LAYERS"], DEFAULT_NUM_QFORMER_LAYERS))
    dropout_rate = float(get_config_value(config, ["dropout_rate", "DROPOUT_RATE"], DEFAULT_DROPOUT_RATE))
    num_classes = int(get_config_value(config, ["num_classes", "NUM_CLASSES"], DEFAULT_NUM_CLASSES))
    freeze_gene_embedding = bool(get_config_value(config, ["freeze_gene_embedding", "FREEZE_GENE_EMBEDDING"], DEFAULT_FREEZE_GENE_EMBEDDING))
    use_mean_residual = bool(get_config_value(config, ["use_mean_residual", "USE_MEAN_RESIDUAL"], DEFAULT_USE_MEAN_RESIDUAL))

    model = QFormerClassifier(
        pretrained_scgpt_matrix=scgpt_matrix,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_queries=num_queries,
        num_qformer_layers=num_qformer_layers,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        freeze_gene_embedding=freeze_gene_embedding,
        use_mean_residual=use_mean_residual,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print("⚠️ Missing keys when loading checkpoint:")
        print(missing[:20], "..." if len(missing) > 20 else "")
    if len(unexpected) > 0:
        print("⚠️ Unexpected keys when loading checkpoint:")
        print(unexpected[:20], "..." if len(unexpected) > 20 else "")

    model.to(device)
    model.eval()
    return model


def read_molecule_csv(molecule_csv: str) -> pd.DataFrame:
    df = pd.read_csv(molecule_csv)
    if df.shape[1] < 2:
        raise ValueError("Molecule CSV must contain at least two columns: molecule name and SMILES.")

    name_col = df.columns[0]
    smiles_col = df.columns[1]

    out = df[[name_col, smiles_col]].copy()
    out.columns = ["Molecule_Name", "SMILES"]
    out["Molecule_Name"] = out["Molecule_Name"].astype(str)
    out["SMILES"] = out["SMILES"].astype(str)

    return out


def read_pathway_csv(pathway_csv: str, gene_vocab: Dict[str, int], use_provided_gene_count: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(pathway_csv)
    if df.shape[1] < 4:
        raise ValueError(
            "Pathway CSV must contain at least four columns: "
            "pathway name, pathway ID, genes, gene count."
        )

    pathway_name_col = df.columns[0]
    pathway_id_col = df.columns[1]
    genes_col = df.columns[2]
    gene_count_col = df.columns[3]

    records = []
    unseen_rows = []

    for i, row in df.iterrows():
        pathway_name = str(row[pathway_name_col])
        pathway_id = str(row[pathway_id_col])
        raw_genes = split_gene_string(row[genes_col])

        gene_ids = []
        unseen_genes = []
        for g in raw_genes:
            # Try exact, uppercase, and stripped forms.
            gid = gene_vocab.get(g)
            if gid is None:
                gid = gene_vocab.get(g.upper())
            if gid is None:
                gid = gene_vocab.get(g.strip())

            if gid is None:
                gene_ids.append(0)
                unseen_genes.append(g)
            else:
                gene_ids.append(int(gid))

        if len(gene_ids) == 0:
            gene_ids = [0]

        provided_count = safe_float(row[gene_count_col], default=np.nan)
        if use_provided_gene_count and not np.isnan(provided_count) and provided_count > 0:
            gene_count_value = float(provided_count)
        else:
            gene_count_value = float(len(raw_genes))

        records.append({
            "Pathway_Name": pathway_name,
            "Pathway_ID": pathway_id,
            "Raw_Genes": ",".join(raw_genes),
            "Provided_Gene_Count": provided_count,
            "Used_Gene_Count": gene_count_value,
            "Mapped_Gene_Count": int(sum([gid > 0 for gid in gene_ids])),
            "Unseen_Gene_Count": int(len(unseen_genes)),
            "Gene_IDs": gene_ids,
        })

        for g in unseen_genes:
            unseen_rows.append({
                "Pathway_Name": pathway_name,
                "Pathway_ID": pathway_id,
                "Unseen_Gene": g,
            })

    pathway_df = pd.DataFrame(records)
    unseen_df = pd.DataFrame(unseen_rows)
    return pathway_df, unseen_df


class MoleculePathwayPairDataset(Dataset):
    def __init__(self, molecule_df: pd.DataFrame, pathway_df: pd.DataFrame):
        self.molecule_df = molecule_df.reset_index(drop=True)
        self.pathway_df = pathway_df.reset_index(drop=True)
        self.n_mol = len(self.molecule_df)
        self.n_pathway = len(self.pathway_df)

        print("--> Building Morgan fingerprints for molecules...")
        self.fps = np.vstack([
            smilestofp(smi) for smi in tqdm(self.molecule_df["SMILES"].tolist())
        ]).astype(np.float32)

    def __len__(self):
        return self.n_mol * self.n_pathway

    def __getitem__(self, idx: int):
        mol_idx = idx // self.n_pathway
        pathway_idx = idx % self.n_pathway

        fp = torch.tensor(self.fps[mol_idx], dtype=torch.float32)

        row = self.pathway_df.iloc[pathway_idx]
        gene_ids = torch.tensor(row["Gene_IDs"], dtype=torch.long)
        gene_count = torch.tensor([np.log1p(float(row["Used_Gene_Count"]))], dtype=torch.float32)

        return fp, gene_ids, gene_count, mol_idx, pathway_idx


def predict_collate_fn(batch):
    fps, gene_ids_list, gene_counts, mol_indices, pathway_indices = zip(*batch)

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
        torch.tensor(mol_indices, dtype=torch.long),
        torch.tensor(pathway_indices, dtype=torch.long),
    )


def classify_score(prob_high: float, threshold: float) -> str:
    return "High/P" if prob_high >= threshold else "Low/N"


def run_prediction(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    gene_vocab = load_gene_vocab(args.data_dir)
    scgpt_matrix = load_scgpt_matrix(args.data_dir)

    print(f"Loaded training gene vocab: {len(gene_vocab)} genes")
    print(f"Loaded scGPT matrix shape: {tuple(scgpt_matrix.shape)}")

    molecule_df = read_molecule_csv(args.molecule_csv)
    pathway_df, unseen_df = read_pathway_csv(
        args.pathway_csv,
        gene_vocab=gene_vocab,
        use_provided_gene_count=args.use_provided_gene_count,
    )

    print("========== Input summary ==========")
    print(f"Molecules: {len(molecule_df)}")
    print(f"Pathways : {len(pathway_df)}")
    print(f"Pairs    : {len(molecule_df) * len(pathway_df)}")
    print(f"Unseen gene mappings to 0: {len(unseen_df)}")
    print("===================================")

    if len(unseen_df) > 0:
        unseen_path = os.path.join(args.output_dir, "unseen_genes_mapped_to_0.csv")
        unseen_df.to_csv(unseen_path, index=False)
        print(f"⚠️ Unseen genes saved to: {unseen_path}")

    model = build_model_from_checkpoint(
        scgpt_matrix=scgpt_matrix,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )

    dataset = MoleculePathwayPairDataset(molecule_df, pathway_df)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=predict_collate_fn,
    )

    pair_rows = []

    print("--> Running prediction over molecule-pathway pairs...")
    with torch.no_grad():
        for fps, gene_ids, mask, gene_counts, mol_idx, pathway_idx in tqdm(loader):
            fps = fps.to(device, non_blocking=True)
            gene_ids = gene_ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            gene_counts = gene_counts.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits, _ = model(fps, gene_ids, mask, gene_counts)

            probs = torch.softmax(logits.float(), dim=1)
            prob_low = probs[:, 0].cpu().numpy()
            prob_high = probs[:, 1].cpu().numpy()
            pred_label = (prob_high >= args.threshold).astype(int)

            mol_idx_np = mol_idx.numpy()
            pathway_idx_np = pathway_idx.numpy()

            for m_i, p_i, p0, p1, lab in zip(mol_idx_np, pathway_idx_np, prob_low, prob_high, pred_label):
                mol = molecule_df.iloc[int(m_i)]
                pw = pathway_df.iloc[int(p_i)]

                pair_rows.append({
                    "Molecule_Name": mol["Molecule_Name"],
                    "SMILES": mol["SMILES"],
                    "Pathway_Name": pw["Pathway_Name"],
                    "Pathway_ID": pw["Pathway_ID"],
                    "Pred_Label": int(lab),
                    "Pred_Class": classify_score(float(p1), args.threshold),
                    "Score_High_Prob": float(p1),
                    "Score_Low_Prob": float(p0),
                    "Interaction_Score": float(p1),
                    "Used_Gene_Count": float(pw["Used_Gene_Count"]),
                    "Mapped_Gene_Count": int(pw["Mapped_Gene_Count"]),
                    "Unseen_Gene_Count": int(pw["Unseen_Gene_Count"]),
                })

    pair_df = pd.DataFrame(pair_rows)

    # Sort for readability: molecule, high score descending.
    pair_df = pair_df.sort_values(
        ["Molecule_Name", "Interaction_Score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    pair_path = os.path.join(args.output_dir, "molecule_pathway_pair_predictions.csv")
    pair_df.to_csv(pair_path, index=False)

    # Summary: total score per molecule, plus each pathway score in columns.
    score_matrix = pair_df.pivot_table(
        index="Molecule_Name",
        columns="Pathway_Name",
        values="Interaction_Score",
        aggfunc="first",
    )

    # Use pathway ID in column name if requested to avoid duplicated names.
    if args.use_pathway_id_in_summary_columns:
        pair_df["Pathway_Column"] = pair_df["Pathway_Name"].astype(str) + "|" + pair_df["Pathway_ID"].astype(str)
        score_matrix = pair_df.pivot_table(
            index="Molecule_Name",
            columns="Pathway_Column",
            values="Interaction_Score",
            aggfunc="first",
        )

    score_matrix = score_matrix.fillna(0.0)

    total_score = score_matrix.sum(axis=1)
    mean_score = score_matrix.mean(axis=1)
    max_score = score_matrix.max(axis=1)
    n_high = (score_matrix >= args.threshold).sum(axis=1)

    mol_info = molecule_df.set_index("Molecule_Name")[["SMILES"]]

    summary_df = pd.DataFrame({
        "Total_Interaction_Score": total_score,
        "Mean_Interaction_Score": mean_score,
        "Max_Interaction_Score": max_score,
        "N_High_Pathways": n_high,
    })

    summary_df = mol_info.join(summary_df, how="right")
    summary_df = summary_df.join(score_matrix, how="left")
    summary_df = summary_df.reset_index().rename(columns={"index": "Molecule_Name"})

    summary_df = summary_df.sort_values("Total_Interaction_Score", ascending=False).reset_index(drop=True)

    summary_path = os.path.join(args.output_dir, "molecule_total_scores_with_pathway_scores.csv")
    summary_df.to_csv(summary_path, index=False)

    # Also save pathway preprocessing summary.
    pathway_export = pathway_df.drop(columns=["Gene_IDs"]).copy()
    pathway_export_path = os.path.join(args.output_dir, "pathway_mapping_summary.csv")
    pathway_export.to_csv(pathway_export_path, index=False)

    config = {
        "molecule_csv": args.molecule_csv,
        "pathway_csv": args.pathway_csv,
        "data_dir": args.data_dir,
        "checkpoint_path": args.checkpoint_path,
        "output_dir": args.output_dir,
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "use_provided_gene_count": args.use_provided_gene_count,
        "score_definition": "Score_High_Prob = P(label=1 / High association)",
        "total_score_definition": "Sum of Interaction_Score across all input pathways for each molecule",
        "n_molecules": int(len(molecule_df)),
        "n_pathways": int(len(pathway_df)),
        "n_pairs": int(len(pair_df)),
        "n_unseen_gene_mappings_to_0": int(len(unseen_df)),
    }

    config_path = os.path.join(args.output_dir, "prediction_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print("\n✅ Prediction finished.")
    print(f"Pairwise predictions saved to: {pair_path}")
    print(f"Molecule summary saved to    : {summary_path}")
    print(f"Pathway mapping summary saved: {pathway_export_path}")
    print(f"Config saved to              : {config_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict molecule-pathway interaction classes and scores with trained QFormer."
    )

    parser.add_argument("--molecule_csv", type=str, required=True,
                        help="CSV with column 1=molecule name, column 2=SMILES.")
    parser.add_argument("--pathway_csv", type=str, required=True,
                        help="CSV with column 1=pathway name, column 2=pathway ID, column 3=genes, column 4=gene count.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Training data directory containing cls_gene_metadata.pkl and scgpt_matrix.pt.")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH,
                        help="Trained QFormer checkpoint path.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Threshold for High/P class based on Score_High_Prob.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--use_provided_gene_count", action="store_true", default=DEFAULT_USE_PROVIDED_GENE_COUNT,
                        help="Use the 4th pathway CSV column as pathway gene count prior.")
    parser.add_argument("--ignore_provided_gene_count", dest="use_provided_gene_count", action="store_false",
                        help="Ignore the 4th pathway CSV column and use parsed gene-list length instead.")
    parser.add_argument("--use_pathway_id_in_summary_columns", action="store_true",
                        help="Use PathwayName|PathwayID as score columns in summary file.")

    return parser.parse_args()


if __name__ == "__main__":
    run_prediction(parse_args())
