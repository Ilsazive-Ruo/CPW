import os
import re
import json
import pickle
import argparse
import warnings

import numpy as np
import pandas as pd
import torch

from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from QFormer import QFormerClassifier

warnings.filterwarnings("ignore")

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def safe_name(x):
    x = re.sub(r'[\\/:*?"<>|]+', "_", str(x))
    return re.sub(r"\s+", "_", x).strip("_")[:80]


def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(2048, dtype=np.float32)
    fp = MORGAN.GetFingerprint(mol)
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def split_genes(x):
    if pd.isna(x):
        return []
    return [g.strip() for g in str(x).replace(";", ",").split(",") if g.strip()]


def load_vocab(data_dir):
    with open(os.path.join(data_dir, "cls_gene_metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    return meta["vocab"]


def load_scgpt(data_dir):
    x = torch.load(os.path.join(data_dir, "scgpt_matrix.pt"), map_location="cpu")
    return x.float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)


def read_molecules(path):
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("Molecule CSV needs >=2 columns: name, SMILES.")
    out = df.iloc[:, :2].copy()
    out.columns = ["Molecule_Name", "SMILES"]
    return out


def read_pathways(path, vocab, use_provided_count=True):
    df = pd.read_csv(path)
    if df.shape[1] < 4:
        raise ValueError("Pathway CSV needs >=4 columns: name, ID, genes, gene count.")

    rows = []
    unseen = []

    for _, r in df.iterrows():
        name = str(r.iloc[0])
        pid = str(r.iloc[1])
        genes = split_genes(r.iloc[2])

        gene_ids = []
        mapped = []

        for g in genes:
            gid = vocab.get(g)
            if gid is None:
                gid = vocab.get(g.upper())
            if gid is None:
                gene_ids.append(0)
                mapped.append(False)
                unseen.append({"Pathway_Name": name, "Pathway_ID": pid, "Unseen_Gene": g})
            else:
                gene_ids.append(int(gid))
                mapped.append(True)

        if not gene_ids:
            genes = ["<EMPTY>"]
            gene_ids = [0]
            mapped = [False]

        try:
            provided = float(r.iloc[3])
        except Exception:
            provided = np.nan

        if use_provided_count and np.isfinite(provided) and provided > 0:
            used_count = provided
        else:
            used_count = len(genes)

        rows.append({
            "Pathway_Name": name,
            "Pathway_ID": pid,
            "Genes": genes,
            "Gene_IDs": gene_ids,
            "Mapped": mapped,
            "Used_Gene_Count": float(used_count),
        })

    return pd.DataFrame(rows), pd.DataFrame(unseen)


def load_model(scgpt, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        cfg = ckpt.get("config", {})
    else:
        state = ckpt
        cfg = {}

    cleaned = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        cleaned[k] = v

    model = QFormerClassifier(
        pretrained_scgpt_matrix=scgpt,
        embed_dim=int(cfg.get("embed_dim", 512)),
        num_heads=int(cfg.get("num_heads", 8)),
        num_queries=int(cfg.get("num_queries", 8)),
        num_qformer_layers=int(cfg.get("num_qformer_layers", 2)),
        dropout_rate=float(cfg.get("dropout_rate", 0.1)),
        num_classes=int(cfg.get("num_classes", 2)),
        freeze_gene_embedding=bool(cfg.get("freeze_gene_embedding", True)),
        use_mean_residual=bool(cfg.get("use_mean_residual", True)),
    ).to(device)

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        raise RuntimeError(f"Missing model keys: {missing[:20]}")
    if unexpected:
        print("Unexpected keys:", unexpected[:20])

    model.eval()
    return model


@torch.no_grad()
def predict_one(model, fp, gene_ids, gene_count_log, device, return_attention=False):
    if not gene_ids:
        gene_ids = [0]

    fp_t = torch.tensor(fp, dtype=torch.float32, device=device).unsqueeze(0)
    gid_t = torch.tensor(gene_ids, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.zeros((1, len(gene_ids)), dtype=torch.bool, device=device)
    cnt_t = torch.tensor([[gene_count_log]], dtype=torch.float32, device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        logits, attn = model(fp_t, gid_t, mask, cnt_t)

    logits = logits.float()
    probs = torch.softmax(logits, dim=1)[0]
    margin = (logits[0, 1] - logits[0, 0]).item()

    out = {
        "Prob_High": float(probs[1].item()),
        "Positive_Logit": float(margin),
    }

    if return_attention:
        a = attn.detach().float().cpu().numpy()
        if a.ndim == 3:
            a = a[0]         # [Q,L]
        elif a.ndim == 4:
            a = a[0].mean(0)
        out["Attention"] = a

    return out


def explain_pair(model, mol, pw, fp, device, top_n, pair_dir):
    genes = list(pw["Genes"])
    gene_ids = list(pw["Gene_IDs"])
    mapped = list(pw["Mapped"])
    fixed_count_log = float(np.log1p(float(pw["Used_Gene_Count"])))

    original = predict_one(
        model, fp, gene_ids, fixed_count_log, device, return_attention=True
    )

    attn = original["Attention"]
    if attn.ndim == 1:
        attn = attn[None, :]
    if attn.shape[-1] != len(gene_ids) and attn.shape[0] == len(gene_ids):
        attn = attn.T

    mean_attn = attn.mean(axis=0)

    rows = []
    for i, (gene, gid, is_mapped) in enumerate(zip(genes, gene_ids, mapped)):
        occluded_ids = gene_ids[:i] + gene_ids[i+1:]
        if not occluded_ids:
            occluded_ids = [0]

        occluded = predict_one(
            model, fp, occluded_ids, fixed_count_log, device, return_attention=False
        )

        row = {
            "Molecule_Name": mol["Molecule_Name"],
            "Pathway_Name": pw["Pathway_Name"],
            "Pathway_ID": pw["Pathway_ID"],
            "Gene_Symbol": gene,
            "Gene_ID": int(gid),
            "Mapped_To_Training_Vocab": bool(is_mapped),
            "Mean_Attention": float(mean_attn[i]),
            "Original_Prob_High": original["Prob_High"],
            "Original_Positive_Logit": original["Positive_Logit"],
            "Occluded_Prob_High": occluded["Prob_High"],
            "Occluded_Positive_Logit": occluded["Positive_Logit"],
            "Delta_Probability": original["Prob_High"] - occluded["Prob_High"],
            "Delta_Logit": original["Positive_Logit"] - occluded["Positive_Logit"],
        }

        for q in range(attn.shape[0]):
            row[f"Attention_Query_{q+1}"] = float(attn[q, i])

        rows.append(row)

    gene_df = pd.DataFrame(rows)
    gene_df["Abs_Delta_Logit"] = gene_df["Delta_Logit"].abs()
    gene_df = gene_df.sort_values("Delta_Logit", ascending=False).reset_index(drop=True)

    os.makedirs(pair_dir, exist_ok=True)
    gene_df.to_csv(os.path.join(pair_dir, "gene_attribution.csv"), index=False)
    gene_df.head(top_n).to_csv(os.path.join(pair_dir, "top_genes.csv"), index=False)

    attn_df = pd.DataFrame(
        attn,
        columns=genes,
        index=[f"Query_{i+1}" for i in range(attn.shape[0])]
    )
    attn_df.to_csv(os.path.join(pair_dir, "qformer_attention_matrix.csv"))

    summary = {
        "Molecule_Name": mol["Molecule_Name"],
        "SMILES": mol["SMILES"],
        "Pathway_Name": pw["Pathway_Name"],
        "Pathway_ID": pw["Pathway_ID"],
        "Used_Gene_Count": float(pw["Used_Gene_Count"]),
        "Original_Prob_High": original["Prob_High"],
        "Original_Positive_Logit": original["Positive_Logit"],
        "Note": "Gene-count scalar held fixed during leave-one-gene-out occlusion."
    }

    with open(os.path.join(pair_dir, "pair_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary, gene_df


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    vocab = load_vocab(args.data_dir)
    scgpt = load_scgpt(args.data_dir)
    molecules = read_molecules(args.molecule_csv)
    pathways, unseen = read_pathways(
        args.pathway_csv, vocab, args.use_provided_gene_count
    )

    if not unseen.empty:
        unseen.to_csv(
            os.path.join(args.output_dir, "unseen_genes_mapped_to_0.csv"),
            index=False
        )

    fps = np.vstack([smiles_to_fp(s) for s in molecules["SMILES"]]).astype(np.float32)
    model = load_model(scgpt, args.checkpoint_path, device)

    # 1) Predict all combinations.
    pair_rows = []

    print(f"Molecules={len(molecules)}, Pathways={len(pathways)}, Pairs={len(molecules)*len(pathways)}")

    for mi, mol in molecules.iterrows():
        for pi, pw in pathways.iterrows():
            pred = predict_one(
                model,
                fps[mi],
                pw["Gene_IDs"],
                float(np.log1p(float(pw["Used_Gene_Count"]))),
                device,
                return_attention=False,
            )

            pair_rows.append({
                "Molecule_Index": int(mi),
                "Pathway_Index": int(pi),
                "Molecule_Name": mol["Molecule_Name"],
                "SMILES": mol["SMILES"],
                "Pathway_Name": pw["Pathway_Name"],
                "Pathway_ID": pw["Pathway_ID"],
                "Score_High_Prob": pred["Prob_High"],
                "Positive_Logit": pred["Positive_Logit"],
                "Used_Gene_Count": float(pw["Used_Gene_Count"]),
            })

    pair_df = pd.DataFrame(pair_rows).sort_values(
        ["Molecule_Name", "Score_High_Prob"],
        ascending=[True, False]
    )

    pair_df.to_csv(
        os.path.join(args.output_dir, "all_pair_predictions.csv"),
        index=False
    )

    # 2) Select top N per molecule.
    eligible = pair_df[pair_df["Score_High_Prob"] >= args.min_score].copy()

    if args.interpret_all:
        selected = eligible.copy()
    else:
        selected = (
            eligible.groupby("Molecule_Name", group_keys=False)
            .head(args.top_pairs_per_molecule)
            .copy()
        )

    selected = selected.sort_values("Score_High_Prob", ascending=False).reset_index(drop=True)
    selected["Interpretation_Rank"] = np.arange(1, len(selected) + 1)

    selected.to_csv(
        os.path.join(args.output_dir, "selected_pairs_for_interpretation.csv"),
        index=False
    )

    # 3) Explain selected pairs.
    summaries = []
    gene_tables = []

    for _, r in selected.iterrows():
        rank = int(r["Interpretation_Rank"])
        mi = int(r["Molecule_Index"])
        pi = int(r["Pathway_Index"])

        folder = (
            f"pair_{rank:03d}_"
            f"{safe_name(r['Molecule_Name'])}_"
            f"{safe_name(r['Pathway_Name'])}"
        )

        print(
            f"[{rank}/{len(selected)}] "
            f"{r['Molecule_Name']} | {r['Pathway_Name']} | "
            f"P(high)={r['Score_High_Prob']:.4f}"
        )

        summary, gene_df = explain_pair(
            model=model,
            mol=molecules.iloc[mi],
            pw=pathways.iloc[pi],
            fp=fps[mi],
            device=device,
            top_n=args.top_n_genes,
            pair_dir=os.path.join(args.output_dir, folder),
        )

        summary["Interpretation_Rank"] = rank
        summaries.append(summary)
        gene_df["Interpretation_Rank"] = rank
        gene_tables.append(gene_df)

    pd.DataFrame(summaries).to_csv(
        os.path.join(args.output_dir, "interpreted_pair_summary.csv"),
        index=False
    )

    if gene_tables:
        pd.concat(gene_tables, ignore_index=True).to_csv(
            os.path.join(args.output_dir, "all_selected_pair_gene_attributions.csv"),
            index=False
        )

    print("\nDone.")
    print("Main files:")
    print("  all_pair_predictions.csv")
    print("  selected_pairs_for_interpretation.csv")
    print("  interpreted_pair_summary.csv")
    print("  all_selected_pair_gene_attributions.csv")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--molecule_csv", required=True)
    p.add_argument("--pathway_csv", required=True)

    p.add_argument("--data_dir", default="CLSData2")
    p.add_argument(
        "--checkpoint_path",
        default="QFormer_Finetune_NP_CSV/best_finetuned_qformer_checkpoint.pt"
    )
    p.add_argument("--output_dir", default="QFormer_Interpretability")

    p.add_argument("--top_pairs_per_molecule", type=int, default=3)
    p.add_argument("--top_n_genes", type=int, default=20)
    p.add_argument("--min_score", type=float, default=0.0)
    p.add_argument("--interpret_all", action="store_true")

    p.add_argument(
        "--use_provided_gene_count",
        action="store_true",
        default=True
    )
    p.add_argument(
        "--ignore_provided_gene_count",
        dest="use_provided_gene_count",
        action="store_false"
    )

    p.add_argument("--cpu", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
