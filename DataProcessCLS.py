"""Prepare Morgan fingerprints and scGPT gene features for classification."""

import json
import pickle
import re
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

CSV_PATH = Path('Source/interaction_processed.csv')
SCGPT_MODEL_DIR = Path('./scGPT')
OUTPUT_DIR = Path('CLSData2')
SEED = 42
NUM_WORKERS = 8
HIGH_QUANTILE = 0.80
LOW_QUANTILE = 0.20
P_MIN = 1e-300
FP_SIZE = 2048
MORGAN_RADIUS = 2

UNMATCHED_GENE_POLICY = 'drop'
MIN_MATCHED_GENES = 1
MIN_MATCH_RATIO_PER_PATHWAY = 0.0
REJECT_LOW_COVERAGE = True

RDLogger.DisableLog('rdApp.error')
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=FP_SIZE)


def colkey(name):
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def find_col(df, names):
    candidate_map = {colkey(c): c for c in df.columns}
    for name in names:
        if colkey(name) in candidate_map:
            return candidate_map[colkey(name)]
    raise ValueError(f'Missing required column {names}; got {list(df.columns)}')


def valid_smiles(value):
    if pd.isna(value) or not str(value).strip():
        return False
    return Chem.MolFromSmiles(str(value).strip()) is not None


def to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES passed feature validation: {smiles!r}')
    bitvect = _MORGAN.GetFingerprint(mol)
    arr = np.zeros(FP_SIZE, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr.astype(np.float32)


def make_fps(smiles_list, workers):
    if workers <= 1:
        return np.asarray([to_fp(s) for s in tqdm(smiles_list, desc='Morgan fingerprints')], dtype=np.float32)
    with Pool(processes=workers) as pool:
        fps = list(tqdm(pool.imap(to_fp, smiles_list, chunksize=128),
                        total=len(smiles_list), desc='Morgan fingerprints'))
    return np.asarray(fps, dtype=np.float32)


def parse_gene_set(value):
    if pd.isna(value):
        return []
    return sorted({g.strip().upper() for g in str(value).split(';') if g.strip()})


def safe_torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_scgpt():
    vocab_path = SCGPT_MODEL_DIR / 'vocab.json'
    model_path = SCGPT_MODEL_DIR / 'best_model.pt'
    for path in [vocab_path, model_path]:
        if not path.is_file():
            raise FileNotFoundError(
                f'{path} is missing: scGPT embeddings unavailable. '
                'Download the pretrained scGPT model and corresponding vocab.json.'
            )
    with open(vocab_path, encoding='utf-8') as fh:
        raw_vocab = json.load(fh)
    if not isinstance(raw_vocab, dict):
        raise ValueError('scGPT vocab.json must map gene names to token IDs')

    scgpt_vocab = {}
    for gene, idx in raw_vocab.items():
        key = str(gene).strip().upper()
        if key in scgpt_vocab and scgpt_vocab[key] != int(idx):
            raise ValueError(f'Case-insensitive collision in scGPT vocab for {key}')
        scgpt_vocab[key] = int(idx)

    ckpt = safe_torch_load(model_path)
    if not isinstance(ckpt, dict):
        raise ValueError('scGPT checkpoint is not a state dictionary')
    state = ckpt.get('model_state_dict', ckpt)
    for prefix in ('module.', '_orig_mod.'):
        if any(k.startswith(prefix) for k in state):
            state = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
    possible_keys = ('encoder.embedding.weight', 'gene_encoder.embedding.weight')
    found = [k for k in possible_keys if k in state]
    if not found:
        raise KeyError(f'Gene embedding key not found in checkpoint. '
                       f'Expected {possible_keys}. Found keys e.g. {list(state)[:15]}')
    weights = state[found[0]].detach().cpu().float().numpy()
    if weights.ndim != 2:
        raise ValueError(f'Expected 2D embedding table, got {weights.shape}')
    if max(scgpt_vocab.values()) >= len(weights) or min(scgpt_vocab.values()) < 0:
        raise ValueError('scGPT vocab token indices do not match embedding table')
    if weights.shape[1] != 512:
        raise ValueError(f'Expected 512d scGPT embeddings; got {weights.shape[1]}')
    print(f'Loaded scGPT embedding table: {weights.shape}; vocab: {len(scgpt_vocab):,}')
    return scgpt_vocab, weights


def classify(scores):
    if not (0 < LOW_QUANTILE < HIGH_QUANTILE < 1):
        raise ValueError('Require 0 < LOW_QUANTILE < HIGH_QUANTILE < 1')
    lo = float(np.quantile(scores, LOW_QUANTILE))
    hi = float(np.quantile(scores, HIGH_QUANTILE))
    if not lo < hi:
        raise ValueError(
            f'Quantile score thresholds overlap (low={lo}, high={hi}). '
            'Many identical corrected P-values; revise label strategy explicitly.'
        )
    labels = np.full(len(scores), -1, dtype=np.int64)
    labels[scores <= lo] = 0
    labels[scores >= hi] = 1
    if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
        raise ValueError('At least one classification group is empty')
    summary = {
        'score_definition': 'S = -log10(CorrectedPValue)',
        'low_quantile': LOW_QUANTILE, 'high_quantile': HIGH_QUANTILE,
        'low_threshold': lo, 'high_threshold': hi,
        'n_total': int(len(scores)), 'n_low': int((labels == 0).sum()),
        'n_high': int((labels == 1).sum()),
        'n_middle': int((labels == -1).sum()),
        'n_valid': int((labels != -1).sum()),
        'low_label': 0, 'high_label': 1, 'middle_label': -1,
    }
    return labels, summary


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if UNMATCHED_GENE_POLICY not in {'drop', 'random'}:
        raise ValueError("UNMATCHED_GENE_POLICY must be 'drop' or 'random'")
    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    smiles_col = find_col(df, ['SMILES'])
    genes_col = find_col(df, ['Gene_Set', 'GeneSet'])
    p_col = find_col(df, ['CorrectedPValue', 'Corrected_PValue', 'AdjustedPValue', 'FDR', 'qvalue'])

    smiles_ok = df[smiles_col].apply(valid_smiles)
    raw_sets = df[genes_col].apply(parse_gene_set)
    genes_ok = raw_sets.apply(len).gt(0)
    pvals = pd.to_numeric(df[p_col], errors='coerce')
    p_ok = pvals.notna() & np.isfinite(pvals) & (pvals >= 0) & (pvals <= 1)

    scgpt_vocab, weights = load_scgpt()
    recognized = raw_sets.map(lambda genes: [g for g in genes if g in scgpt_vocab])
    n_matched = recognized.map(len)
    coverage = n_matched / raw_sets.map(len).replace(0, np.nan)
    coverage = coverage.fillna(0.0)
    if UNMATCHED_GENE_POLICY == 'drop':
        embedding_ok = (n_matched >= MIN_MATCHED_GENES) & (coverage >= MIN_MATCH_RATIO_PER_PATHWAY)
    else:
        embedding_ok = (raw_sets.map(len) > 0)

    audit = pd.DataFrame({
        'original_row': np.arange(len(df)),
        'SMILES_valid': smiles_ok,
        'Gene_Set_nonempty': genes_ok,
        'CorrectedPValue_valid': p_ok,
        'Gene_Count_raw_unique': raw_sets.map(len),
        'scGPT_Gene_Count': n_matched,
        'scGPT_Gene_Coverage': coverage,
        'Embedding_coverage_pass': embedding_ok,
    })
    if 'PathwayID' in df.columns:
        audit['PathwayID'] = df['PathwayID']
    if 'Gene_Set' in df.columns:
        audit['Gene_Set'] = df['Gene_Set']
    good = smiles_ok & genes_ok & p_ok & (embedding_ok if REJECT_LOW_COVERAGE else True)
    audit['Kept'] = good
    audit.to_csv(OUTPUT_DIR / 'sample_qc_audit.csv', index=False)
    before = len(df)
    df = df.loc[good].copy().reset_index(drop=True)
    sets = raw_sets.loc[good].tolist()
    if not len(df):
        raise ValueError('No samples remain after QC; inspect sample_qc_audit.csv')
    print(f'Retained samples: {len(df):,}/{before:,}')
    print(f'scGPT coverage (retained; mean): {coverage.loc[good].mean():.2%}')

    if 'PathwayID' in df.columns:
        check = pd.DataFrame({'PathwayID': df['PathwayID'].to_numpy(),
                              'Gene_Set_Normalized': [';'.join(s) for s in sets]})
        inconsistent = check.groupby('PathwayID')['Gene_Set_Normalized'].nunique()
        if (inconsistent > 1).any():
            raise ValueError(f'Same PathwayID has inconsistent Gene_Set: {inconsistent[inconsistent > 1].head().to_dict()}')

    if UNMATCHED_GENE_POLICY == 'drop':
        model_sets = [[g for g in row if g in scgpt_vocab] for row in sets]
    else:
        model_sets = sets
    all_genes = sorted({g for row in model_sets for g in row})
    gene_vocab = {g: i + 1 for i, g in enumerate(all_genes)}
    pathway_gene_ids = [[gene_vocab[g] for g in row] for row in model_sets]
    if any(len(x) == 0 for x in pathway_gene_ids):
        raise ValueError('Empty model gene set remains after QC')

    rng = np.random.default_rng(SEED)
    matrix = np.zeros((len(gene_vocab) + 1, weights.shape[1]), dtype=np.float32)
    unmatched_genes = []
    for gene, idx in gene_vocab.items():
        if gene in scgpt_vocab:
            matrix[idx] = weights[scgpt_vocab[gene]]
        else:
            unmatched_genes.append(gene)
            matrix[idx] = rng.normal(0, 0.01, weights.shape[1])
    torch.save(torch.tensor(matrix, dtype=torch.float32), OUTPUT_DIR / 'scgpt_matrix.pt')
    pd.DataFrame({'gene': unmatched_genes}).to_csv(OUTPUT_DIR / 'unmatched_scgpt_genes.csv', index=False)

    p = pd.to_numeric(df[p_col]).to_numpy(dtype=np.float64)
    scores = -np.log10(np.clip(p, P_MIN, 1.0)).astype(np.float32)
    labels, thresholds = classify(scores)
    kept_mask = labels != -1
    fps = make_fps(df[smiles_col].astype(str).tolist(), NUM_WORKERS)

    np.save(OUTPUT_DIR / 'all_morgan_fps.npy', fps)
    np.save(OUTPUT_DIR / 'all_s_scores.npy', scores)
    np.save(OUTPUT_DIR / 'all_class_labels.npy', labels)
    np.save(OUTPUT_DIR / 'valid_classification_mask.npy', kept_mask)
    with open(OUTPUT_DIR / 'all_gene_metadata.pkl', 'wb') as fh:
        pickle.dump({'vocab': gene_vocab, 'gene_ids': pathway_gene_ids}, fh)

    cls_ids = [row for row, keep in zip(pathway_gene_ids, kept_mask) if keep]
    np.save(OUTPUT_DIR / 'cls_morgan_fps.npy', fps[kept_mask])
    np.save(OUTPUT_DIR / 'cls_labels.npy', labels[kept_mask])
    np.save(OUTPUT_DIR / 'cls_s_scores.npy', scores[kept_mask])
    with open(OUTPUT_DIR / 'cls_gene_metadata.pkl', 'wb') as fh:
        pickle.dump({'vocab': gene_vocab, 'gene_ids': cls_ids}, fh)

    thresholds.update({
        'original_rows_before_qc': int(before),
        'rows_after_qc': int(len(df)),
        'gene_vocab_size': int(len(gene_vocab)),
        'scgpt_matched_gene_count': int(len(gene_vocab) - len(unmatched_genes)),
        'unmatched_gene_count': int(len(unmatched_genes)),
        'mean_scgpt_gene_coverage': float(coverage.loc[good].mean()),
        'unmatched_gene_policy': UNMATCHED_GENE_POLICY,
        'min_matched_genes': MIN_MATCHED_GENES,
        'min_match_ratio_per_pathway': MIN_MATCH_RATIO_PER_PATHWAY,
        'seed': SEED,
    })
    with open(OUTPUT_DIR / 'classification_thresholds.json', 'w', encoding='utf-8') as fh:
        json.dump(thresholds, fh, indent=2, ensure_ascii=False)

    df_out = df.copy()
    df_out['S_score'] = scores
    df_out['class_label'] = labels
    df_out['class_name'] = df_out['class_label'].map({1: 'High association', 0: 'Low association', -1: 'Middle / excluded'})
    df_out['scGPT_Gene_Count'] = [sum(g in scgpt_vocab for g in gs) for gs in sets]
    df_out.to_csv(OUTPUT_DIR / 'interaction_classification_labeled.csv', index=False)
    print(f'Classification samples: {len(cls_ids):,}, '
          f'{len(gene_vocab):,} genes, scGPT embedding shape={matrix.shape}')
    print(f'Files saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
