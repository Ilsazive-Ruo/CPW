"""Build chemical–pathway records using CTD gene annotations."""

import gzip
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

INPUT_CSV = Path('Source/CTD_chem_pathways_enriched.csv')
CTD_GENE_PATHWAY = Path('Source/CTD_genes_pathways.tsv.gz')
CTD_GENE_VOCAB = Path('Source/CTD_genes.tsv.gz')
HUMAN_GENE_INFO = Path('Source/Homo_sapiens.gene_info.gz')
REQUIRE_CONFIRMED_HUMAN = True

OUTPUT_CSV = Path('Source/interaction_processed.csv')
OUTPUT_UNMATCHED = Path('Source/unmatched_pathways.csv')
OUTPUT_MISSING_SMILES = Path('Source/missing_smiles.csv')
OUTPUT_AUDIT = Path('Source/pathway_processing_audit.json')
PUBCHEM_CACHE = Path('Source/pubchem_smiles_cache.csv')
QUERY_PUBCHEM_FOR_MISSING = True
CACHE_FLUSH_EVERY = 100


def colkey(name):
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def col(df, *names, required=True):
    keys = {colkey(c): c for c in df.columns}
    for candidate in names:
        if colkey(candidate) in keys:
            return keys[colkey(candidate)]
    if required:
        raise ValueError(f'Missing columns {names}; found: {list(df.columns)}')
    return None


def open_text(path):
    return gzip.open(path, 'rt', encoding='utf-8-sig') if str(path).endswith('.gz') else open(path, 'rt', encoding='utf-8-sig')


def load_ctd_tsv(path, required, usecols=None):
    if not path.is_file():
        raise FileNotFoundError(f'Missing CTD file: {path}')
    header = None
    with open_text(path) as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            tokens = [x.strip() for x in line.lstrip('#').strip().split('\t')]
            if set(map(colkey, required)).issubset(set(map(colkey, tokens))):
                header = tokens
    if header is None:
        maybe = pd.read_csv(path, sep='\t', comment='#', nrows=0)
        if set(map(colkey, required)).issubset(set(map(colkey, maybe.columns))):
            return pd.read_csv(path, sep='\t', comment='#', dtype=str, usecols=usecols, low_memory=False)
        raise ValueError(
            f'Cannot identify CTD header in {path}; expected {required}. '
            'Inspect the original file format rather than assuming column order.'
        )
    return pd.read_csv(
        path, sep='\t', comment='#', header=None, names=header,
        usecols=usecols, dtype=str, low_memory=False
    )


def load_human_ids(path):
    return set(
        pd.read_csv(path, sep='\t', header=None, comment='#',
                    dtype=str, usecols=[1]).iloc[:, 0].dropna().str.strip()
    )


def get_gene_sets():
    gp = load_ctd_tsv(CTD_GENE_PATHWAY, ['GeneID', 'PathwayID'])
    gid = col(gp, 'GeneID')
    pid = col(gp, 'PathwayID')
    symbol = col(gp, 'GeneSymbol', required=False)
    gp = gp.rename(columns={gid: 'GeneID', pid: 'PathwayID'})
    if symbol:
        gp = gp.rename(columns={symbol: 'GeneSymbol'})
    gp = gp.dropna(subset=['GeneID', 'PathwayID']).copy()
    gp['GeneID'] = gp['GeneID'].astype(str).str.strip()
    gp['PathwayID'] = gp['PathwayID'].astype(str).str.strip()
    gp = gp[(gp['GeneID'] != '') & (gp['PathwayID'] != '')]

    human_applied = HUMAN_GENE_INFO.exists()
    if REQUIRE_CONFIRMED_HUMAN and not human_applied:
        raise FileNotFoundError(f'Human-only filtering requires {HUMAN_GENE_INFO}')
    if human_applied:
        human_ids = load_human_ids(HUMAN_GENE_INFO)
        before = len(gp)
        gp = gp.loc[gp['GeneID'].isin(human_ids)].copy()
        print(f'Human GeneID filter: {before:,} -> {len(gp):,}')
    else:
        print('Human GeneIDs were not restricted to Homo sapiens. '
              f'Add {HUMAN_GENE_INFO} for strict human filtering.')

    if 'GeneSymbol' not in gp.columns:
        genes = load_ctd_tsv(CTD_GENE_VOCAB, ['GeneID', 'GeneSymbol'],
                             usecols=lambda c: colkey(c) in {'geneid', 'genesymbol'})
        genes = genes.rename(columns={col(genes, 'GeneID'): 'GeneID',
                                      col(genes, 'GeneSymbol'): 'GeneSymbol'})
        genes = genes.dropna(subset=['GeneID', 'GeneSymbol']).copy()
        genes['GeneID'] = genes['GeneID'].str.strip()
        genes['GeneSymbol'] = genes['GeneSymbol'].str.strip()
        conflicting = genes.groupby('GeneID')['GeneSymbol'].nunique()
        if (conflicting > 1).any():
            raise ValueError('CTD_genes maps some GeneIDs to multiple GeneSymbols; audit the input version.')
        genes = genes.drop_duplicates('GeneID')
        gp = gp.merge(genes, on='GeneID', how='left', validate='many_to_one')

    gp = gp.dropna(subset=['GeneSymbol']).copy()
    gp['GeneSymbol'] = gp['GeneSymbol'].astype(str).str.upper().str.strip()
    gp = gp[(gp['GeneSymbol'] != '') & (~gp['GeneSymbol'].isin(['NAN', 'NA', '-']))]
    gp = gp.drop_duplicates(['PathwayID', 'GeneSymbol'])
    mapping = gp.groupby('PathwayID')['GeneSymbol'].agg(lambda s: ';'.join(sorted(s))).to_dict()
    print(f'CTD pathways: {len(mapping):,}')
    return mapping, human_applied


def load_pubchem_cache(path):
    if not path.is_file():
        return {}
    cache = pd.read_csv(path, dtype=str).fillna('')
    if not {'ChemicalName', 'SMILES', 'CID'}.issubset(cache.columns):
        raise ValueError(f'PubChem cache must contain ChemicalName, SMILES, CID: {path}')
    return {str(row.ChemicalName): (row.SMILES, row.CID) for row in cache.itertuples(index=False)}


def flush_pubchem_cache(cache):
    PUBCHEM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [(name, smiles, cid) for name, (smiles, cid) in cache.items()],
        columns=['ChemicalName', 'SMILES', 'CID']
    ).to_csv(PUBCHEM_CACHE, index=False)


def add_pubchem_smiles(df):
    chemical_col = col(df, '# ChemicalName', 'ChemicalName')
    smiles_col = col(df, 'SMILES', required=False)
    cid_col = col(df, 'CID', required=False)
    if smiles_col is None:
        df['SMILES'] = ''
    elif smiles_col != 'SMILES':
        df = df.rename(columns={smiles_col: 'SMILES'})
    if cid_col is None:
        df['CID'] = ''
    elif cid_col != 'CID':
        df = df.rename(columns={cid_col: 'CID'})

    df['SMILES'] = df['SMILES'].fillna('').astype(str).str.strip()
    df['CID'] = df['CID'].fillna('').astype(str).str.strip()
    names = df[chemical_col].fillna('').astype(str).str.strip()
    cache = load_pubchem_cache(PUBCHEM_CACHE)
    missing_names = sorted(set(names[(df['SMILES'] == '') & (names != '')]))

    if QUERY_PUBCHEM_FOR_MISSING and missing_names:
        try:
            import pubchempy as pcp
        except ImportError as exc:
            raise ImportError('Install PubChemPy or set QUERY_PUBCHEM_FOR_MISSING=False') from exc
        for i, name in enumerate(tqdm(missing_names, desc='PubChem name lookups'), 1):
            if name not in cache:
                try:
                    results = pcp.get_compounds(name, namespace='name')
                    if len(results) == 1:
                        item = results[0]
                        smiles = getattr(item, 'canonical_smiles', None) or getattr(item, 'connectivity_smiles', None) or ''
                        cache[name] = (str(smiles or ''), str(item.cid or ''))
                    else:
                        cache[name] = ('', '')
                except Exception as exc:
                    print(f'PubChem lookup failed for {name!r}: {exc}')

            if i % CACHE_FLUSH_EVERY == 0:
                flush_pubchem_cache(cache)
        flush_pubchem_cache(cache)

    lookup_smiles = names.map(lambda n: cache.get(n, ('', ''))[0])
    lookup_cid = names.map(lambda n: cache.get(n, ('', ''))[1])
    df['SMILES'] = df['SMILES'].mask(df['SMILES'] == '', lookup_smiles).fillna('')
    df['CID'] = df['CID'].mask(df['CID'] == '', lookup_cid).fillna('')
    return df


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(INPUT_CSV)
    df = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)
    original_rows = len(df)
    path_id_col = col(df, 'PathwayID', 'Pathway_ID', 'Pathway ID')
    path_name_col = col(df, 'PathwayName', 'Pathway_Name', required=False)
    df['PathwayID'] = df[path_id_col].fillna('').astype(str).str.strip()
    mapping, human_applied = get_gene_sets()

    df['Gene_Set'] = df['PathwayID'].map(mapping)
    matched = df['Gene_Set'].notna() & (df['Gene_Set'] != '')
    unmatched = df.loc[~matched].copy()
    columns = ['PathwayID'] + ([path_name_col] if path_name_col else [])
    (unmatched.groupby(columns, dropna=False).size().reset_index(name='Sample_Count')
              .sort_values('Sample_Count', ascending=False)
              .to_csv(OUTPUT_UNMATCHED, index=False))
    df = df.loc[matched].copy()
    if df.empty:
        raise RuntimeError('No PathwayID matched CTD gene-pathway file. Check CTD versions and PathwayID formats.')
    df['Matched_Pathway'] = df[path_name_col] if path_name_col else df['PathwayID']
    df['Gene_Count'] = df['Gene_Set'].str.count(';') + 1
    df['Pathway_Match_Score'] = 100.0
    df['Gene_Set_Source'] = 'CTD_genes_pathways (exact PathwayID)'
    df = add_pubchem_smiles(df)

    missing_smiles = df['SMILES'].eq('')
    df.loc[missing_smiles].to_csv(OUTPUT_MISSING_SMILES, index=False)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    audit = {
        'original_rows': original_rows,
        'matched_rows': int(len(df)),
        'unmatched_rows': int(len(unmatched)),
        'matched_unique_pathways': int(df['PathwayID'].nunique()),
        'unmatched_unique_pathways': int(unmatched['PathwayID'].nunique()),
        'matched_unique_genes': int(len({g for genes in df['Gene_Set'].drop_duplicates() for g in genes.split(';')})),
        'missing_smiles_rows': int(missing_smiles.sum()),
        'human_gene_id_filter_applied': bool(human_applied),
        'gene_set_source': str(CTD_GENE_PATHWAY),
        'gene_symbol_source': str(CTD_GENE_VOCAB),
        'gene_id_match': 'exact PathwayID',
    }
    OUTPUT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f'Saved: {OUTPUT_CSV}; unmatched: {OUTPUT_UNMATCHED}')


if __name__ == '__main__':
    main()
