# CPW

**Network-Informed Deep Learning Framework for Pathway-Guided Active Pharmaceutical Ingredient Prioritization Against Drug-Induced Liver Injury**

This repository provides the code for a molecular–pathway interaction prediction framework used to prioritize candidate active pharmaceutical ingredients for drug-induced liver injury (DILI). Molecules are represented by 2,048-bit Morgan fingerprints, pathways by scGPT-based gene embeddings, and their associations are modeled using a Q-Former classifier. The repository also includes cross-validation, cold-start evaluation, and interpretability analyses.

## Installation

```bash
git clone https://github.com/Ilsazive-Ruo/CPW.git
cd CPW
conda env create -n cpw -f environment.yml
conda activate cpw
```

The supplied `environment.yml` contains platform-specific dependencies and may require adjustment for your system.

## Data preparation and training

Place the CTD input files in `Source/` and the scGPT `best_model.pt` and `vocab.json` files in `scGPT/`. The preprocessing scripts expect the CTD chemical–pathway enrichment CSV, CTD gene–pathway and gene annotation files, and human gene annotations (see the file paths at the top of `GetSmiAndGeneSet.py`). Then run:

```bash
python GetSmiAndGeneSet.py
python DataProcessCLS.py
python QFormer_train.py
```

For evaluation, use `Qformer_CV.py` (cross-validation) or `CV_qformer_cold_start.py` (cold-start evaluation). Additional analysis scripts are provided in the repository.

## Pretrained weights and prediction

Trained model weights are available from **[Figshare (10.6084/m9.figshare.34476108)](https://doi.org/10.6084/m9.figshare.34476108)**.

For prediction, provide a molecule CSV (first two columns: molecule name and SMILES), a pathway CSV (first four columns: pathway name, pathway ID, gene list, gene count), and the downloaded checkpoint:

```bash
python Predictor.py \
  --molecule_csv molecules.csv \
  --pathway_csv pathways.csv \
  --checkpoint_path path/to/checkpoint.pt \
  --data_dir CLSData2
```

The prediction script also requires `CLSData2/cls_gene_metadata.pkl` and `CLSData2/scgpt_matrix.pt` from the corresponding preprocessing run. Results are saved to `QFormer_Predictions/` by default.

## License

The repository's source code is distributed under the [MIT License](LICENSE). Third-party datasets and pretrained scGPT assets remain subject to their respective terms; see the [scGPT repository](https://github.com/bowang-lab/scGPT).
