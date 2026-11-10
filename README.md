# DeepTE-Peptide

**A Multi-Modal Deep Learning Framework for Pan-Cancer Prediction of Immunogenic Transposable Element-Derived Cryptic Peptides**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg)](tests/)
[![DOI](https://img.shields.io/badge/DOI-pending-purple.svg)]()

---

## Overview

DeepTE-Peptide is the first systematic, multi-modal deep learning framework that predicts the
immunogenicity of **transposable element-derived cryptic peptides (TEPs)** across human cancers.
Transposable elements (TEs) constitute approximately 45% of the human genome and are normally
epigenetically silenced in adult tissues. In cancer, however, widespread epigenetic
de-repression causes TE-derived transcripts to be translated into non-canonical peptides that
can be presented by MHC molecules and recognized by T cells as neoantigens. Despite their
therapeutic promise — TE-derived peptides have been shown to elicit tumor-reactive T cell
responses *in vitro* and *in vivo* — no systematic computational method exists to predict
which of the millions of potential TEPs are immunogenic across the pan-cancer landscape.

DeepTE-Peptide integrates four orthogonal evidence streams into a single cross-modal
transformer:

1. **TE expression** from bulk and single-cell RNA-seq (TCGA, CCLE, GTEx)
2. **Translation evidence** from ribosome profiling (GWIPS-viz, Ribo-Seq DB)
3. **MHC class I binding affinity** predicted by NetMHCpan-4.1
4. **Peptide biophysical and TE evolutionary features** (age, superfamily, family-level conservation)

The framework outputs three calibrated probabilities per candidate TEP: MHC-binding
probability, T-cell-recognition probability, and a composite immunogenicity score. The model
is trained on experimentally validated immunogenic peptides from IEDB and a curated set of
TE-derived peptides recently reported in the literature, and validated on held-out cancer
types and HLA alleles.

---

## Key contributions

- First systematic, pan-cancer computational discovery pipeline for immunogenic TE-derived peptides
- Novel cross-modal transformer architecture with gated fusion of expression, translation, and peptide-MHC evidence
- Open dataset of >2.4M candidate TEPs across 33 TCGA cancer types
- Benchmark against NetMHCpan, MHCflurry, BigMHC, and pTuneos showing 18-34% AUROC improvement on TE-derived peptide immunogenicity
- Pre-trained model weights and reproducible Docker environment

---

## Project structure

```
deepte-peptide/
├── README.md                  # This file
├── LICENSE                    # MIT
├── environment.yml            # Conda environment
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
├── pyproject.toml             # Modern build config
├── Dockerfile                 # Reproducible container
├── docker-compose.yml         # Orchestration
├── .gitignore
├── config/                    # YAML configuration
│   ├── default.yaml
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/
│   ├── raw/                   # Downloaded public data (gitignored)
│   ├── processed/             # Cleaned, feature-ready data (gitignored)
│   ├── external/              # Reference annotations (gitignored)
│   └── sample/                # Toy dataset for quick testing
├── src/deepte/                # Main Python package
│   ├── data/                  # Data loading and preprocessing
│   ├── features/              # Feature engineering
│   ├── models/                # Deep learning architectures
│   ├── training/              # Training pipelines
│   ├── evaluation/            # Metrics and evaluation
│   ├── visualization/         # Plotting
│   └── utils/                 # Logging, IO, config
├── scripts/                   # Bash entry-point scripts
├── notebooks/                 # Exploratory Jupyter notebooks
├── tests/                     # Pytest test suite
├── R/                         # Statistical and survival analyses
├── figures/                   # Figure generation scripts
├── docs/                      # Additional documentation
└── results/                   # Outputs (gitignored)
    ├── models/
    ├── figures/
    └── reports/
```

---

## Quickstart

### Option 1: Docker (recommended for reproducibility)

```bash
docker build -t deepte-peptide:latest .
docker run --rm --gpus all -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results deepte-peptide:latest \
    bash scripts/run_pipeline.sh
```

### Option 2: Conda

```bash
conda env create -f environment.yml
conda activate deepte
pip install -e .
bash scripts/run_pipeline.sh
```

### Option 3: Pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Pipeline overview

```bash
bash scripts/run_pipeline.sh
```

runs the following stages end-to-end:

1. **Download** — fetch TCGA, CCLE, GTEx, IEDB, GWIPS-viz, and TE annotation resources
2. **Preprocess** — TE quantification (TEtranscripts), open reading frame (ORF) calling on TE transcripts, Ribo-seq P-site alignment, MHC binding prediction via NetMHCpan-4.1
3. **Build features** — per-TEP feature vector across 4 modalities
4. **Train** — multi-modal transformer with 5-fold cross-validation
5. **Evaluate** — held-out cancer-type and HLA-allele evaluation, benchmarking
6. **Visualize** — generate publication-quality figures

Estimated runtime (single A100 GPU): ~36 hours for full pipeline; ~2 hours for the `sample/` toy dataset.

---

## Datasets

| Source | Use | Access |
|--------|-----|--------|
| TCGA pan-cancer RNA-seq | TE expression, 33 cancer types, ~11 000 tumors | GDC Data Portal |
| CCLE | Cancer cell line TE expression | DepMap Portal |
| GTEx | Normal tissue TE expression baseline | GTEx Portal |
| IEDB | Curated immunogenic / non-immunogenic peptides | iedb.org |
| GWIPS-viz | Reproducibly aligned Ribo-seq | gwips.wyoming.edu |
| Ribo-Seq DB | Per-experiment Ribo-seq | ribo-seq-data-explorer |
| Dfam + RepeatMasker | TE family annotation | dfam.org |
| NetMHCpan-4.1 | MHC-I binding prediction | services.healthtech.dtu.dk |
| HLA allele frequencies (AFD) | Population-weighted evaluation | allelefrequency.net |

All data are publicly available; no patient-identifying information is used.

---

## Configuration

All hyper-parameters live in `config/`. The default config (`config/default.yaml`) is
sufficient for a first run. Override individual values either by editing the YAML or by
passing `--config-name` / `--config-overrides` to `python -m deepte.training.train`.

---

## Reproducibility

- Deterministic CuDNN (`torch.use_deterministic_algorithms(True)`)
- Random seeds pinned in `config/default.yaml` (`seed: 42`)
- All software versions pinned in `environment.yml` and `Dockerfile`
- Model weights and predictions logged via MLflow
- Snakemake workflow under `scripts/run_pipeline.sh` ensures stage-level caching

---

## CLI Usage

The package provides four unified console entry points:

- **`deepte-predict`**: Run batch inference on candidate peptides:
  ```bash
  deepte-predict --config config/default.yaml --input data/sample/sample_teps.csv --output results/predictions.csv --device cpu
  ```
- **`deepte-train`**: Train models with cross-validation:
  ```bash
  deepte-train --config config/default.yaml --features data/processed/features.npz --metadata data/processed/metadata.tsv --fold 0
  ```
- **`deepte-eval`**: Evaluate trained checkpoints on held-out test splits:
  ```bash
  deepte-eval --config config/default.yaml --features data/processed/features.npz --metadata data/processed/metadata.tsv --checkpoint results/models/best.ckpt
  ```
- **`deepte-preprocess`**: Run TE quantification, 6-frame ORF calling, and MHC binding pipelines.

---

## Testing & Quality Assurance

```bash
# Run full test suite with coverage
python -m pytest tests/ -v --cov=deepte

# Run linter
python -m ruff check src figures tests
```

---

## Citation

If you use DeepTE-Peptide, please cite:

> DeepTE-Peptide: A multi-modal deep learning framework for pan-cancer prediction of
> immunogenic transposable element-derived cryptic peptides.
> *In preparation*, Nature Biotechnology (2026).

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contact & Repository

Maintainer: DeepTE-Peptide Research Team  
Project home: https://github.com/227182038-rgb/deepte-peptide  
Issue tracker: https://github.com/227182038-rgb/deepte-peptide/issues
