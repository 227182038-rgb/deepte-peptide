#!/usr/bin/env bash
# Download all public datasets used by DeepTE-Peptide.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source config/default_paths.sh 2>/dev/null || true

DATA_RAW="${DEEPTE_DATA_DIR:-data}/raw"
DATA_EXT="${DEEPTE_DATA_DIR:-data}/external"
mkdir -p "$DATA_RAW"/{tcga,gtex,ccle,iedb,riboseq,dfam,gencode,hla}
mkdir -p "$DATA_EXT"

echo "=== [1/9] Downloading IEDB epitope data ==="
python -m deepte.data.download --source iedb_epitope --out "$DATA_RAW/iedb/epitopes.csv"

echo "=== [2/9] Downloading IEDB MHC binding assays ==="
python -m deepte.data.download --source iedb_mhc_assay --out "$DATA_RAW/iedb/mhc_assays.csv"

echo "=== [3/9] Downloading IEDB T-cell assays ==="
python -m deepte.data.download --source iedb_tcell_assay --out "$DATA_RAW/iedb/tcell_assays.csv"

echo "=== [4/9] Downloading GTEx v8 RNA-seq ==="
python -m deepte.data.download --source gtex_v8 --out "$DATA_RAW/gtex/gtex_v8_tpm.gct.gz"

echo "=== [5/9] Downloading CCLE RNA-seq ==="
python -m deepte.data.download --source ccle --out "$DATA_RAW/ccle/ccle_tpm.csv"

echo "=== [6/9] Downloading Dfam 3.7 ==="
python -m deepte.data.download --source dfam_v37 --out "$DATA_RAW/dfam/Dfam.h5.gz"
gunzip -kf "$DATA_RAW/dfam/Dfam.h5.gz"

echo "=== [7/9] Downloading RepeatMasker hg38 ==="
python -m deepte.data.download --source repeatmasker_db --out "$DATA_RAW/dfam/hg38_rm.tar.gz"
tar -xzf "$DATA_RAW/dfam/hg38_rm.tar.gz" -C "$DATA_EXT/"

echo "=== [8/9] Downloading GENCODE v44 ==="
python -m deepte.data.download --source gencode_v44 --out "$DATA_RAW/gencode/gencode.v44.gtf.gz"

echo "=== [9/9] Downloading HLA allele frequencies ==="
python -m deepte.data.download --source hla_freq --out "$DATA_RAW/hla/allele_frequencies.csv"

echo "Done."
