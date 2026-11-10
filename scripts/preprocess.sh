#!/usr/bin/env bash
# Preprocess raw data: TE quantification, ORF calling, Ribo-seq P-site, MHC binding.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-config/default.yaml}"
SAMPLES_TSV="${2:-data/processed/samples.tsv}"

echo "=== Preprocessing samples ==="
python -m deepte.data.preprocess \
    --config "$CONFIG" \
    --samples "$SAMPLES_TSV"

echo "=== Building feature tensors ==="
python -c "
from deepte.utils.io import load_config
from deepte.features.build_features import FeatureBuilder
cfg = load_config('$CONFIG')
fb = FeatureBuilder(cfg)
fb.build_from_parquet('data/processed/candidate_teps.parquet', 'data/processed/features/')
print('Features saved.')
"
