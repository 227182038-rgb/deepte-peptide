#!/usr/bin/env bash
# Train DeepTE-Peptide model (5-fold cross-validation).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-config/default.yaml}"
FEATURES="${2:-data/processed/features/features.npz}"
METADATA="${3:-data/processed/features/metadata.tsv}"
N_FOLDS="${N_FOLDS:-5}"

for fold in $(seq 0 $((N_FOLDS - 1))); do
    echo "=== Training fold $fold/$N_FOLDS ==="
    python -m deepte.training.train \
        --config "$CONFIG" \
        --features "$FEATURES" \
        --metadata "$METADATA" \
        --fold "$fold"
done

echo "=== Training complete ==="
