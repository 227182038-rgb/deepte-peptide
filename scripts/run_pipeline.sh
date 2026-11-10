#!/usr/bin/env bash
# End-to-end DeepTE-Peptide pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash scripts/run_pipeline.sh [config] [samples.tsv]"
    echo ""
    echo "Runs the full DeepTE-Peptide pipeline:"
    echo "  1. Download public data (TCGA, GTEx, IEDB, Dfam, etc.)"
    echo "  2. Preprocess (TE quantification, ORF calling, Ribo-seq, MHC binding)"
    echo "  3. Build feature tensors"
    echo "  4. Train 5-fold cross-validation"
    echo "  5. Evaluate on held-out test set"
    echo "  6. Generate publication-quality figures"
    echo "  7. Run R statistical analyses"
    echo ""
    exit 0
fi

CONFIG="${1:-config/default.yaml}"
SAMPLES="${2:-data/processed/samples.tsv}"
STAGES="${STAGES:-download,preprocess,train,eval,figures,r}"

IFS=',' read -ra STAGE_ARRAY <<< "$STAGES"

for stage in "${STAGE_ARRAY[@]}"; do
    case "$stage" in
        download)
            echo "=== Stage 1: Download data ==="
            bash scripts/download_data.sh
            ;;
        preprocess)
            echo "=== Stage 2: Preprocess ==="
            bash scripts/preprocess.sh "$CONFIG" "$SAMPLES"
            ;;
        train)
            echo "=== Stage 3: Train ==="
            bash scripts/train.sh "$CONFIG"
            ;;
        eval)
            echo "=== Stage 4: Evaluate ==="
            bash scripts/evaluate.sh "$CONFIG"
            ;;
        figures)
            echo "=== Stage 5: Generate figures ==="
            python figures/generate_figures.py --config "$CONFIG"
            ;;
        r)
            echo "=== Stage 6: R statistical analyses ==="
            Rscript R/differential_analysis.R || true
            Rscript R/survival_analysis.R || true
            Rscript R/enrichment_analysis.R || true
            ;;
        *)
            echo "Unknown stage: $stage"
            exit 1
            ;;
    esac
done

echo ""
echo "=== Pipeline complete ==="
echo "Results: results/"
echo "Figures: results/figures/"
echo "Reports: results/reports/"
