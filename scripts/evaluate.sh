#!/usr/bin/env bash
# Evaluate trained models on held-out test set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-config/default.yaml}"
FEATURES="${2:-data/processed/features/features.npz}"
METADATA="${3:-data/processed/features/metadata.tsv}"
N_FOLDS="${N_FOLDS:-5}"

mkdir -p results/reports

for fold in $(seq 0 $((N_FOLDS - 1))); do
    CKPT="results/models/best.ckpt"
    if [ ! -f "$CKPT" ]; then
        CKPT="results/models/last.ckpt"
    fi
    echo "=== Evaluating fold $fold ==="
    python -m deepte.evaluation.evaluate \
        --config "$CONFIG" \
        --features "$FEATURES" \
        --metadata "$METADATA" \
        --checkpoint "$CKPT" \
        --output "results/reports/evaluation_fold${fold}.json" \
        --fold "$fold"
done

echo "=== Aggregating fold evaluations ==="
python -c "
import json, glob, numpy as np
files = sorted(glob.glob('results/reports/evaluation_fold*.json'))
print(f'Found {len(files)} fold reports')
all_metrics = {}
for f in files:
    with open(f) as fh:
        m = json.load(fh)
    for head, vals in m.items():
        all_metrics.setdefault(head, {})
        for k, v in vals.items():
            all_metrics[head].setdefault(k, []).append(v)
agg = {}
for head, vals in all_metrics.items():
    agg[head] = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in vals.items()}
with open('results/reports/evaluation_aggregated.json', 'w') as f:
    json.dump(agg, f, indent=2)
print('Saved aggregated evaluation.')
"
