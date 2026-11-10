"""CLI entry point for running inference with DeepTE-Peptide."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from deepte.features.expression_features import ExpressionFeatureBuilder
from deepte.features.sequence_features import (
    encode_peptide,
    onehot_encode_mhc_allele,
)
from deepte.models.te_peptide_net import DeepTEPeptideNet
from deepte.utils.io import load_config
from deepte.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def predict_dataframe(
    df: pd.DataFrame,
    model: DeepTEPeptideNet,
    config: dict,
    device: torch.device,
    max_peptide_len: int = 25,
    max_mhc_len: int = 30,
) -> pd.DataFrame:
    """Predict immunogenicity scores for candidate peptides in a DataFrame."""
    model.eval()
    model.to(device)

    expr_builder = ExpressionFeatureBuilder()
    cancer_types = (
        df["cancer_type"].unique().tolist()
        if "cancer_type" in df.columns
        else ["PANCAN"]
    )
    expr_builder.fit_cancer_types(cancer_types)

    n = len(df)
    results = df.copy()

    mhc_preds = []
    tcell_preds = []
    immuno_preds = []

    batch_size = config.get("evaluation", {}).get("batch_size", 64)
    with torch.no_grad():
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch_df = df.iloc[start_idx:end_idx]
            b = len(batch_df)

            seq_t = torch.zeros(b, max_peptide_len, dtype=torch.long)
            hla_t = torch.zeros(b, max_mhc_len, dtype=torch.long)
            expr_t = torch.zeros(b, 4, dtype=torch.float32)
            trans_t = torch.zeros(b, 1, 8, dtype=torch.float32)
            te_feat_t = torch.zeros(b, 64, dtype=torch.float32)
            cancer_t = torch.zeros(b, dtype=torch.long)

            for i, (_, row) in enumerate(batch_df.iterrows()):
                pep = str(row.get("peptide_seq", ""))
                hla = str(row.get("hla_allele", "HLA-A*02:01"))
                cancer = str(row.get("cancer_type", "PANCAN"))

                seq_t[i] = torch.from_numpy(encode_peptide(pep, max_peptide_len)).long()
                hla_onehot = onehot_encode_mhc_allele(hla, max_mhc_len)
                hla_t[i] = torch.from_numpy(hla_onehot.argmax(axis=-1)).long()

                feats = expr_builder.build_features(
                    sample_id=str(row.get("sample_id", f"sample_{i}")),
                    cancer_type=cancer,
                    te_expression=float(row.get("te_expression", 0.0)),
                    riboseq_psite_count=int(row.get("riboseq_psite_count", 0)),
                    riboseq_periodicity=float(row.get("riboseq_periodicity", 0.0)),
                )
                expr_t[i] = torch.from_numpy(feats["expression"])
                trans_t[i, 0] = torch.from_numpy(feats["translation"])
                te_feat_t[i] = torch.from_numpy(feats["te_features"])
                c_idx = int(feats["cancer_idx"])
                cancer_t[i] = c_idx if c_idx >= 0 else 0

            batch = {
                "peptide_tokens": seq_t.to(device),
                "mhc_tokens": hla_t.to(device),
                "peptide_mask": (seq_t == 0).to(device),
                "mhc_mask": (hla_t == 0).to(device),
                "te_expression": expr_t.to(device),
                "ribo_features": trans_t.to(device),
                "te_features": te_feat_t.to(device),
                "cancer_idx": cancer_t.to(device),
            }

            out = model(batch)
            mhc_preds.extend(out["mhc_binding"].cpu().numpy().tolist())
            tcell_preds.extend(out["tcell_recognition"].cpu().numpy().tolist())
            immuno_preds.extend(out["immunogenicity_score"].cpu().numpy().tolist())

    results["pred_mhc_binding"] = mhc_preds
    results["pred_tcell_recognition"] = tcell_preds
    results["pred_immunogenicity_score"] = immuno_preds
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict immunogenicity of TE-derived peptides")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    parser.add_argument(
        "--checkpoint", default=None, help="Path to model checkpoint .ckpt (optional)"
    )
    parser.add_argument("--input", required=True, help="Input CSV/TSV containing peptide sequences")
    parser.add_argument("--output", required=True, help="Output CSV path for predictions")
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    setup_logging(level="INFO")
    cfg = load_config(args.config)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    model = DeepTEPeptideNet(cfg)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
            logger.info(f"Loaded weights from {ckpt_path}")
        else:
            logger.warning(f"Checkpoint {ckpt_path} not found. Running with initialized weights.")
    else:
        logger.info("No checkpoint provided. Running with initialized weights.")

    in_path = Path(args.input)
    sep = "\t" if in_path.suffix in [".tsv", ".txt"] else ","
    df = pd.read_csv(in_path, sep=sep)
    logger.info(f"Loaded {len(df)} candidate peptides from {in_path}")

    preds_df = predict_dataframe(df, model, cfg, device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(out_path, index=False)
    logger.info(f"Predictions written to {out_path}")


if __name__ == "__main__":
    main()
