"""Top-level feature builder that orchestrates all modality encoders."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from deepte.features.expression_features import ExpressionFeatureBuilder
from deepte.features.sequence_features import (
    compute_physicochemical_features,
    encode_peptide,
    onehot_encode_mhc_allele,
)
from deepte.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureBuilder:
    """Builds the per-TEP feature tensors used by DeepTEPeptideNet."""

    def __init__(self, config: dict, max_peptide_len: int = 25, max_mhc_len: int = 30):
        self.cfg = config
        self.max_peptide_len = max_peptide_len
        self.max_mhc_len = max_mhc_len
        self.expr_builder = ExpressionFeatureBuilder()

    def build_from_parquet(self, parquet_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
        """Convert a candidate TEP parquet file into per-modality .npz arrays.

        Returns
        -------
        Dict[str, Path]
            Paths to sequence, expression, translation, te_features, labels arrays.
        """
        df = pd.read_parquet(parquet_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Fit cancer types
        self.expr_builder.fit_cancer_types(df["cancer_type"].unique().tolist())

        # Allocate arrays
        n = len(df)
        seq_arr = np.zeros((n, self.max_peptide_len), dtype=np.int32)
        phys_arr = np.zeros((n, 12), dtype=np.float32)
        expr_arr = np.zeros((n, 4), dtype=np.float32)
        trans_arr = np.zeros((n, 8), dtype=np.float32)
        te_feat_arr = np.zeros((n, 64), dtype=np.float32)
        cancer_idx_arr = np.zeros((n,), dtype=np.int64)
        hla_arr = np.zeros((n, self.max_mhc_len, 49), dtype=np.float32)
        label_mhc = np.zeros((n,), dtype=np.float32)
        label_tcell = np.zeros((n,), dtype=np.float32)
        label_immuno = np.zeros((n,), dtype=np.float32)

        logger.info(f"Building features for {n:,} candidate TEPs")

        for i, row in tqdm(df.iterrows(), total=n, desc="Featurizing"):
            seq_arr[i] = encode_peptide(row["peptide_seq"], self.max_peptide_len)
            phys_arr[i] = compute_physicochemical_features(row["peptide_seq"])
            hla_arr[i] = onehot_encode_mhc_allele(row["hla_allele"], self.max_mhc_len)

            feats = self.expr_builder.build_features(
                sample_id=row["sample_id"],
                cancer_type=row["cancer_type"],
                te_expression=row.get("te_expression", 0.0),
                riboseq_psite_count=row.get("riboseq_psite_count", 0),
                riboseq_periodicity=row.get("riboseq_periodicity", 0.0),
            )
            expr_arr[i] = feats["expression"]
            trans_arr[i] = feats["translation"]
            te_feat_arr[i] = feats["te_features"]
            cancer_idx_arr[i] = feats["cancer_idx"]

            # Populate TE biophysical features in slots 0-9 of te_feat_arr
            # (rest stay zero for future expansion)
            self._populate_te_features(te_feat_arr, i, row)

            # Labels (if present)
            label_mhc[i] = float(row.get("mhc_binding_score", 0.0))
            label_tcell[i] = float(row.get("tcell_recognition", 0))
            label_immuno[i] = float(row.get("immunogenicity_score", 0.0))

        np.savez(
            out_dir / "features.npz",
            sequence=seq_arr,
            mhc_allele=hla_arr,
            physicochemical=phys_arr,
            expression=expr_arr,
            translation=trans_arr,
            te_features=te_feat_arr,
            cancer_idx=cancer_idx_arr,
            label_mhc_binding=label_mhc,
            label_tcell_recognition=label_tcell,
            label_immunogenicity=label_immuno,
        )

        # Save metadata
        df[
            [
                "peptide_id",
                "peptide_seq",
                "te_family",
                "te_superfamily",
                "cancer_type",
                "sample_id",
                "hla_allele",
            ]
        ].to_csv(out_dir / "metadata.tsv", sep="\t", index=False)

        logger.info(f"Saved features to {out_dir / 'features.npz'}")
        return {
            "features": out_dir / "features.npz",
            "metadata": out_dir / "metadata.tsv",
        }

    def _populate_te_features(self, arr: np.ndarray, i: int, row: pd.Series) -> None:
        """Fill the 64-d TE feature vector."""
        # One-hot encode superfamily
        sf_map = {"LINE": 0, "SINE": 1, "LTR": 2, "DNA": 3, "RC": 4, "Retroposon": 5}
        if row.get("te_superfamily") in sf_map:
            arr[i, sf_map[row["te_superfamily"]]] = 1.0

        # One-hot encode top families (50 slots starting at index 6)
        # In production we'd learn the family vocabulary from the training set
        # Deterministic hash across Python processes
        fam_str = str(row.get("te_family", "Unknown"))
        fam_hash = zlib.crc32(fam_str.encode("utf-8")) % 50
        arr[i, 6 + fam_hash] = 1.0

        # Scalar features (indices 56-63)
        arr[i, 56] = float(row.get("te_age_estimate", 50.0)) / 100.0  # normalised Myr
        arr[i, 57] = float(row.get("gc_content", 0.5))
        arr[i, 58] = float(row.get("length", 0.0)) / 10000.0
        arr[i, 59] = float(row.get("conservation_score", 0.0))
        arr[i, 60] = float(row.get("dna_methylation", 0.5))
        arr[i, 61] = float(row.get("h3k9me3_signal", 0.0))
        arr[i, 62] = float(row.get("h3k27me3_signal", 0.0))
        arr[i, 63] = float(row.get("chromatin_accessibility", 0.0))


def load_features(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load a features.npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}
