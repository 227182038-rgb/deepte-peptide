"""Expression and Ribo-seq feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ExpressionFeatureBuilder:
    """Construct expression-derived features from TE quantification."""

    def __init__(self, n_cancer_types: int = 33, embedding_dim: int = 32):
        self.n_cancer_types = n_cancer_types
        self.embedding_dim = embedding_dim
        # Cancer type -> integer index
        self._cancer_to_idx: dict[str, int] = {}

    def fit_cancer_types(self, cancer_types: list[str]) -> None:
        unique = sorted(set(cancer_types))
        self._cancer_to_idx = {c: i for i, c in enumerate(unique)}
        self.n_cancer_types = len(unique)

    def cancer_to_idx(self, cancer: str) -> int:
        return self._cancer_to_idx.get(cancer, -1)

    def build_features(
        self,
        sample_id: str,
        cancer_type: str,
        te_expression: float,
        riboseq_psite_count: int,
        riboseq_periodicity: float,
    ) -> dict[str, np.ndarray]:
        """Return per-modality feature arrays for a single TEP candidate.

        Returns
        -------
        Dict[str, np.ndarray]
            Keys: "expression", "translation", "te_features"
        """
        # Expression feature vector
        expr_vec = np.array(
            [
                te_expression,
                np.log1p(max(te_expression, 0.0)),
                float(te_expression > 1.0),  # expressed?
                float(te_expression > 4.0),  # highly expressed?
            ],
            dtype=np.float32,
        )

        # Translation features (Ribo-seq evidence)
        trans_vec = np.array(
            [
                float(riboseq_psite_count),
                np.log1p(max(riboseq_psite_count, 0)),
                riboseq_periodicity,
                float(riboseq_psite_count > 5),
                float(riboseq_periodicity > 0.5),
                float(riboseq_psite_count > 0) * riboseq_periodicity,
                float(riboseq_psite_count >= 10),
                float(riboseq_periodicity >= 0.7),
            ],
            dtype=np.float32,
        )

        # TE biophysical / evolutionary features (placeholder values;
        # populated by build_features.py)
        te_feat = np.zeros(64, dtype=np.float32)

        return {
            "expression": expr_vec,
            "translation": trans_vec,
            "te_features": te_feat,
            "cancer_idx": np.array(self.cancer_to_idx(cancer_type), dtype=np.int64),
        }

    def aggregate_per_cancer_type(
        self,
        expression_df: pd.DataFrame,
        sample_col: str = "sample_id",
        cancer_col: str = "cancer_type",
        te_col: str = "te_id",
        expr_col: str = "log2_tpm1",
    ) -> pd.DataFrame:
        """Return per-cancer-type mean expression per TE.

        Useful as a population-level prior when per-sample Ribo-seq is missing.
        """
        return (
            expression_df.groupby([cancer_col, te_col])[expr_col]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
