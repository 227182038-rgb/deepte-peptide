"""Unit tests for data preprocessing modules."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepte.data.preprocess import _call_orfs, _safe_log2, _translate_codon
from deepte.features.expression_features import ExpressionFeatureBuilder
from deepte.features.sequence_features import (
    AA_TO_IDX,
    AA_VOCAB,
    compute_physicochemical_features,
    encode_peptide,
    onehot_encode_mhc_allele,
)


class TestORFCalling:
    def test_translate_codon(self):
        assert _translate_codon("ATG") == "M"
        assert _translate_codon("TAA") == "*"
        assert _translate_codon("atg") == "M"

    def test_safe_log2(self):
        assert _safe_log2(0.0) == 0.0
        assert _safe_log2(1.0) == 0.0
        assert _safe_log2(2.0) == 1.0

    def test_call_orfs_simple(self):
        seq = "ATG" + "GCT" * 5 + "TAA"   # M-A-A-A-A-A-*
        peps = list(_call_orfs(seq, "test", "tx1", min_aa=3, max_aa=10))
        assert len(peps) >= 1
        assert peps[0][0] == "MAAAAA"

    def test_call_orfs_too_short(self):
        seq = "ATGTAA"
        peps = list(_call_orfs(seq, "test", "tx1", min_aa=8, max_aa=25))
        assert len(peps) == 0

    def test_call_orfs_six_frame(self):
        # Multiple ORFs in different frames
        seq = "ATGGCT" * 10 + "TAA" + "xATGAAATTTGGGAAATTTTAA"
        peps = list(_call_orfs(seq, "test", "tx1", min_aa=4, max_aa=25))
        # Should find at least the first ORF
        assert len(peps) >= 1


class TestSequenceFeatures:
    def test_aa_vocab(self):
        assert "<PAD>" in AA_VOCAB
        assert "A" in AA_TO_IDX
        assert AA_TO_IDX["<PAD>"] == 0

    def test_encode_peptide(self):
        pep = "MARTKQTAR"
        enc = encode_peptide(pep, max_len=15)
        assert enc.shape == (15,)
        assert enc[0] == 1  # BOS
        assert enc[len(pep) + 1] == 2  # EOS

    def test_encode_peptide_truncation(self):
        pep = "M" * 50
        enc = encode_peptide(pep, max_len=10)
        assert enc.shape == (10,)

    def test_mhc_onehot(self):
        enc = onehot_encode_mhc_allele("HLA-A*02:01", max_len=30)
        assert enc.shape == (30, 49)
        # Each position should be one-hot
        assert enc.sum() == len("HLA-A*02:01")

    def test_physicochemical_features(self):
        feats = compute_physicochemical_features("MAAARK")
        assert feats.shape == (12,)
        # Hydropathy should be finite
        assert np.isfinite(feats[0])
        # Length is 6
        assert feats[-1] == 6.0


class TestExpressionFeatures:
    def test_build_features(self):
        builder = ExpressionFeatureBuilder()
        builder.fit_cancer_types(["BRCA", "LUAD"])
        feats = builder.build_features(
            sample_id="S1", cancer_type="BRCA",
            te_expression=5.0, riboseq_psite_count=10,
            riboseq_periodicity=0.6,
        )
        assert "expression" in feats
        assert feats["expression"].shape == (4,)
        assert feats["translation"].shape == (8,)
        assert feats["te_features"].shape == (64,)
        assert feats["cancer_idx"] == 0  # BRCA sorts first

    def test_unknown_cancer(self):
        builder = ExpressionFeatureBuilder()
        builder.fit_cancer_types(["BRCA"])
        assert builder.cancer_to_idx("UNKNOWN") == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
