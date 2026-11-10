"""Unit tests for feature engineering."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepte.features.sequence_features import (
    compute_physicochemical_features,
    compute_positional_kmer_features,
    encode_peptide,
)


class TestPhysicochemicalFeatures:
    def test_features_finite(self):
        for pep in ["", "A", "M", "MAAARK", "MFILPQSTWY"]:
            feats = compute_physicochemical_features(pep)
            assert feats.shape == (12,)
            assert np.isfinite(feats).all()

    def test_charge_density(self):
        # Positively charged peptide
        pos = compute_physicochemical_features("KKKKR")
        # Negative peptide
        neg = compute_physicochemical_features("DDDEE")
        assert pos[10] > neg[10]  # charge_density


class TestKmerFeatures:
    def test_kmer_vector(self):
        vec = compute_positional_kmer_features("MART", k=3)
        # 20^3 = 8000 possible 3-mers
        assert vec.shape == (8000,)
        # MART has 2 3-mers (MAR, ART); after normalisation sum is 1.0
        assert abs(vec.sum() - 1.0) < 0.01


class TestEncodePeptide:
    def test_known_peptide(self):
        enc = encode_peptide("MAAA", max_len=10)
        assert enc[0] == 1   # BOS
        # indices: 0=PAD, 1=BOS, 2=EOS, 3=UNK, 4=CLS, 5=SEP, 6=A, 7=R, 8=N, 9=D, 10=C, 11=E
        # 12=Q, 13=G, 14=H, 15=I, 16=L, 17=K, 18=M, 19=F, 20=P, 21=S, 22=T, 23=W, 24=Y, 25=V
        assert enc[1] == 18  # M


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
