"""Peptide and MHC allele sequence feature encoding."""

from __future__ import annotations

import numpy as np

# 20 amino acids + special tokens: <PAD>, <BOS>, <EOS>, <UNK>, <CLS>, <SEP>
AA_VOCAB: list[str] = [
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>",
    "<CLS>",
    "<SEP>",
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(AA_VOCAB)}
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# Kyte-Doolittle hydropathy
KYTE_DOOLITTLE: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "E": -3.5,
    "Q": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

# Isoelectric point (pI) per residue (Lehninger)
PI_VALUES: dict[str, float] = {
    "A": 6.00,
    "R": 10.76,
    "N": 5.41,
    "D": 2.77,
    "C": 5.07,
    "E": 3.22,
    "Q": 5.65,
    "G": 5.97,
    "H": 7.59,
    "I": 6.02,
    "L": 5.98,
    "K": 9.74,
    "M": 5.74,
    "F": 5.48,
    "P": 6.30,
    "S": 5.68,
    "T": 5.60,
    "W": 5.89,
    "Y": 5.66,
    "V": 5.96,
}

# Molecular weight (Da)
MW_VALUES: dict[str, float] = {
    "A": 89.09,
    "R": 174.20,
    "N": 132.12,
    "D": 133.10,
    "C": 121.16,
    "E": 147.13,
    "Q": 146.15,
    "G": 75.03,
    "H": 155.16,
    "I": 131.17,
    "L": 131.17,
    "K": 146.19,
    "M": 149.21,
    "F": 165.19,
    "P": 115.13,
    "S": 105.09,
    "T": 119.12,
    "W": 204.23,
    "Y": 181.19,
    "V": 117.15,
}


def encode_peptide(peptide: str, max_len: int = 25) -> np.ndarray:
    """Tokenize and pad a peptide sequence.

    Returns
    -------
    np.ndarray
        1-D int32 array of length max_len, padded with PAD_IDX.
    """
    tokens = [AA_TO_IDX.get(a, UNK_IDX) for a in peptide.upper()]
    tokens = [BOS_IDX, *tokens, EOS_IDX]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [PAD_IDX] * (max_len - len(tokens))
    return np.array(tokens, dtype=np.int32)


def onehot_encode_mhc_allele(allele: str, max_len: int = 30) -> np.ndarray:
    """One-hot encode the HLA allele string for sequence-based encoding.

    Parameters
    ----------
    allele
        e.g. "HLA-A*02:01"
    max_len
        Max character length (padded with zeros).

    Returns
    -------
    np.ndarray
        Shape (max_len, 49) one-hot encoding (alphabet = printable ASCII subset, padded to 49).
    """
    # 49-character alphabet (matches MHCAlleleEncoder embedding size)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*-:./+=;_@ #&"
    assert len(alphabet) == 49, f"Alphabet has {len(alphabet)} chars, expected 49"
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    encoding = np.zeros((max_len, len(alphabet)), dtype=np.float32)
    for i, c in enumerate(allele.upper()[:max_len]):
        if c in char_to_idx:
            encoding[i, char_to_idx[c]] = 1.0
    return encoding


def compute_physicochemical_features(peptide: str) -> np.ndarray:
    """Compute 12 biophysical descriptors for a peptide.

    Returns
    -------
    np.ndarray
        1-D float32 array of length 12.
    """
    pep = peptide.upper()
    n = max(len(pep), 1)
    if not pep:
        # Return zeros for empty peptide
        return np.zeros(12, dtype=np.float32)
    hydropathy = float(np.mean([KYTE_DOOLITTLE.get(a, 0.0) for a in pep]))
    pi_mean = float(np.mean([PI_VALUES.get(a, 7.0) for a in pep]))
    mw_total = float(np.sum([MW_VALUES.get(a, 110.0) for a in pep]))

    aromatic = sum(1 for a in pep if a in {"F", "W", "Y"})
    positive = sum(1 for a in pep if a in {"K", "R", "H"})
    negative = sum(1 for a in pep if a in {"D", "E"})
    polar = sum(1 for a in pep if a in {"S", "T", "N", "Q", "C"})
    hydrophobic = sum(1 for a in pep if a in {"A", "V", "L", "I", "M", "F", "W", "Y"})
    proline = sum(1 for a in pep if a == "P")
    cysteine = sum(1 for a in pep if a == "C")

    net_charge = positive - negative
    charge_density = net_charge / n
    length = float(len(pep))

    features = np.array(
        [
            hydropathy,
            pi_mean,
            mw_total / 1000.0,
            aromatic / n,
            positive / n,
            negative / n,
            polar / n,
            hydrophobic / n,
            proline / n,
            cysteine / n,
            charge_density,
            length,
        ],
        dtype=np.float32,
    )
    return features


def compute_positional_kmer_features(peptide: str, k: int = 3) -> np.ndarray:
    """Compute 3-mer frequency vector."""
    from itertools import product

    # All 20 standard amino acids
    amino = "ARNDCQGHILKMFPSTWYV"  # 19 chars - missing 'E'
    amino = "ARNDCEQGHILKMFPSTWYV"  # 20 chars - include 'E'
    assert len(amino) == 20, f"Amino has {len(amino)} chars"
    kmers = ["".join(p) for p in product(amino, repeat=k)]
    kmer_to_idx = {km: i for i, km in enumerate(kmers)}
    vec = np.zeros(len(kmers), dtype=np.float32)
    for i in range(len(peptide) - k + 1):
        kmer = peptide[i : i + k].upper()
        if kmer in kmer_to_idx:
            vec[kmer_to_idx[kmer]] += 1
    if vec.sum() > 0:
        vec /= vec.sum()
    return vec
