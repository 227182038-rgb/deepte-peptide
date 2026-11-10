"""End-to-end preprocessing: TE quantification, ORF calling, Ribo-seq P-site, MHC binding."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from Bio.Seq import Seq
from tqdm import tqdm

from deepte.data.mhc_binding import MHCBindingPredictor
from deepte.data.riboseq_parser import RiboSeqParser
from deepte.data.te_annotation import TEAnnotator
from deepte.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CandidateTEP:
    """A candidate transposable-element-derived peptide."""

    peptide_id: str
    peptide_seq: str
    te_family: str
    te_superfamily: str
    te_locus: str  # chr:start-end:strand
    orf_start: int  # 1-based, transcript-relative
    orf_end: int
    transcript_id: str
    cancer_type: str
    sample_id: str
    hla_allele: str
    te_expression: float  # log2(TPM+1)
    riboseq_psite_count: int
    riboseq_periodicity: float


class Preprocessor:
    """Top-level preprocessing pipeline."""

    def __init__(self, config: dict):
        self.cfg = config
        self.raw_dir = Path(config["paths"]["raw_data"])
        self.processed_dir = Path(config["paths"]["processed_data"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.te_annotator = TEAnnotator(config)
        self.ribo_parser = RiboSeqParser(config)
        self.mhc_predictor = MHCBindingPredictor(config)

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------
    def run(self, samples: list[dict]) -> Path:
        """Run full preprocessing for a list of sample metadata dicts.

        Parameters
        ----------
        samples
            Each dict has keys: sample_id, cancer_type, hla_alleles,
            rna_bam, ribo_bam (optional).

        Returns
        -------
        Path
            Output parquet file with one row per candidate TEP.
        """
        logger.info(f"Preprocessing {len(samples)} samples")
        all_records: list[dict] = []

        for s in tqdm(samples, desc="Preprocessing samples"):
            try:
                records = self._process_one_sample(s)
                all_records.extend(records)
            except Exception as e:
                logger.error(f"Sample {s['sample_id']} failed: {e}")

        df = pd.DataFrame(all_records)
        out_path = self.processed_dir / "candidate_teps.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(df):,} candidate TEPs -> {out_path}")
        return out_path

    # -----------------------------------------------------------------
    # Per-sample pipeline
    # -----------------------------------------------------------------
    def _process_one_sample(self, sample: dict) -> list[dict]:
        # Step 1: TE quantification via TEtranscripts
        te_counts = self._quantify_te_expression(sample)

        # Step 2: Identify TE-derived transcripts
        te_transcripts = self.te_annotator.get_te_transcripts(sample["sample_id"])

        # Step 3: ORF calling on TE transcripts (6-frame)
        candidate_peptides: list[tuple[str, str, int, int]] = (
            []
        )  # (transcript_id, pep_seq, start, end)
        for tx_id, tx_seq in te_transcripts:
            for pep, start, end in _call_orfs(tx_seq, sample["sample_id"], tx_id):
                candidate_peptides.append((tx_id, pep, start, end))

        # Step 4: Ribo-seq P-site alignment
        ribo_evidence = self.ribo_parser.parse_sample(sample.get("ribo_bam"))

        # Step 5: MHC binding prediction
        for hla in sample["hla_alleles"]:
            peptides_only = [p[1] for p in candidate_peptides]
            binding_preds = self.mhc_predictor.predict_batch(peptides_only, hla)

            records = []
            for (tx_id, pep, start, end), binding in zip(
                candidate_peptides, binding_preds, strict=False
            ):
                records.append(
                    {
                        "peptide_id": _make_peptide_id(pep, sample["sample_id"], hla),
                        "peptide_seq": pep,
                        "te_family": self.te_annotator.lookup_family(tx_id),
                        "te_superfamily": self.te_annotator.lookup_superfamily(tx_id),
                        "te_locus": self.te_annotator.lookup_locus(tx_id),
                        "orf_start": start,
                        "orf_end": end,
                        "transcript_id": tx_id,
                        "cancer_type": sample["cancer_type"],
                        "sample_id": sample["sample_id"],
                        "hla_allele": hla,
                        "te_expression": te_counts.get(tx_id, 0.0),
                        "riboseq_psite_count": ribo_evidence.get(tx_id, {}).get("psite_count", 0),
                        "riboseq_periodicity": ribo_evidence.get(tx_id, {}).get("periodicity", 0.0),
                        "mhc_binding_score": binding["score"],
                        "mhc_binding_rank": binding["rank"],
                    }
                )
            return records

    # -----------------------------------------------------------------
    # TE expression quantification
    # -----------------------------------------------------------------
    def _quantify_te_expression(self, sample: dict) -> dict[str, float]:
        """Run TEtranscripts on a single RNA-seq BAM; return log2(TPM+1)."""
        bam = sample["rna_bam"]
        out_dir = self.processed_dir / "tequant" / sample["sample_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "TEtranscripts",
            "-t",
            str(bam),
            "--mode",
            "multi",
            "--format",
            "BAM",
            "--stranded",
            "no",
            "--TE",
            str(self.cfg["data"]["te_annotations"]["repeatmasker_library"]),
            "--project",
            sample["sample_id"],
            "--outdir",
            str(out_dir),
        ]
        logger.debug(f"Running TEtranscripts: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError:
            logger.warning("TEtranscripts not available; skipping quantification")
            return {}

        cnt = pd.read_csv(out_dir / f"{sample['sample_id']}.cntTable", sep="\t")
        # Convert raw counts to log2(TPM+1)
        total = cnt["count"].sum()
        cnt["tpm"] = cnt["count"] / max(total, 1) * 1e6
        cnt["log2_tpm1"] = (cnt["tpm"] + 1).apply(lambda x: _safe_log2(x))
        return dict(zip(cnt["gene/TE"], cnt["log2_tpm1"], strict=False))


# ---------------------------------------------------------------------
# ORF calling helpers
# ---------------------------------------------------------------------
def _safe_log2(x: float) -> float:
    import math

    return math.log2(x) if x > 0 else 0.0


def _call_orfs(
    seq: str, sample_id: str, tx_id: str, min_aa: int = 8, max_aa: int = 25
) -> Iterator[tuple[str, int, int]]:
    """6-frame ORF caller; yields peptides of length [min_aa, max_aa].

    For each candidate peptide yielded, returns (peptide_seq, start_1based, end_1based).
    """
    if len(seq) < min_aa * 3:
        return

    for frame in range(3):
        sub = seq[frame:]
        n_codons = len(sub) // 3
        for i in range(n_codons):
            codon = sub[i * 3 : i * 3 + 3]
            if codon.upper() == "ATG":
                # Find next stop
                pep = []
                for j in range(i, n_codons):
                    c = sub[j * 3 : j * 3 + 3].upper()
                    if c in {"TAA", "TAG", "TGA"}:
                        break
                    pep.append(_translate_codon(c))
                pep_str = "".join(pep)
                if min_aa <= len(pep_str) <= max_aa:
                    start_1b = frame + i * 3 + 1
                    end_1b = start_1b + len(pep_str) * 3 - 1
                    yield pep_str, start_1b, end_1b

    # Reverse complement frame
    rc = str(Seq(seq).reverse_complement())
    for frame in range(3):
        sub = rc[frame:]
        n_codons = len(sub) // 3
        for i in range(n_codons):
            codon = sub[i * 3 : i * 3 + 3]
            if codon.upper() == "ATG":
                pep = []
                for j in range(i, n_codons):
                    c = sub[j * 3 : j * 3 + 3].upper()
                    if c in {"TAA", "TAG", "TGA"}:
                        break
                    pep.append(_translate_codon(c))
                pep_str = "".join(pep)
                if min_aa <= len(pep_str) <= max_aa:
                    start_1b = frame + i * 3 + 1
                    end_1b = start_1b + len(pep_str) * 3 - 1
                    yield pep_str, start_1b, end_1b


_CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


def _translate_codon(codon: str) -> str:
    return _CODON_TABLE.get(codon.upper(), "X")


def _make_peptide_id(seq: str, sample_id: str, hla: str) -> str:
    h = hashlib.md5(f"{seq}|{sample_id}|{hla}".encode()).hexdigest()[:12]
    return f"TEP_{h}"


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="DeepTE-Peptide preprocessing")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--samples", required=True, help="TSV file with sample metadata")
    args = parser.parse_args()

    from deepte.utils.io import load_config

    cfg = load_config(args.config)
    samples_df = pd.read_csv(args.samples, sep="\t")
    samples = samples_df.to_dict("records")

    pre = Preprocessor(cfg)
    pre.run(samples)


if __name__ == "__main__":
    main()
