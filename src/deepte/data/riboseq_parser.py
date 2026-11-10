"""Parse Ribo-seq data: P-site alignment, periodicity, ORF translation evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Optional pysam for BAM parsing
try:
    import pysam

    HAS_PYSAM = True
except ImportError:
    pysam = None
    HAS_PYSAM = False

from deepte.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PSiteEvidence:
    """Per-transcript Ribo-seq P-site evidence."""

    transcript_id: str
    psite_count: int
    periodicity: float
    orf_coverage: float
    three_nt_periodicity: bool


class RiboSeqParser:
    """Parses Ribo-seq BAM files to extract per-transcript translation evidence."""

    def __init__(self, config: dict):
        self.cfg = config
        self.min_periodicity = config["data"]["riboseq_evidence"]["min_psite_count"]
        self.min_psite = config["data"]["riboseq_evidence"]["min_psite_count"]
        self._psite_offset = 12  # 5' end of ribosome-protected fragment -> P-site

    def parse_sample(self, ribo_bam: str | None) -> dict[str, dict]:
        """Parse a single sample's Ribo-seq BAM file.

        Parameters
        ----------
        ribo_bam
            Path to a Ribo-seq BAM file, or None if no Ribo-seq is available for this sample.

        Returns
        -------
        Dict[str, Dict]
            Mapping from transcript_id to a dict of P-site evidence.
        """
        if ribo_bam is None or not Path(ribo_bam).exists():
            logger.warning(f"Ribo-seq BAM not available: {ribo_bam}")
            return {}

        if not HAS_PYSAM:
            logger.warning("pysam not installed; skipping Ribo-seq parsing")
            return {}

        psite_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        total_reads: dict[str, int] = defaultdict(int)

        with pysam.AlignmentFile(ribo_bam, "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped or read.is_secondary:
                    continue
                tx_id = read.reference_name
                # Compute P-site position from 5' end of read
                p_site = read.reference_start + (
                    self._psite_offset if not read.is_reverse else -self._psite_offset
                )
                frame = p_site % 3
                psite_counts[tx_id][frame] += 1
                total_reads[tx_id] += 1

        out: dict[str, dict] = {}
        for tx_id, frames in psite_counts.items():
            n = total_reads[tx_id]
            if n < self.min_psite:
                continue
            counts = np.array([frames.get(f, 0) for f in range(3)])
            # Periodicity score = proportion of reads in dominant frame
            periodicity = float(counts.max() / counts.sum()) if counts.sum() > 0 else 0.0
            out[tx_id] = {
                "psite_count": int(n),
                "periodicity": periodicity,
                "frame_distribution": counts.tolist(),
                "three_frame_periodicity": bool(periodicity >= 0.5),
                "orf_coverage": float(counts.sum() / max(n, 1)),
            }

        logger.info(f"Parsed {len(out):,} transcripts from {ribo_bam}")
        return out

    def aggregate_across_samples(self, sample_bams: list[str]) -> pd.DataFrame:
        """Aggregate P-site evidence across multiple samples."""
        records = []
        for bam in sample_bams:
            ev = self.parse_sample(bam)
            for tx_id, vals in ev.items():
                vals["transcript_id"] = tx_id
                vals["source_bam"] = bam
                records.append(vals)
        return pd.DataFrame(records)
