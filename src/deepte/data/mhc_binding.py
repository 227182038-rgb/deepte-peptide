"""MHC class I binding prediction wrapper."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from deepte.utils.logging import get_logger

logger = get_logger(__name__)


class MHCBindingPredictor:
    """Wraps NetMHCpan-4.1 (and MHCflurry as a fallback) for binding prediction."""

    def __init__(self, config: dict):
        self.cfg = config["data"]["mhc_binding"]
        self.lengths = self.cfg["length"]
        self.strong_threshold = self.cfg["threshold_strong_binder"]
        self.weak_threshold = self.cfg["threshold_weak_binder"]
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Detect available MHC binding predictor."""
        for cmd in ["netMHCpan", "mhcflurry-predict"]:
            try:
                subprocess.run([cmd, "-h"], capture_output=True, check=False, timeout=5)
                return cmd
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        logger.warning("No MHC binding backend available; using random baseline")
        return "random"

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def predict_batch(self, peptides: list[str], hla_allele: str) -> list[dict]:
        """Predict MHC binding for a batch of peptides.

        Parameters
        ----------
        peptides
            List of peptide sequences (8-11 AA).
        hla_allele
            HLA allele in NetMHCpan format, e.g. "HLA-A02:01".

        Returns
        -------
        List[Dict]
            One dict per peptide with keys: peptide, score (0-1), rank (%).
        """
        if not peptides:
            return []
        if self._backend == "netMHCpan":
            return self._netmhcpan(peptides, hla_allele)
        if self._backend == "mhcflurry-predict":
            return self._mhcflurry(peptides, hla_allele)
        return self._random_baseline(peptides, hla_allele)

    # -----------------------------------------------------------------
    # Backends
    # -----------------------------------------------------------------
    def _netmhcpan(self, peptides: list[str], hla: str) -> list[dict]:
        with tempfile.TemporaryDirectory() as d:
            pep_file = Path(d) / "peptides.txt"
            pep_file.write_text("\n".join(peptides))
            out_file = Path(d) / "out.tsv"

            cmd = [
                "netMHCpan",
                "-p",
                str(pep_file),
                "-a",
                hla,
                "-xls",
                "-xlsfile",
                str(out_file),
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=3600)
            except subprocess.CalledProcessError as e:
                logger.error(f"netMHCpan failed: {e.stderr.decode()[:500]}")
                return self._random_baseline(peptides, hla)

            df = pd.read_csv(out_file, sep="\t", skiprows=1)
            results = []
            for _, row in df.iterrows():
                results.append(
                    {
                        "peptide": row["Peptide"],
                        "score": float(
                            row["Score_BA"] if "Score_BA" in row else row.get("Score", 0.0)
                        ),
                        "rank": float(row.get("Rank", 100.0)),
                    }
                )
            return results

    def _mhcflurry(self, peptides: list[str], hla: str) -> list[dict]:
        try:
            from mhcflurry import Class1AffinityPredictor

            predictor = Class1AffinityPredictor.load()
        except Exception as e:
            logger.error(f"MHCflurry load failed: {e}")
            return self._random_baseline(peptides, hla)

        df = predictor.predict_to_dataframe(peptide=peptides, allele=hla)
        results = []
        for _, row in df.iterrows():
            # Convert IC50 nM to 1-log50k affinity score (NetMHCpan convention)
            ic50 = float(row["prediction"])
            score = 1.0 - np.log10(ic50) / np.log10(50000)
            results.append(
                {
                    "peptide": row["peptide"],
                    "score": float(np.clip(score, 0.0, 1.0)),
                    "rank": float(row.get("prediction_percentile", 100.0)),
                }
            )
        return results

    def _random_baseline(self, peptides: list[str], hla: str) -> list[dict]:
        rng = np.random.default_rng(seed=len(peptides))
        return [
            {
                "peptide": p,
                "score": float(rng.random()),
                "rank": float(rng.uniform(0.0, 100.0)),
            }
            for p in peptides
        ]
