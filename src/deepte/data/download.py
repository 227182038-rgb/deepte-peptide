"""Download public datasets used by DeepTE-Peptide."""

from __future__ import annotations

import gzip
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import ClassVar

from deepte.utils.logging import get_logger

logger = get_logger(__name__)


class DataDownloader:
    """Orchestrates download of all public data sources."""

    ENDPOINTS: ClassVar[dict[str, str]] = {
        "tcga_manifest": "https://api.gdc.cancer.gov/data/{}",  # requires UUID
        "gtex_v8": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna_seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz",
        "ccle": "https://depmap.org/portal/download/all/",
        "iedb_epitope": "https://www.iedb.org/downloader.php?file_id=34",
        "iedb_mhc_assay": "https://www.iedb.org/downloader.php?file_id=42",
        "iedb_tcell_assay": "https://www.iedb.org/downloader.php?file_id=43",
        "gwips_viz_human": "http://gwips.wyoming.edu/download/human/",
        "dfam_v37": "https://www.dfam.org/releases/Dfam_3.7/families/Dfam.h5.gz",
        "repeatmasker_db": (
            "https://www.repeatmasker.org/prebuilt/hg38/Homo_sapiens RepeatMasker database.tar.gz"
        ),
        "gencode_v44": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz",
        "netmhcpan_data": "https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/data.tar.gz",
        "hla_freq": "http://www.allelefrequencies.net/hla/",
    }

    def __init__(self, raw_dir: str | Path, external_dir: str | Path, max_workers: int = 4):
        self.raw_dir = Path(raw_dir)
        self.external_dir = Path(external_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

    def download_all(self, sources: list[str] | None = None) -> dict[str, Path]:
        """Download all or a subset of public datasets.

        Parameters
        ----------
        sources
            Subset of ENDPOINTS keys to download. If None, downloads all.

        Returns
        -------
        Dict[str, Path]
            Mapping from source key to local file path.
        """
        sources = sources or list(self.ENDPOINTS.keys())
        results: dict[str, Path] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._download_one, s): s for s in sources}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    results[key] = fut.result()
                except Exception as e:
                    logger.error(f"Failed to download {key}: {e}")

        logger.info(f"Downloaded {len(results)}/{len(sources)} sources successfully")
        return results

    def _download_one(self, source: str) -> Path:
        url = self.ENDPOINTS[source]
        out_path = self.raw_dir / f"{source}.bin"
        logger.info(f"Downloading {source} from {url}")

        # Resume support
        headers = {}
        mode = "wb"
        if out_path.exists():
            headers["Range"] = f"bytes={out_path.stat().st_size}-"
            mode = "ab"

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=300) as resp, open(out_path, mode) as f:
            shutil.copyfileobj(resp, f)

        logger.info(f"Saved {source} -> {out_path}")
        return out_path

    def decompress_gz(self, src: Path, dst: Path) -> None:
        with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    def fetch_tcga_rnaseq(self, cancer_type: str, manifest_path: Path) -> Path:
        """Use GDC Data Transfer Tool to fetch TCGA RNA-seq BAMs per cancer type."""
        out_dir = self.raw_dir / "tcga" / cancer_type
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["gdc-client", "download", "-m", str(manifest_path), "-d", str(out_dir)]
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return out_dir

    def fetch_gtex(self) -> Path:
        """Download GTEx v8 RNA-seq TPM matrix."""
        return self._download_one("gtex_v8")
