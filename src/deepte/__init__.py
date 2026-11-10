"""DeepTE-Peptide: Multi-modal deep learning for pan-cancer TE-peptide immunogenicity prediction."""

__version__ = "0.1.0"
__author__ = "DeepTE-Peptide Research Team"
__license__ = "MIT"

from deepte.utils.io import load_config, set_seed
from deepte.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["__version__", "load_config", "logger", "set_seed"]
