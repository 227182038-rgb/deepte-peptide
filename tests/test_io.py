"""Unit tests for I/O and utility helpers."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepte.utils.io import hash_str, load_config, load_json, save_json, set_seed


def test_hash_str():
    h1 = hash_str("L1HS")
    h2 = hash_str("L1HS")
    h3 = hash_str("AluY")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 8


def test_save_and_load_json(tmp_path):
    p = tmp_path / "test.json"
    data = {
        "int": np.int64(42),
        "float": np.float32(3.14),
        "arr": np.array([1, 2, 3]),
        "tensor": torch.tensor([4.0, 5.0]),
        "path": Path("some/path"),
    }
    save_json(data, p)
    loaded = load_json(p)
    assert loaded["int"] == 42
    assert abs(loaded["float"] - 3.14) < 1e-4
    assert loaded["arr"] == [1, 2, 3]
    assert loaded["tensor"] == [4.0, 5.0]


def test_set_seed():
    set_seed(123, deterministic=False)
    v1 = torch.rand(5)
    set_seed(123, deterministic=False)
    v2 = torch.rand(5)
    assert torch.allclose(v1, v2)


def test_load_config(tmp_path):
    p = tmp_path / "test.yaml"
    p.write_text("seed: 42\npaths:\n  data: ${env:TEST_VAR,default_data}\n")
    cfg = load_config(p)
    assert cfg["seed"] == 42
    assert cfg["paths"]["data"] == "default_data"
