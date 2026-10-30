import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tetris_rl.training.io_utils import (
    append_jsonl,
    observation_to_tensor,
    write_json,
)


def test_observation_to_tensor_adds_batch_axis_without_copying_values() -> None:
    observation = np.arange(24, dtype=np.float32).reshape(2, 3, 4)[:, :, ::-1]

    tensor = observation_to_tensor(observation, torch.device("cpu"))

    assert tensor.shape == (1, 2, 3, 4)
    assert tensor.dtype == torch.float32
    np.testing.assert_array_equal(tensor[0].numpy(), observation)


def test_write_json_is_sorted_indented_and_rejects_nan(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"

    write_json(path, {"b": 1, "a": [2]})

    assert path.read_text(encoding="utf-8") == '{\n  "a": [\n    2\n  ],\n  "b": 1\n}\n'
    with pytest.raises(ValueError):
        write_json(path, {"loss": math.nan})


def test_append_jsonl_keeps_earlier_lines_and_allows_nan(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"

    append_jsonl(path, {"update": 1, "loss": 0.5})
    append_jsonl(path, {"update": 2, "loss": math.nan})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"loss": 0.5, "update": 1}
    assert math.isnan(json.loads(lines[1])["loss"])
