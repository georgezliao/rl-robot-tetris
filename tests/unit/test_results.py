import csv
import json
from pathlib import Path

import pytest

from tetris_rl.training.evaluate import EpisodeResult, EvaluationResult
from tetris_rl.training.results import export_results, results_directory_name


def evaluation_result(tmp_path: Path) -> EvaluationResult:
    episode = EpisodeResult(
        episode_index=0,
        episode_id="abc",
        seed=10000,
        steps=12,
        terminated=True,
        truncated=False,
        episode_return=1.0,
        score=40,
        lines=1,
        pieces=5,
        recording_directory=str(tmp_path / "local-only"),
    )
    return EvaluationResult(
        run_directory=tmp_path / "experiments" / "visual_reference_ppo-20261030T0Z",
        episodes=(episode,),
        summary={"episode_count": 1, "lines_per_100_pieces": 20.0},
    )


def test_directory_name_combines_date_device_and_run() -> None:
    assert (
        results_directory_name("2026-10-30", "Apple M3 Pro", "visual-run")
        == "2026-10-30-apple-m3-pro-visual-run"
    )


def test_export_writes_summary_episodes_and_provenance(tmp_path: Path) -> None:
    provenance = {"device": {"type": "cpu", "description": "Apple M3"}}

    directory = export_results(
        evaluation_result(tmp_path),
        provenance,
        tmp_path / "results",
        date="2026-10-30",
    )

    assert directory.name == "2026-10-30-apple-m3-visual_reference_ppo-20261030T0Z"
    assert sorted(path.name for path in directory.iterdir()) == [
        "episodes.csv",
        "provenance.json",
        "summary.json",
    ]
    assert (
        json.loads((directory / "summary.json").read_text())["lines_per_100_pieces"]
        == 20.0
    )
    assert json.loads((directory / "provenance.json").read_text()) == provenance
    with (directory / "episodes.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows == [
        {
            "episode_index": "0",
            "seed": "10000",
            "steps": "12",
            "terminated": "True",
            "truncated": "False",
            "episode_return": "1.0",
            "score": "40",
            "lines": "1",
            "pieces": "5",
            "episode_id": "abc",
        }
    ]


def test_export_refuses_to_overwrite(tmp_path: Path) -> None:
    arguments = (
        evaluation_result(tmp_path),
        {"device": {"description": "cpu"}},
        tmp_path / "results",
    )
    export_results(*arguments, date="2026-10-30")

    with pytest.raises(FileExistsError):
        export_results(*arguments, date="2026-10-30")
