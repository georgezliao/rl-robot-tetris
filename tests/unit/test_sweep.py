import csv
import json
from pathlib import Path

from tetris_rl.training.evaluate import EvaluationResult
from tetris_rl.training.sweep import curve_rows, list_checkpoints, write_learning_curve


def test_checkpoints_are_listed_in_step_order_with_final_last(tmp_path: Path) -> None:
    directory = tmp_path / "run" / "checkpoints"
    directory.mkdir(parents=True)
    for name in ("final.pt", "step-100096.pt", "step-50048.pt", "notes.txt"):
        (directory / name).touch()

    names = [path.name for path in list_checkpoints(tmp_path / "run")]

    assert names == ["step-50048.pt", "step-100096.pt", "final.pt"]


def fake_evaluator(path: Path) -> EvaluationResult:
    steps = {"step-50048.pt": 50048, "final.pt": 100000}[path.name]
    pieces = steps / 10000  # a made-up curve that rises with training steps
    return EvaluationResult(
        run_directory=Path(f"experiments/evaluation/eval-{steps}"),
        episodes=(),
        summary={
            "checkpoint_environment_steps": steps,
            "episode_count": 100,
            "terminated_episodes": 100,
            "truncated_episodes": 0,
            "mean_survival_pieces": pieces,
            "lines_per_100_pieces": None if steps < 100000 else 2.5,
            "mean_lines": 0.0 if steps < 100000 else 0.25,
        },
    )


def test_curve_is_one_row_per_checkpoint_written_as_csv(tmp_path: Path) -> None:
    rows = curve_rows([Path("step-50048.pt"), Path("final.pt")], fake_evaluator)

    directory = write_learning_curve(
        rows,
        {"device": {"description": "Apple M4"}},
        tmp_path / "results",
        "visual_reference_ppo_terminal-seed0-20261031T0Z",
        date="2026-10-31",
    )

    assert directory.name == (
        "2026-10-31-apple-m4-curve-visual_reference_ppo_terminal-seed0-20261031T0Z"
    )
    with (directory / "learning_curve.csv").open(encoding="utf-8") as file:
        written = list(csv.DictReader(file))
    assert [row["checkpoint"] for row in written] == ["step-50048.pt", "final.pt"]
    assert [row["environment_steps"] for row in written] == ["50048", "100000"]
    assert [row["mean_survival_pieces"] for row in written] == ["5.0048", "10.0"]
    # An undefined rate (no pieces placed) is written as an empty cell.
    assert [row["lines_per_100_pieces"] for row in written] == ["", "2.5"]
    assert written[1]["evaluation_run"] == "eval-100000"
    assert json.loads((directory / "provenance.json").read_text()) == {
        "device": {"description": "Apple M4"}
    }
