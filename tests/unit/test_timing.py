import pytest
import torch

from tetris_rl.training.timing import Stopwatch, summarize_durations


def test_summary_is_in_milliseconds() -> None:
    summary = summarize_durations([0.001 * value for value in range(1, 11)])

    assert summary["count"] == 10
    assert summary["median_ms"] == pytest.approx(5.5)
    # Linear interpolation: 90th percentile of 1..10 is 9.1.
    assert summary["p90_ms"] == pytest.approx(9.1)
    assert summary["mean_ms"] == pytest.approx(5.5)


def test_summary_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        summarize_durations([])


def test_stopwatch_records_one_duration_per_section() -> None:
    stopwatch = Stopwatch(torch.device("cpu"))
    for _ in range(3):
        with stopwatch:
            pass

    assert len(stopwatch.durations) == 3
    assert all(duration >= 0.0 for duration in stopwatch.durations)
