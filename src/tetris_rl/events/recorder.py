"""Validated JSON/JSONL storage for generic episode records."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from tetris_rl.events.schema import EpisodeManifest, StepEvent

MANIFEST_FILENAME = "manifest.json"
STEPS_FILENAME = "steps.jsonl"


@dataclass(frozen=True, slots=True)
class RecordedEpisode:
    manifest: EpisodeManifest
    steps: tuple[StepEvent, ...]


class EpisodeRecorder:
    """Write one manifest and an ordered JSONL stream without silent overwrite."""

    def __init__(
        self,
        output_directory: Path | str,
        manifest: EpisodeManifest,
        *,
        overwrite: bool = False,
    ) -> None:
        self.output_directory = Path(output_directory)
        self.manifest = manifest
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_directory / MANIFEST_FILENAME
        self.steps_path = self.output_directory / STEPS_FILENAME

        if not overwrite and (self.manifest_path.exists() or self.steps_path.exists()):
            raise FileExistsError(
                f"Recording already exists in {self.output_directory}"
            )
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(
                manifest.to_dict(),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        self._steps_handle = self.steps_path.open("w", encoding="utf-8")

        self._next_step = 0
        self._cumulative_reward = 0.0
        self._terminal_seen = False
        self._closed = False

    def record(self, event: StepEvent) -> None:
        if self._closed:
            raise RuntimeError("Cannot record after the recorder is closed")
        if self._terminal_seen:
            raise ValueError("Cannot record a step after a terminal event")
        if event.episode_id != self.manifest.episode_id:
            raise ValueError("Event episode_id does not match the manifest")
        if event.step_index != self._next_step:
            raise ValueError(
                f"Expected step_index {self._next_step}, received {event.step_index}"
            )
        if event.action >= len(self.manifest.action_spec):
            raise ValueError(
                f"Action {event.action} is outside the manifest action_spec"
            )
        expected_label = self.manifest.action_spec[event.action]
        if event.action_label != expected_label:
            raise ValueError(
                f"Action label {event.action_label!r} does not match {expected_label!r}"
            )
        expected_cumulative = self._cumulative_reward + event.reward
        if not math.isclose(
            event.cumulative_reward,
            expected_cumulative,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "cumulative_reward does not equal the preceding total plus reward"
            )

        json.dump(event.to_dict(), self._steps_handle, sort_keys=True, allow_nan=False)
        self._steps_handle.write("\n")
        self._steps_handle.flush()
        self._next_step += 1
        self._cumulative_reward = event.cumulative_reward
        self._terminal_seen = event.done

    def close(self) -> None:
        if not self._closed:
            self._steps_handle.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def read_episode(directory: Path | str) -> RecordedEpisode:
    """Load and validate a recorded episode from a directory."""
    directory = Path(directory)
    with (directory / MANIFEST_FILENAME).open(encoding="utf-8") as handle:
        manifest = EpisodeManifest.from_dict(json.load(handle))

    steps: list[StepEvent] = []
    cumulative_reward = 0.0
    with (directory / STEPS_FILENAME).open(encoding="utf-8") as handle:
        for expected_index, line in enumerate(handle):
            event = StepEvent.from_dict(json.loads(line))
            if event.episode_id != manifest.episode_id:
                raise ValueError("Recorded event does not match manifest episode_id")
            if event.step_index != expected_index:
                raise ValueError(
                    f"Expected recorded step {expected_index}, got {event.step_index}"
                )
            if event.action >= len(manifest.action_spec):
                raise ValueError(f"Recorded action is invalid at step {expected_index}")
            if event.action_label != manifest.action_spec[event.action]:
                raise ValueError(
                    f"Recorded action label is invalid at step {expected_index}"
                )
            if steps and steps[-1].done:
                raise ValueError("Recorded episode contains steps after termination")
            cumulative_reward += event.reward
            if not math.isclose(
                event.cumulative_reward,
                cumulative_reward,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(f"Invalid cumulative reward at step {expected_index}")
            steps.append(event)

    return RecordedEpisode(manifest=manifest, steps=tuple(steps))
