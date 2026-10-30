"""Python environment adapter for the original JavaFX Tetris game."""

from __future__ import annotations

import base64
import json
import selectors
import subprocess
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np

from tetris_rl.types import ControlAction, Info, Observation, StepResult


class BridgeError(RuntimeError):
    """Raised when the Java game bridge exits or violates its protocol."""


class JavaFxControlEnv:
    """Expose the pasted JavaFX game through the project's four-value RL API.

    One process is retained across episodes. Calls are synchronous, so the board
    cannot advance between actions and training is reproducible for a given seed.
    """

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        startup_timeout: float = 15.0,
        observation_size: tuple[int, int] = (128, 96),
        frame_stack: int = 4,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        default_command = [str(repo_root / "scripts" / "run_java_bridge.sh")]
        self.command = list(command or default_command)
        self.startup_timeout = startup_timeout
        if observation_size[0] < 1 or observation_size[1] < 1:
            raise ValueError("observation_size dimensions must be positive")
        if frame_stack < 1:
            raise ValueError("frame_stack must be positive")
        self.observation_size = observation_size
        self.frame_stack = frame_stack
        self._frames: deque[Observation] = deque(maxlen=frame_stack)
        self._process: subprocess.Popen[str] | None = None
        self._episode_active = False
        self._done = False

    def reset(self, seed: int | None = None) -> Observation:
        self._ensure_started()
        payload = self._request(f"RESET {0 if seed is None else seed}")
        frame = self._decode_frame(payload)
        self._frames.clear()
        self._frames.extend(frame.copy() for _ in range(self.frame_stack))
        self._episode_active = True
        self._done = False
        return self._stacked_observation()

    def step(self, action: ControlAction | int) -> StepResult:
        action_id = int(action)
        if action_id not in {member.value for member in ControlAction}:
            raise ValueError(f"Unknown control action: {action_id}")
        if not self._episode_active:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is terminal; call reset() before step()")

        self._ensure_started()
        payload = self._request(f"STEP {action_id}")
        self._frames.append(self._decode_frame(payload))
        observation = self._stacked_observation()
        reward = float(payload["reward"])
        done = bool(payload["done"])
        self._done = done
        info = self._info(payload)
        return observation, reward, done, info

    def close(self) -> None:
        process = self._process
        self._process = None
        self._episode_active = False
        self._done = False
        if process is None or process.poll() is not None:
            return

        assert process.stdin is not None
        try:
            process.stdin.write("CLOSE\n")
            process.stdin.flush()
            process.wait(timeout=5)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self._process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(self._process.stdout, selectors.EVENT_READ)
        events = selector.select(timeout=self.startup_timeout)
        selector.close()
        if not events:
            self._process.terminate()
            self._process.wait(timeout=5)
            detail = self._read_error()
            self._process = None
            raise BridgeError(
                "Timed out starting the JavaFX bridge. Ensure a desktop session "
                f"and JavaFX runtime are available. {detail}".strip()
            )
        ready = self._process.stdout.readline().strip()
        if ready != "READY":
            detail = self._read_error()
            self.close()
            raise BridgeError(
                f"Java bridge did not start (received {ready!r}). {detail}".strip()
            )

    def _request(self, command: str) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise BridgeError("Java bridge is not running")

        try:
            process.stdin.write(command + "\n")
            process.stdin.flush()
        except BrokenPipeError as exception:
            raise BridgeError(
                f"Java bridge stopped. {self._read_error()}"
            ) from exception

        response = process.stdout.readline()
        if not response:
            raise BridgeError(f"Java bridge stopped. {self._read_error()}".strip())

        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exception:
            raise BridgeError(f"Invalid bridge response: {response!r}") from exception

        if "error" in payload:
            raise BridgeError(str(payload["error"]))
        return payload

    def _read_error(self) -> str:
        process = self._process
        if process is None or process.stderr is None or process.poll() is None:
            return ""
        return process.stderr.read().strip()

    def _decode_frame(self, payload: dict[str, Any]) -> Observation:
        observation = payload.get("observation")
        if not isinstance(observation, dict):
            raise BridgeError("Bridge response is missing a visual observation")
        if observation.get("encoding") != "rgb24-base64":
            raise BridgeError(
                f"Unsupported frame encoding: {observation.get('encoding')}"
            )

        try:
            width = int(observation["width"])
            height = int(observation["height"])
            channels = int(observation["channels"])
            raw = base64.b64decode(observation["data"], validate=True)
        except (KeyError, TypeError, ValueError) as exception:
            raise BridgeError("Malformed visual observation") from exception

        if width < 1 or height < 1 or channels != 3:
            raise BridgeError(f"Invalid visual dimensions: {width}x{height}x{channels}")
        expected_bytes = width * height * channels
        if len(raw) != expected_bytes:
            raise BridgeError(
                f"Expected {expected_bytes} RGB bytes, received {len(raw)}"
            )

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels)
        output_height, output_width = self.observation_size
        row_indices = np.linspace(0, height - 1, output_height).astype(np.intp)
        column_indices = np.linspace(0, width - 1, output_width).astype(np.intp)
        resized = frame[row_indices[:, None], column_indices[None, :]]
        return np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0

    def _stacked_observation(self) -> Observation:
        if len(self._frames) != self.frame_stack:
            raise BridgeError("Frame stack is not initialized; call reset() first")
        return np.concatenate(tuple(self._frames), axis=0)

    @staticmethod
    def _info(payload: dict[str, Any]) -> Info:
        info = payload.get("info")
        if not isinstance(info, dict):
            raise BridgeError("Bridge response is missing an info object")
        return info
