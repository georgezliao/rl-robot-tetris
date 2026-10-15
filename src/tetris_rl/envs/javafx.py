"""Python environment adapter for the original JavaFX Tetris game."""

from __future__ import annotations

import json
import selectors
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    ) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        default_command = [str(repo_root / "scripts" / "run_java_bridge.sh")]
        self.command = list(command or default_command)
        self.startup_timeout = startup_timeout
        self._process: subprocess.Popen[str] | None = None

    def reset(self, seed: int | None = None) -> Observation:
        self._ensure_started()
        payload = self._request(f"RESET {0 if seed is None else seed}")
        return self._observation(payload)

    def step(self, action: ControlAction | int) -> StepResult:
        action_id = int(action)
        if action_id not in {member.value for member in ControlAction}:
            raise ValueError(f"Unknown control action: {action_id}")

        self._ensure_started()
        payload = self._request(f"STEP {action_id}")
        observation = self._observation(payload)
        reward = float(payload["reward"])
        done = bool(payload["done"])
        info = self._info(payload)
        return observation, reward, done, info

    def close(self) -> None:
        process = self._process
        self._process = None
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

    def __enter__(self) -> JavaFxControlEnv:
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
            raise BridgeError(f"Java bridge stopped. {self._read_error()}") from exception

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

    @staticmethod
    def _observation(payload: dict[str, Any]) -> Observation:
        grid = np.asarray(payload["observation"], dtype=np.float32)
        if grid.shape != (20, 10):
            raise BridgeError(f"Expected a 20x10 board, received {grid.shape}")
        return grid[np.newaxis, :, :]

    @staticmethod
    def _info(payload: dict[str, Any]) -> Info:
        info = payload.get("info")
        if not isinstance(info, dict):
            raise BridgeError("Bridge response is missing an info object")
        return info
