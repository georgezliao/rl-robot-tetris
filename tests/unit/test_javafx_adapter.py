import base64

import numpy as np
import pytest

from tetris_rl.envs.javafx import BridgeError, JavaFxControlEnv
from tetris_rl.types import ControlAction


def visual_payload(frame: np.ndarray) -> dict[str, object]:
    height, width, channels = frame.shape
    return {
        "observation": {
            "encoding": "rgb24-base64",
            "width": width,
            "height": height,
            "channels": channels,
            "data": base64.b64encode(frame.tobytes()).decode("ascii"),
        }
    }


def test_observation_decodes_resizes_and_normalizes_rgb_pixels() -> None:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    env = JavaFxControlEnv(observation_size=(2, 3), frame_stack=1)

    observation = env._decode_frame(visual_payload(frame))

    assert observation.shape == (3, 2, 3)
    assert observation.dtype == np.float32
    np.testing.assert_array_equal(observation[0], np.ones((2, 3)))
    np.testing.assert_array_equal(observation[1:], np.zeros((2, 2, 3)))


def test_observation_rejects_wrong_byte_count() -> None:
    env = JavaFxControlEnv(frame_stack=1)
    payload = visual_payload(np.zeros((2, 2, 3), dtype=np.uint8))
    payload["observation"]["data"] = base64.b64encode(b"too short").decode()

    with pytest.raises(BridgeError):
        env._decode_frame(payload)


def test_step_requires_reset_without_starting_bridge() -> None:
    env = JavaFxControlEnv()

    with pytest.raises(RuntimeError, match="reset"):
        env.step(ControlAction.NOOP)
