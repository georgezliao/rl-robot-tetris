"""Stable identifiers for numerical observations."""

import hashlib

import numpy as np
import numpy.typing as npt


def observation_sha256(observation: npt.NDArray[np.generic]) -> str:
    """Hash an array's dtype, shape, and C-order bytes with SHA-256."""
    contiguous = np.ascontiguousarray(observation)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(size) for size in contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()
