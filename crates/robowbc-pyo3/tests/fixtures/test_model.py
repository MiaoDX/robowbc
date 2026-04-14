"""Fixture model for robowbc-pyo3 unit tests.

Defines a ``predict`` callable that accepts a 1-D numpy float32 array
(the flattened observation) and returns a 1-D numpy float32 array of
length 4 (matching the 4-joint test robot).

The observation layout is:
  [joint_positions (4), joint_velocities (4), gravity (3), angular_velocity (3), command...]

This fixture intentionally uses no PyTorch so the test suite can run in
any environment that has Python 3.10+ and NumPy.
"""
import numpy as np


def predict(obs_flat: np.ndarray) -> np.ndarray:
    """Return the first four elements of the observation as joint targets."""
    return obs_flat[:4].astype(np.float32)
