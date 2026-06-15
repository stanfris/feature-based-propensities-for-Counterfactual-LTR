import numpy as np
import jax
import jax.numpy as jnp
from jax import Array


def pad(x: np.ndarray, max_n: int):
    """
    Pads items with zeros to reach max_n in first (batch) dimension.

    E.g.: x = np.array([5, 4, 3]), n = 5
    -> np.array([5, 4, 3, 0, 0])

    E.g.: x = np.array([[5, 4, 3], [1, 2, 3]]), n = 4
    -> np.array([[5, 4, 3], [1, 2, 3], [0, 0, 0], [0, 0, 0]])
    """
    # Ensure single array, not, e.g., list of arrays:
    x = np.stack(x)
    # Add padding:
    padding = max(max_n - x.shape[0], 0)
    pad_width = [(0, padding)]

    for i in range(x.ndim - 1):
        pad_width.append((0, 0))

    return np.pad(x, pad_width, mode="constant")


def log1p(x: Array) -> Array:
    # Use float64 preprocessing to avoid overflow from extreme sentinel values
    # when JAX's default dtype is float32.
    x_np = np.asarray(x, dtype=np.float64)
    return np.sign(x_np) * np.log1p(np.abs(x_np))
