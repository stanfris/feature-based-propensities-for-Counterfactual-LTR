from __future__ import annotations

from typing import Any

import numpy as np


def stack_dict_collate(batch: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """
    Canonical collator for fixed-shape dataset examples.

    Prebuilt and simulated click datasets already expose examples with aligned
    shapes, so batching only needs to stack values per key.
    """
    keys = batch[0].keys()
    return {key: np.stack([sample[key] for sample in batch]) for key in keys}


__all__ = ["stack_dict_collate"]
