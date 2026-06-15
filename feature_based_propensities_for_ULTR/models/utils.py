from jax import Array
import jax.numpy as jnp
from flax import nnx
from typing import Callable, List
from flax import nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp
from pathlib import Path

def get_sequential(
    features: int,
    hidden_units: int,
    layers: int,
    dropout: float,
    *,
    rngs: nnx.Rngs,
) -> List[Callable]:
    modules = []

    for _ in range(layers):
        modules.extend(
            [
                nnx.Linear(in_features=features, out_features=hidden_units, rngs=rngs),
                nnx.relu,
                nnx.Dropout(rate=dropout, rngs=rngs),
            ]
        )
        features = hidden_units

    return modules


def get_sigmoid_mlp(
    input_dim: int,
    hidden_units: List[int],
    final_activation: bool,
    *,
    rngs: nnx.Rngs,
) -> List[Callable]:
    modules = []
    features = input_dim

    # hidden layers
    for h in hidden_units:
        modules.extend(
            [
                nnx.Linear(
                    in_features=features,
                    out_features=h,
                    dtype=jnp.float64,
                    rngs=rngs,
                ),
                nnx.sigmoid,
            ]
        )
        features = h

    # output layer
    modules.append(
        nnx.Linear(
            in_features=features,
            out_features=1,
            dtype=jnp.float64,
            rngs=rngs,
        )
    )

    if final_activation:
        modules.append(nnx.sigmoid)

    return modules


def load_model_params(model, ckpt_dir="checkpoint", rng_seed=0):
    """
    Load model parameters into an existing model, keeping RNG separate.
    """
    ckptr = ocp.StandardCheckpointer()
    
    # Split the new model into RNG and other state
    _, _, other_state = nnx.split(model, nnx.RngState, ...)
    
    # Restore saved parameters (other_state)
    ckpt_path = Path(ckpt_dir).resolve()
    restored_other_state = ckptr.restore(ckpt_path, other_state)

    # Merge restored state into the live model
    nnx.update(model, restored_other_state)

    print("Parameters successfully restored!")
    return model
