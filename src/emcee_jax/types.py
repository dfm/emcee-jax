from typing import Any, Callable, Dict, Iterable, NamedTuple, Optional, Union

import jax.numpy as jnp
import numpy as np

Array = Any
PyTree = Union[Array, Iterable[Array], Dict[Any, Array]]
SampleStats = Dict[str, Array]


class WalkerState(NamedTuple):
    coordinates: PyTree
    deterministics: PyTree
    log_probability: Array
    augments: Optional[PyTree] = None


class FlatWalkerState(NamedTuple):
    coordinates: Array
    deterministics: Array
    log_probability: Array
    augments: Optional[PyTree] = None


LogProbFn = Callable[..., Array]
WrappedLogProbFn = Callable[[Array], Array]
