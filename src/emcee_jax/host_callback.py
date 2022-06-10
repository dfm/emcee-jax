__all__ = ["wrap_python_log_prob_fn"]

from functools import wraps
from typing import Callable, List, Tuple

import jax
import numpy as np
from jax.custom_batching import custom_vmap
from jax.experimental import host_callback

from emcee_jax.types import Array, LogProbFn


def wrap_python_log_prob_fn(
    python_log_prob_fn: Callable[..., Array]
) -> LogProbFn:
    @wraps(python_log_prob_fn)
    @custom_vmap
    def log_prob_fn(params: Array) -> Array:
        return host_callback.call(
            python_log_prob_fn,
            params,
            result_shape=jax.ShapeDtypeStruct((), params.dtype),
        )

    @log_prob_fn.def_vmap
    def vmap_rule(
        axis_size: int, in_batched: List[bool], params: Array
    ) -> Tuple[Array, bool]:
        assert in_batched[0]
        assert axis_size == params.shape[0]
        return (
            host_callback.call(
                lambda x: np.stack([python_log_prob_fn(y) for y in x]),
                params,
                result_shape=jax.ShapeDtypeStruct((axis_size,), params.dtype),
            ),
            True,
        )

    return log_prob_fn
