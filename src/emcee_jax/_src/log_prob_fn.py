from typing import Any, Callable, Generator, Tuple, Union

import jax
import jax.linear_util as lu
import jax.numpy as jnp

from emcee_jax._src.types import Array, PyTree

LogProbFn = Callable[..., Union[Array, Tuple[Array, PyTree]]]


def wrap_log_prob_fn(
    log_prob_fn: LogProbFn, *log_prob_args: Any, **log_prob_kwargs: Any
) -> lu.WrappedFun:
    wrapped_log_prob_fn = lu.wrap_init(log_prob_fn)
    return handle_deterministics_and_nans(
        wrapped_log_prob_fn, *log_prob_args, **log_prob_kwargs
    )


@lu.transformation
def handle_deterministics_and_nans(
    *args: Any, **kwargs: Any
) -> Generator[Tuple[Any, Any], Union[Any, Tuple[Any, Any]], None]:
    result = yield args, kwargs

    # Unwrap deterministics if they are provided or default to None
    if isinstance(result, tuple):
        log_prob, *deterministics = result
        if len(deterministics) == 1:
            deterministics = deterministics[0]
    else:
        log_prob = result
        deterministics = None

    if log_prob is None:
        raise ValueError(
            "A log probability function must return a scalar value, got None"
        )
    if log_prob.shape != ():
        raise ValueError(
            "A log probability function must return a scalar; "
            f"computed shape is '{log_prob.shape}', expected '()'"
        )

    # Handle the case where the computed log probability is NaN by replacing it
    # with negative infinity so that it gets rejected
    log_prob = jax.lax.cond(
        jnp.isnan(log_prob), lambda: -jnp.inf, lambda: log_prob
    )

    yield log_prob, deterministics
