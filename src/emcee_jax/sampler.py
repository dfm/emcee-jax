from functools import partial, wraps
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
from jax import random

from emcee_jax.moves import stretch
from emcee_jax.ravel_util import ravel_ensemble
from emcee_jax.types import Array, LogProbFn, MoveFn, Stats, Walker


def build_sampler(
    log_prob_fn: LogProbFn,
    *,
    move: MoveFn = stretch,
    log_prob_args: Tuple[Any, ...] = (),
    log_prob_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable[..., Tuple[Walker, Stats]]:
    log_prob_kwargs = {} if log_prob_kwargs is None else log_prob_kwargs

    @partial(jax.jit, static_argnames=["steps"])
    def sample(
        random_key: random.KeyArray,
        ensemble: Union[Walker, Array],
        steps: int = 1000,
    ) -> Tuple[Walker, Stats]:
        if not isinstance(ensemble, Walker):
            ensemble = Walker(
                coords=ensemble,
                log_probability=jax.vmap(log_prob_fn)(ensemble),
            )

        @jax.vmap
        @wraps(log_prob_fn)
        def wrapped_log_prob_fn(x: Array) -> Array:
            assert log_prob_kwargs is not None
            x = unravel(x)
            return log_prob_fn(x, *log_prob_args, **log_prob_kwargs)

        log_prob = ensemble.log_probability
        ensemble, unravel = ravel_ensemble(ensemble.coords)
        init, step = move(wrapped_log_prob_fn)
        state = init(ensemble)

        def wrapped_step(
            carry: Tuple[Array, Array], key: random.KeyArray
        ) -> Tuple[Tuple[Array, Array], Tuple[Stats, Array, Array]]:
            ensemble, log_prob = carry
            stats, ensemble, log_prob = step(state, key, ensemble, log_prob)
            return (ensemble, log_prob), (stats, ensemble, log_prob)

        carry = (ensemble, log_prob)
        _, (stats, ensemble, log_prob) = jax.lax.scan(
            wrapped_step, carry, random.split(random_key, steps)
        )
        return (
            Walker(
                coords=jax.vmap(jax.vmap(unravel))(ensemble),
                log_probability=log_prob,
            ),
            stats,
        )

    return sample
