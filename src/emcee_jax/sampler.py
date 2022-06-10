from functools import partial, wraps
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import jax
from jax import random
import jax.linear_util as lu
from jax.flatten_util import ravel_pytree

from emcee_jax.moves import stretch
from emcee_jax.ravel_util import ravel_ensemble
from emcee_jax.types import (
    Array,
    Deterministics,
    LogProbFn,
    MoveFn,
    SamplerStats,
    Trace,
    WalkerState,
)


def build_sampler(
    log_prob_fn: LogProbFn,
    *,
    move: MoveFn = stretch,
    log_prob_args: Tuple[Any, ...] = (),
    log_prob_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable[..., Trace]:
    log_prob_kwargs = {} if log_prob_kwargs is None else log_prob_kwargs
    wrapped_log_prob_fn = lu.wrap_init(
        partial(log_prob_fn, *log_prob_args, **log_prob_kwargs)
    )
    wrapped_log_prob_fn = handle_deterministics(wrapped_log_prob_fn)

    @partial(jax.jit, static_argnames=["steps"])
    def sample(
        random_key: random.KeyArray,
        ensemble: Union[WalkerState, Array],
        steps: int = 1000,
    ) -> Trace:
        # Handle cases when either an ensemble or array is passed as input
        if not isinstance(ensemble, WalkerState):
            coordinates = ensemble
            log_probability, deterministics = jax.vmap(
                wrapped_log_prob_fn.call_wrapped
            )(coordinates)
        else:
            coordinates = ensemble.coordinates
            log_probability = ensemble.log_probability
            deterministics = ensemble.deterministics

        # Work out the ravel/unravel operations for this ensemble
        coordinates, unravel_coordinates = ravel_ensemble(coordinates)
        deterministics, unravel_deterministics = ravel_ensemble(deterministics)

        # Deal with the case where deterministics are not provided
        deterministics = deterministics.reshape((coordinates.shape[0], -1))

        # TODO: This call to vmap could probably be replaced directly with the
        # batching primitives to take advantage of chaining of linear_util
        # wrapped functions for better performance.
        flat_log_prob_fn = jax.vmap(
            flatten_log_prob_fn(
                wrapped_log_prob_fn, unravel_coordinates
            ).call_wrapped
        )

        # Set up a flattened version of the ensemble state
        initial_ensemble = WalkerState(
            coordinates=coordinates,
            deterministics=deterministics,
            log_probability=log_probability,
        )

        # Initialize the move function
        init, step = move(flat_log_prob_fn)
        state = init(coordinates)

        def wrapped_step(
            previous_ensemble: WalkerState, key: random.KeyArray
        ) -> Tuple[WalkerState, Tuple[SamplerStats, WalkerState]]:
            stats, next_ensemble = step(state, key, previous_ensemble)
            return next_ensemble, (stats, next_ensemble)

        # Run the sampler
        final_ensemble, (sampler_stats, samples) = jax.lax.scan(
            wrapped_step, initial_ensemble, random.split(random_key, steps)
        )

        # Unravel the final state to have the correct structure
        unravel_coordinates = jax.vmap(unravel_coordinates)
        unravel_deterministics = jax.vmap(unravel_deterministics)
        coordinates = unravel_coordinates(final_ensemble.coordinates)
        deterministics = unravel_deterministics(final_ensemble.deterministics)
        final_ensemble = WalkerState(
            coordinates=coordinates,
            deterministics=deterministics,
            log_probability=final_ensemble.log_probability,
        )

        # Unravel the chain to have the correct structure
        coordinates = jax.vmap(unravel_coordinates)(samples.coordinates)
        deterministics = jax.vmap(unravel_deterministics)(
            samples.deterministics
        )
        samples = WalkerState(
            coordinates=coordinates,
            deterministics=deterministics,
            log_probability=samples.log_probability,
        )
        return Trace(
            final_state=final_ensemble, samples=samples, stats=sampler_stats
        )

    return sample


@lu.transformation
def handle_deterministics(
    *args: Any, **kwargs: Any
) -> Generator[Tuple[Any, Any], Tuple[Any, Any], None]:
    result = yield args, kwargs

    try:
        log_prob, *deterministics = result
    except (ValueError, TypeError):
        log_prob = result
        deterministics = None

    if deterministics is not None and len(deterministics) == 1:
        deterministics = deterministics[0]

    yield log_prob, deterministics


@lu.transformation
def flatten_log_prob_fn(
    unravel: Callable[[Array], Array], x: Array
) -> Generator[
    Tuple[Array, Union[Deterministics, Dict[str, Any]]],
    Tuple[Array, Union[Deterministics, Dict[str, Any]]],
    None,
]:
    x_flat = unravel(x)
    log_probability, deterministics = yield (x_flat,), {}
    deterministics, _ = ravel_pytree(deterministics)
    yield log_probability, deterministics
