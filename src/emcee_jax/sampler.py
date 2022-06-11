__all__ = ["build_sampler"]

from functools import partial
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import jax
import jax.linear_util as lu
from jax import random
from jax.flatten_util import ravel_pytree

from emcee_jax.moves import Move, StepState, Stretch
from emcee_jax.ravel_util import ravel_ensemble
from emcee_jax.trace import Trace
from emcee_jax.types import (
    Array,
    FlatWalkerState,
    LogProbFn,
    PyTree,
    SampleStats,
    WalkerState,
)


def build_sampler(
    log_prob_fn: LogProbFn,
    *,
    move: Optional[Move] = None,
    log_prob_args: Tuple[Any, ...] = (),
    log_prob_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable[..., Trace]:
    move = Stretch() if move is None else move
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
        is_state = isinstance(ensemble, WalkerState)
        is_state = is_state or isinstance(ensemble, FlatWalkerState)
        if not is_state:
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

        if log_probability is None:
            raise ValueError(
                "The initial log_probability is None, but it must be a scalar "
                "for each walker"
            )
        if log_probability.shape != coordinates.shape[:1]:
            raise ValueError(
                "Invalid shape for initial log_probability: expected "
                f"{coordinates.shape[:1]} but found {log_probability.shape}"
            )

        # TODO: This call to vmap could probably be replaced directly with the
        # batching primitives to take advantage of chaining of linear_util
        # wrapped functions for better performance.
        flat_log_prob_fn = flatten_log_prob_fn(
            wrapped_log_prob_fn, unravel_coordinates
        ).call_wrapped

        # Set up a flattened version of the ensemble state
        initial_ensemble = FlatWalkerState(
            coordinates=coordinates,
            deterministics=deterministics,
            log_probability=log_probability,
        )

        # Initialize the move function
        assert move is not None
        initial_carry = move.init(flat_log_prob_fn, initial_ensemble)
        step = partial(move.step, flat_log_prob_fn)

        def wrapped_step(
            carry: StepState, key: random.KeyArray
        ) -> Tuple[StepState, Tuple[StepState, SampleStats]]:
            carry, stats = step(key, carry)
            return carry, (carry, stats)

        # Run the sampler
        (_, final_ensemble), ((_, samples), sample_stats) = jax.lax.scan(
            wrapped_step, initial_carry, random.split(random_key, steps)
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
            final_state=final_ensemble,
            samples=samples,
            sample_stats=sample_stats,
        )

    return sample


@lu.transformation
def handle_deterministics(
    *args: Any, **kwargs: Any
) -> Generator[Tuple[Any, Any], Union[Any, Tuple[Any, Any]], None]:
    result = yield args, kwargs

    if isinstance(result, tuple):
        log_prob, *deterministics = result
        if len(deterministics) == 1:
            deterministics = deterministics[0]
    else:
        log_prob = result
        deterministics = None

    yield log_prob, deterministics


@lu.transformation
def flatten_log_prob_fn(
    unravel: Callable[[Array], Array], x: Array
) -> Generator[
    Tuple[Array, Union[PyTree, Dict[str, Any]]],
    Tuple[Array, Union[PyTree, Dict[str, Any]]],
    None,
]:
    x_flat = unravel(x)
    log_probability, deterministics = yield (x_flat,), {}
    deterministics, _ = ravel_pytree(deterministics)
    yield log_probability, deterministics
