__all__ = ["build_sampler"]

from functools import partial
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import jax
import jax.linear_util as lu
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree

from emcee_jax.moves.core import Move, StepState, Stretch
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
    wrapped_log_prob_fn = handle_deterministics_and_nans(wrapped_log_prob_fn)

    @partial(jax.jit, static_argnames=["steps"])
    def sample(
        random_key: random.KeyArray,
        ensemble: Union[WalkerState, Array],
        steps: int = 1000,
        tune: Optional[int] = None,
    ) -> Trace:
        init_key, tune_key, sample_key = jax.random.split(random_key, 3)

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
        initial_carry = move.init(flat_log_prob_fn, init_key, initial_ensemble)

        if tune is not None:

            def tune_step(
                carry: StepState, key: random.KeyArray
            ) -> Tuple[StepState, Tuple[StepState, SampleStats]]:
                assert move is not None
                carry, stats = move.step(
                    flat_log_prob_fn, key, carry, tune=True
                )
                return carry, (carry, stats)

            # Run the sampler
            initial_carry, _ = jax.lax.scan(
                tune_step, initial_carry, random.split(tune_key, steps)
            )

        def wrapped_step(
            carry: StepState, key: random.KeyArray
        ) -> Tuple[StepState, Tuple[StepState, SampleStats]]:
            assert move is not None
            carry, stats = move.step(flat_log_prob_fn, key, carry)
            return carry, (carry, stats)

        # Run the sampler
        (_, flat_ensemble, final_extras), (
            (_, flat_samples, extras_trace),
            sample_stats,
        ) = jax.lax.scan(
            wrapped_step, initial_carry, random.split(sample_key, steps)
        )

        # Unravel the final state to have the correct structure
        unravel_coordinates = jax.vmap(unravel_coordinates)
        unravel_deterministics = jax.vmap(unravel_deterministics)
        coordinates = unravel_coordinates(flat_ensemble.coordinates)
        deterministics = unravel_deterministics(flat_ensemble.deterministics)
        final_ensemble = WalkerState(
            coordinates=coordinates,
            deterministics=deterministics,
            log_probability=flat_ensemble.log_probability,
        )

        # Unravel the chain to have the correct structure
        coordinates = jax.vmap(unravel_coordinates)(flat_samples.coordinates)
        deterministics = jax.vmap(unravel_deterministics)(
            flat_samples.deterministics
        )
        samples = WalkerState(
            coordinates=coordinates,
            deterministics=deterministics,
            log_probability=flat_samples.log_probability,
        )
        return Trace(
            final_state=final_ensemble,
            final_extras=final_extras,
            samples=samples,
            extras=extras_trace,
            sample_stats=sample_stats,
        )

    return sample


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
