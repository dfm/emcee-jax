from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random

from emcee_jax.moves.core import MoveState, RedBlue, StepState
from emcee_jax.moves.util import apply_accept
from emcee_jax.types import (
    Array,
    FlatWalkerState,
    PyTree,
    SampleStats,
    WrappedLogProbFn,
)


class Slice(RedBlue):
    max_doubles: int = 10_000
    max_shrinks: int = 100
    initial_step_size: float = 1.0
    tune_max_doubles: Optional[int] = None
    tune_max_shrinks: Optional[int] = None

    def init(
        self,
        log_prob_fn: WrappedLogProbFn,
        random_key: random.KeyArray,
        ensemble: FlatWalkerState,
    ) -> StepState:
        return StepState(
            move_state={"step_size": self.initial_step_size},
            walker_state=ensemble,
            extras=None,
        )

    def get_directions(
        self, size: int, key: random.KeyArray, complementary: Array
    ) -> Array:
        del size, key, complementary
        raise NotImplementedError

    def propose(
        self,
        log_prob_fn: WrappedLogProbFn,
        state: MoveState,
        key: random.KeyArray,
        target_walkers: FlatWalkerState,
        target_extras: PyTree,
        compl_walkers: FlatWalkerState,
        compl_extras: PyTree,
        *,
        tune: bool,
    ) -> Tuple[MoveState, FlatWalkerState, PyTree, SampleStats]:
        del compl_extras
        assert state is not None
        num_target = target_walkers.coordinates.shape[0]
        key0, *keys = random.split(key, num_target + 1)
        directions = state["step_size"] * self.get_directions(
            num_target, key0, compl_walkers.coordinates
        )

        if tune and self.tune_max_doubles is not None:
            max_doubles = self.tune_max_doubles
        else:
            max_doubles = self.max_doubles

        if tune and self.tune_max_shrinks is not None:
            max_shrinks = self.tune_max_shrinks
        else:
            max_shrinks = self.max_shrinks

        sample_func = partial(
            slice_sample, max_doubles, max_shrinks, log_prob_fn
        )
        updated, stats = jax.vmap(sample_func)(
            jnp.asarray(keys), target_walkers, directions
        )
        accept = jnp.logical_and(stats["bounds_ok"], stats["sample_ok"])
        stats["accept"] = accept
        stats["accept_prob"] = jnp.ones_like(updated.log_probability)
        stats["step_size"] = jnp.full_like(
            updated.log_probability, state["step_size"]
        )
        updated = apply_accept(accept, target_walkers, updated)

        if tune:
            num_doubles = 0.5 * jnp.mean(
                stats["num_doubles_left"] + stats["num_doubles_right"]
            )
            num_shrinks = jnp.mean(stats["num_shrinks"])
            factor = 2 * num_doubles / (num_doubles + num_shrinks)
            next_state = dict(state, step_size=state["step_size"] * factor)
        else:
            next_state = state

        return next_state, updated, target_extras, stats


class DiffEvolSlice(Slice):
    def get_directions(
        self, size: int, key: random.KeyArray, c: Array
    ) -> Array:
        # See the ``DiffEvol`` move for an explanation of the following
        choose2 = partial(random.choice, a=c, replace=False, shape=(2,))
        helpers = jax.vmap(choose2)(random.split(key, size))
        return jnp.squeeze(jnp.diff(helpers, axis=1))


def slice_sample(
    max_doubles: int,
    max_shrinks: int,
    log_prob_fn: WrappedLogProbFn,
    random_key: random.KeyArray,
    initial: FlatWalkerState,
    dx: Array,
) -> Tuple[FlatWalkerState, Dict[str, Any]]:
    level_key, doubling_key, shrink_key = random.split(random_key, 3)
    level = initial.log_probability - random.exponential(level_key)

    (
        left,
        right,
        num_doubles_left,
        num_doubles_right,
        bounds_ok,
    ) = _find_bounds_by_doubling_while_loop(
        max_doubles, log_prob_fn, level, doubling_key, initial.coordinates, dx
    )

    final, num_shrinks, sample_ok = _sample_by_shrinking_while_loop(
        max_shrinks, log_prob_fn, level, shrink_key, initial, left, right
    )

    return final, {
        "level": level,
        "num_doubles_left": num_doubles_left,
        "num_doubles_right": num_doubles_right,
        "bounds_ok": bounds_ok,
        "num_shrinks": num_shrinks,
        "sample_ok": sample_ok,
    }


def _find_bounds_by_doubling_while_loop(
    max_doubles: int,
    log_prob_fn: WrappedLogProbFn,
    level: Array,
    key: random.KeyArray,
    x0: Array,
    dx: Array,
) -> Tuple[Array, Array, Array, Array, Array]:
    def doubling(
        direction: float, args: Tuple[Array, Array, Array]
    ) -> Tuple[Array, Array, Array]:
        count, found, loc = args
        next_loc = loc + direction * dx
        log_prob, _ = log_prob_fn(next_loc)
        return count + 1, found | jnp.less(log_prob, level), next_loc

    cond = lambda args: jnp.logical_and(
        jnp.less(args[0], max_doubles), jnp.any(jnp.logical_not(args[1]))
    )
    r = random.uniform(key)
    num_left, left_ok, left = jax.lax.while_loop(
        cond, partial(doubling, -1), (0, False, x0 - r * dx)
    )
    num_right, right_ok, right = jax.lax.while_loop(
        cond, partial(doubling, 1), (0, False, x0 + (1 - r) * dx)
    )

    return left, right, num_left, num_right, jnp.logical_and(left_ok, right_ok)


def _sample_by_shrinking_while_loop(
    max_shrinks: int,
    log_prob_fn: WrappedLogProbFn,
    level: Array,
    key: random.KeyArray,
    initial: FlatWalkerState,
    left: Array,
    right: Array,
) -> Tuple[FlatWalkerState, Array, Array]:
    def shrinking(
        args: Tuple[
            Array, Array, Array, Array, random.KeyArray, FlatWalkerState
        ]
    ) -> Tuple[Array, Array, Array, Array, random.KeyArray, FlatWalkerState]:
        count, found, left, right, key, state = args
        key, next_key = random.split(key)
        u = random.uniform(key)
        x = (1 - u) * left + u * right
        log_prob, deterministics = log_prob_fn(x)
        next_state = state._replace(
            coordinates=x,
            deterministics=deterministics,
            log_probability=log_prob,
        )
        next_left = jnp.where(jnp.less(x, x0), x, left)
        next_right = jnp.where(jnp.greater_equal(x, x0), x, right)
        accept = jnp.greater_equal(log_prob, level)
        return (
            count + 1,
            found | accept,
            next_left,
            next_right,
            next_key,
            next_state,
        )

    x0 = initial.coordinates
    cond = lambda args: jnp.logical_and(
        jnp.less(args[0], max_shrinks), jnp.any(jnp.logical_not(args[1]))
    )
    count, ok, *_, state = jax.lax.while_loop(
        cond,
        shrinking,
        (0, False, left, right, key, initial),
    )
    return state, count, ok
