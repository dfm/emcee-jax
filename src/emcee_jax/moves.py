__all__ = ["Stretch", "DiffEvol"]

from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import tree_map

from emcee_jax.types import (
    Array,
    FlatWalkerState,
    SampleStats,
    WrappedLogProbFn,
)

MoveState = Optional[Any]


class StepState(NamedTuple):
    move_state: MoveState
    walker_state: FlatWalkerState


class Move:
    def init(
        self, log_prob_fn: WrappedLogProbFn, ensemble: FlatWalkerState
    ) -> StepState:
        return StepState(move_state=None, walker_state=ensemble)

    def step(
        self,
        log_prob_fn: WrappedLogProbFn,
        random_key: random.KeyArray,
        state: StepState,
    ) -> Tuple[StepState, SampleStats]:
        raise NotImplementedError


class RedBlue(Move):
    def propose(
        self,
        log_prob_fn: WrappedLogProbFn,
        state: MoveState,
        key: random.KeyArray,
        target: FlatWalkerState,
        complementary: FlatWalkerState,
    ) -> Tuple[MoveState, FlatWalkerState, Array, SampleStats]:
        raise NotImplementedError

    def step(
        self,
        log_prob_fn: WrappedLogProbFn,
        random_key: random.KeyArray,
        state: StepState,
    ) -> Tuple[StepState, SampleStats]:
        move_state, ensemble = state
        key1, key2 = random.split(random_key)
        nwalkers, _ = ensemble.coordinates.shape
        mid = nwalkers // 2

        half1 = tree_map(lambda x: x[:mid], ensemble)
        half2 = tree_map(lambda x: x[mid:], ensemble)

        def half_step(
            current_state: MoveState,
            key: random.KeyArray,
            target: FlatWalkerState,
            complementary: FlatWalkerState,
        ) -> Tuple[MoveState, Array, FlatWalkerState, SampleStats]:
            key1, key2 = random.split(key)
            new_state, new_target, factor, stats = self.propose(
                log_prob_fn, current_state, key1, target, complementary
            )
            diff = new_target.log_probability - target.log_probability
            diff += factor
            accept_prob = stats["accept_prob"] = jnp.exp(diff)
            accept = stats["accept"] = accept_prob > random.uniform(
                key2, shape=diff.shape
            )
            return (new_state, accept, new_target, stats)

        move_state, acc1, prop, stats1 = half_step(
            move_state, key1, half1, half2
        )
        half1 = apply_accept(acc1, half1, prop)
        move_state, acc2, prop, stats2 = half_step(
            move_state, key2, half2, half1
        )
        half2 = apply_accept(acc2, half2, prop)
        stats = tree_map(lambda *x: jnp.concatenate(x, axis=0), stats1, stats2)
        updated = tree_map(lambda *x: jnp.concatenate(x, axis=0), half1, half2)

        return StepState(move_state=move_state, walker_state=updated), stats


class SimpleRedBlue(RedBlue):
    def propose_simple(
        self, key: random.KeyArray, s: Array, c: Array
    ) -> Tuple[Array, Array]:
        raise NotImplementedError

    def propose(
        self,
        log_prob_fn: WrappedLogProbFn,
        state: MoveState,
        key: random.KeyArray,
        target: FlatWalkerState,
        complementary: FlatWalkerState,
    ) -> Tuple[MoveState, FlatWalkerState, Array, SampleStats]:
        q, f = self.propose_simple(
            key, target.coordinates, complementary.coordinates
        )
        nlp, ndet = jax.vmap(log_prob_fn)(q)
        updated = FlatWalkerState(
            coordinates=q,
            deterministics=ndet,
            log_probability=nlp,
            augments=target.augments,
        )
        return state, updated, f, {}


@dataclass(frozen=True)
class Stretch(SimpleRedBlue):
    a: Array = 2.0

    def propose_simple(
        self, key: random.KeyArray, s: Array, c: Array
    ) -> Tuple[Array, Array]:
        ns, ndim = s.shape
        key1, key2 = random.split(key)
        u = random.uniform(key1, shape=(ns,))
        z = jnp.square((self.a - 1) * u + 1) / self.a
        c = random.choice(key2, c, shape=(ns,))
        q = c - (c - s) * z[..., None]
        return q, (ndim - 1) * jnp.log(z)


@dataclass(frozen=True)
class DiffEvol(SimpleRedBlue):
    gamma: Optional[Array] = None
    sigma: Array = 1.0e-5

    def propose_simple(
        self, key: random.KeyArray, s: Array, c: Array
    ) -> Tuple[Array, Array]:
        ns, ndim = s.shape
        key1, key2 = jax.random.split(key)

        # This is a magic formula from the paper
        gamma0 = 2.38 / np.sqrt(2 * ndim) if self.gamma is None else self.gamma

        # These two slightly complicated lines are just to select two helper
        # walkers per target walker _without replacement_. This means that we'll
        # always get two different complementary walkers per target walker.
        choose2 = partial(jax.random.choice, a=c, replace=False, shape=(2,))
        helpers = jax.vmap(choose2)(jax.random.split(key1, ns))

        # Use the vector between helper walkers to update the target walker
        delta = jnp.squeeze(jnp.diff(helpers, axis=1))
        norm = jax.random.normal(key2, shape=(ns,))
        delta = (gamma0 + self.sigma * norm[:, None]) * delta

        return s + delta, jnp.zeros(ns)


def apply_accept(
    accept: Array, target: FlatWalkerState, other: FlatWalkerState
) -> FlatWalkerState:
    accepter = jax.vmap(lambda a, x, y: jnp.where(a, y, x))
    accepter = partial(accepter, accept)
    return tree_map(accepter, target, other)
