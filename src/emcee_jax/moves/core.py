__all__ = ["Stretch", "DiffEvol"]

from collections import OrderedDict
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import tree_map
from jax_dataclasses import pytree_dataclass

from emcee_jax.moves.util import apply_accept
from emcee_jax.types import (
    Array,
    FlatWalkerState,
    PyTree,
    SampleStats,
    WrappedLogProbFn,
)

MoveState = Optional[Any]


class StepState(NamedTuple):
    move_state: MoveState
    walker_state: FlatWalkerState
    extras: PyTree


@pytree_dataclass
class Move:
    if TYPE_CHECKING:

        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        pytree_dataclass(cls)

    def init(
        self,
        log_prob_fn: WrappedLogProbFn,
        random_key: random.KeyArray,
        ensemble: FlatWalkerState,
    ) -> StepState:
        del log_prob_fn, random_key
        return StepState(move_state=None, walker_state=ensemble, extras=None)

    def step(
        self,
        log_prob_fn: WrappedLogProbFn,
        random_key: random.KeyArray,
        state: StepState,
        *,
        tune: bool = False,
    ) -> Tuple[StepState, SampleStats]:
        del log_prob_fn, random_key, state, tune
        raise NotImplementedError


class Composed(Move):
    moves: Sequence[Tuple[str, Move]]

    def init(
        self,
        log_prob_fn: WrappedLogProbFn,
        random_key: random.KeyArray,
        ensemble: FlatWalkerState,
    ) -> StepState:
        keys = random.split(random_key, len(self.moves))
        move_state = OrderedDict()
        extras = OrderedDict()
        for key, (name, move) in zip(keys, self.moves):
            state = move.init(log_prob_fn, key, ensemble)
            move_state[name] = state.move_state
            extras[name] = state.extras
        return StepState(
            move_state=move_state,
            walker_state=ensemble,
            extras=extras,
        )

    def step(
        self,
        log_prob_fn: WrappedLogProbFn,
        random_key: random.KeyArray,
        state: StepState,
        *,
        tune: bool = False,
    ) -> Tuple[StepState, SampleStats]:
        keys = random.split(random_key, len(self.moves))

        move_state = state.move_state
        walker_state = state.walker_state
        extras = state.extras
        if TYPE_CHECKING:
            assert isinstance(move_state, OrderedDict)
            assert isinstance(extras, OrderedDict)

        new_move_state = OrderedDict()
        new_extras = OrderedDict()
        stats = OrderedDict()

        for key, (name, move) in zip(keys, self.moves):
            new_state, stats_ = move.step(
                log_prob_fn,
                key,
                StepState(
                    move_state=move_state[name],
                    walker_state=walker_state,
                    extras=extras[name],
                ),
                tune=tune,
            )
            new_move_state[name] = new_state.move_state
            new_extras[name] = new_state.extras
            stats[name] = stats_
            walker_state = new_state.walker_state

        return (
            StepState(
                move_state=new_move_state,
                walker_state=walker_state,
                extras=new_extras,
            ),
            stats,
        )


def compose(*moves: Move, **named_moves: Move) -> Composed:
    transformed = []
    for ind, move in enumerate(moves):
        transformed.append((f"{move.__class__.__name__}_{ind}", move))
    for name, move in named_moves.items():
        transformed.append((name, move))
    return Composed(moves=transformed)


class RedBlue(Move):
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
        del log_prob_fn, state, key, tune
        del target_walkers, target_extras
        del compl_walkers, compl_extras
        raise NotImplementedError

    def step(
        self,
        log_prob_fn: WrappedLogProbFn,
        random_key: random.KeyArray,
        state: StepState,
        *,
        tune: bool = False,
    ) -> Tuple[StepState, SampleStats]:
        move_state, ensemble, extras = state
        key1, key2 = random.split(random_key)
        nwalkers, _ = ensemble.coordinates.shape
        mid = nwalkers // 2

        ens1 = tree_map(lambda x: x[:mid], ensemble)
        ext1 = tree_map(lambda x: x[:mid], extras)
        ens2 = tree_map(lambda x: x[mid:], ensemble)
        ext2 = tree_map(lambda x: x[mid:], extras)

        half_step = partial(self.propose, log_prob_fn, tune=tune)
        move_state, ens1, ext1, stats1 = half_step(
            move_state, key1, ens1, ext1, ens2, ext2
        )
        move_state, ens2, ext2, stats2 = half_step(
            move_state, key2, ens2, ext2, ens1, ext1
        )
        stats = tree_map(lambda *x: jnp.concatenate(x, axis=0), stats1, stats2)
        ensemble = tree_map(lambda *x: jnp.concatenate(x, axis=0), ens1, ens2)
        extras = tree_map(lambda *x: jnp.concatenate(x, axis=0), ext1, ext2)
        return (
            StepState(
                move_state=move_state, walker_state=ensemble, extras=extras
            ),
            stats,
        )


class SimpleRedBlue(RedBlue):
    def propose_simple(
        self, key: random.KeyArray, s: Array, c: Array
    ) -> Tuple[Array, Array]:
        del key, s, c
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
        del compl_extras, tune
        key1, key2 = random.split(key)
        q, f = self.propose_simple(
            key1, target_walkers.coordinates, compl_walkers.coordinates
        )
        nlp, ndet = jax.vmap(log_prob_fn)(q)
        updated = target_walkers._replace(
            coordinates=q,
            deterministics=ndet,
            log_probability=nlp,
        )

        diff = nlp - target_walkers.log_probability + f
        accept_prob = jnp.exp(diff)
        accept = accept_prob > random.uniform(key2, shape=diff.shape)
        updated = apply_accept(accept, target_walkers, updated)
        return (
            state,
            updated,
            target_extras,
            {"accept": accept, "accept_prob": accept_prob},
        )


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


class DiffEvol(SimpleRedBlue):
    gamma: Optional[Array] = None
    sigma: Array = 1.0e-5

    def propose_simple(
        self, key: random.KeyArray, s: Array, c: Array
    ) -> Tuple[Array, Array]:
        ns, ndim = s.shape
        key1, key2 = random.split(key)

        # This is a magic formula from the paper
        gamma0 = 2.38 / np.sqrt(2 * ndim) if self.gamma is None else self.gamma

        # These two slightly complicated lines are just to select two helper
        # walkers per target walker _without replacement_. This means that we'll
        # always get two different complementary walkers per target walker.
        choose2 = partial(random.choice, a=c, replace=False, shape=(2,))
        helpers = jax.vmap(choose2)(random.split(key1, ns))

        # Use the vector between helper walkers to update the target walker
        delta = jnp.squeeze(jnp.diff(helpers, axis=1))
        norm = random.normal(key2, shape=(ns,))
        delta = (gamma0 + self.sigma * norm[:, None]) * delta

        return s + delta, jnp.zeros(ns)
