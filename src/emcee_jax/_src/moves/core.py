from collections import OrderedDict
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple

import jax
import jax.linear_util as lu
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import tree_map
from jax_dataclasses import pytree_dataclass

from emcee_jax._src.ensemble import Ensemble, get_ensemble_shape
from emcee_jax._src.moves.util import apply_accept
from emcee_jax._src.types import Array, Extras, PyTree, SampleStats

MoveState = Optional[Any]


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
        random_key: random.KeyArray,
        ensemble: Ensemble,
    ) -> Tuple[MoveState, Extras]:
        del random_key, ensemble
        return None, None

    def step(
        self,
        log_prob_fn: lu.WrappedFun,
        random_key: random.KeyArray,
        state: MoveState,
        ensemble: Ensemble,
        extras: Extras,
        *,
        tune: bool = False,
    ) -> Tuple[Tuple[MoveState, Ensemble, Extras], SampleStats]:
        del log_prob_fn, random_key, state, ensemble, extras, tune
        raise NotImplementedError


class Composed(Move):
    moves: Sequence[Tuple[str, Move]]

    def init(
        self,
        random_key: random.KeyArray,
        ensemble: Ensemble,
    ) -> Tuple[MoveState, Extras]:
        keys = random.split(random_key, len(self.moves))
        state = OrderedDict()
        extras = OrderedDict()
        for key, (name, move) in zip(keys, self.moves):
            state[name], extras[name] = move.init(key, ensemble)
        return state, extras

    def step(
        self,
        log_prob_fn: lu.WrappedFun,
        random_key: random.KeyArray,
        state: MoveState,
        ensemble: Ensemble,
        extras: Extras,
        *,
        tune: bool = False,
    ) -> Tuple[Tuple[MoveState, Ensemble, Extras], SampleStats]:
        if TYPE_CHECKING:
            assert isinstance(state, OrderedDict)
            assert isinstance(extras, OrderedDict)
        keys = random.split(random_key, len(self.moves))
        new_state = OrderedDict()
        new_extras = OrderedDict()
        stats = OrderedDict()
        for key, (name, move) in zip(keys, self.moves):
            new, stats[name] = move.step(
                log_prob_fn,
                key,
                state[name],
                ensemble,
                extras[name],
                tune=tune,
            )
            new_state[name], ensemble, new_extras[name] = new
        return (new_state, ensemble, new_extras), stats


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
        log_prob_fn: lu.WrappedFun,
        state: MoveState,
        key: random.KeyArray,
        target_walkers: Ensemble,
        target_extras: Extras,
        compl_walkers: Ensemble,
        compl_extras: Extras,
        *,
        tune: bool,
    ) -> Tuple[MoveState, Ensemble, PyTree, SampleStats]:
        del log_prob_fn, state, key, tune
        del target_walkers, target_extras
        del compl_walkers, compl_extras
        raise NotImplementedError

    def step(
        self,
        log_prob_fn: lu.WrappedFun,
        random_key: random.KeyArray,
        state: MoveState,
        ensemble: Ensemble,
        extras: Extras,
        *,
        tune: bool = False,
    ) -> Tuple[Tuple[MoveState, Ensemble, Extras], SampleStats]:
        # move_state, ensemble, extras = state
        key1, key2 = random.split(random_key)
        nwalkers, _ = get_ensemble_shape(ensemble)
        mid = nwalkers // 2

        ens1 = tree_map(lambda x: x[:mid], ensemble)
        ext1 = tree_map(lambda x: x[:mid], extras)
        ens2 = tree_map(lambda x: x[mid:], ensemble)
        ext2 = tree_map(lambda x: x[mid:], extras)

        half_step = partial(self.propose, log_prob_fn, tune=tune)
        state, ens1, ext1, stats1 = half_step(
            state, key1, ens1, ext1, ens2, ext2
        )
        state, ens2, ext2, stats2 = half_step(
            state, key2, ens2, ext2, ens1, ext1
        )
        stats = tree_map(lambda *x: jnp.concatenate(x, axis=0), stats1, stats2)
        ensemble = tree_map(lambda *x: jnp.concatenate(x, axis=0), ens1, ens2)
        extras = tree_map(lambda *x: jnp.concatenate(x, axis=0), ext1, ext2)
        return (state, ensemble, extras), stats


class SimpleRedBlue(RedBlue):
    def propose_simple(
        self, key: random.KeyArray, s: PyTree, c: PyTree
    ) -> Tuple[PyTree, Array]:
        del key, s, c
        raise NotImplementedError

    def propose(
        self,
        log_prob_fn: lu.WrappedFun,
        state: MoveState,
        key: random.KeyArray,
        target_walkers: Ensemble,
        target_extras: Extras,
        compl_walkers: Ensemble,
        compl_extras: Extras,
        *,
        tune: bool,
    ) -> Tuple[MoveState, Ensemble, PyTree, SampleStats]:
        del compl_extras, tune
        key1, key2 = random.split(key)
        q, f = self.propose_simple(
            key1, target_walkers.coordinates, compl_walkers.coordinates
        )
        nlp, ndet = jax.vmap(log_prob_fn.call_wrapped)(q)
        updated = target_walkers._replace(
            coordinates=q,
            deterministics=ndet,
            log_probability=nlp,
        )
        diff = nlp - target_walkers.log_probability + f
        accept_prob = jnp.minimum(jnp.exp(diff), 1)
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
        self, key: random.KeyArray, s: PyTree, c: PyTree
    ) -> Tuple[PyTree, Array]:
        ns, ndim = get_ensemble_shape(s)
        nc, _ = get_ensemble_shape(c)
        key1, key2 = random.split(key)
        u = random.uniform(key1, shape=(ns,))
        z = jnp.square((self.a - 1) * u + 1) / self.a
        ind = random.choice(key2, nc, shape=(ns,))
        updater = jax.vmap(lambda s, c, z: c - (c - s) * z)
        q = tree_map(lambda s, c: updater(s, c[ind], z), s, c)
        return q, (ndim - 1) * jnp.log(z)


class DiffEvol(SimpleRedBlue):
    gamma: Optional[Array] = None
    sigma: Array = 1.0e-5

    def propose_simple(
        self, key: random.KeyArray, s: PyTree, c: PyTree
    ) -> Tuple[Array, Array]:
        ns, ndim = get_ensemble_shape(s)
        nc, _ = get_ensemble_shape(c)
        key1, key2 = random.split(key)

        # This is a magic formula from the paper
        gamma0 = 2.38 / np.sqrt(2 * ndim) if self.gamma is None else self.gamma

        # These two slightly complicated lines are just to select two helper
        # walkers per target walker _without replacement_. This means that we'll
        # always get two different complementary walkers per target walker.
        choose2 = partial(random.choice, a=nc, replace=False, shape=(2,))
        inds = jax.vmap(choose2)(random.split(key1, ns))
        norm = random.normal(key2, shape=(ns,))

        @jax.vmap
        def update(s: Array, c: Array, norm: Array) -> Array:
            delta = c[1] - c[0]
            delta = (gamma0 + self.sigma * norm) * delta
            return s + delta

        return tree_map(
            lambda s, c: update(s, c[inds], norm), s, c
        ), jnp.zeros(ns)
