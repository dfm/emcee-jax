__all__ = ["Stretch", "DiffEvol"]

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import tree_map

from emcee_jax.moves.util import apply_accept
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
        self,
        log_prob_fn: WrappedLogProbFn,
        random_key: random.KeyArray,
        ensemble: FlatWalkerState,
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
    ) -> Tuple[MoveState, FlatWalkerState, SampleStats]:
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
        ) -> Tuple[MoveState, FlatWalkerState, SampleStats]:
            new_state, new_target, stats = self.propose(
                log_prob_fn, current_state, key, target, complementary
            )
            return (new_state, new_target, stats)

        move_state, half1, stats1 = half_step(move_state, key1, half1, half2)
        move_state, half2, stats2 = half_step(move_state, key2, half2, half1)
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
    ) -> Tuple[MoveState, FlatWalkerState, SampleStats]:
        key1, key2 = random.split(key)
        q, f = self.propose_simple(
            key1, target.coordinates, complementary.coordinates
        )
        nlp, ndet = jax.vmap(log_prob_fn)(q)
        updated = FlatWalkerState(
            coordinates=q,
            deterministics=ndet,
            log_probability=nlp,
            augments=target.augments,
        )

        diff = nlp - target.log_probability + f
        accept_prob = jnp.exp(diff)
        accept = accept_prob > random.uniform(key2, shape=diff.shape)
        updated = apply_accept(accept, target, updated)
        return (
            state,
            updated,
            {"accept": accept, "accept_prob": accept_prob},
        )


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


# def apply_accept(
#     accept: Array, target: FlatWalkerState, other: FlatWalkerState
# ) -> FlatWalkerState:
#     accepter = jax.vmap(lambda a, x, y: jnp.where(a, y, x))
#     accepter = partial(accepter, accept)
#     return tree_map(accepter, target, other)


# MEADS: https://proceedings.mlr.press/v151/hoffman22a.html


class HMCState(NamedTuple):
    q: Array
    p: Array
    log_prob: Array
    d_log_prob: Array
    deterministics: Array


def _leapfrog(
    value_and_grad: Callable[[Array], Tuple[Array, Array]],
    state: HMCState,
    *,
    eps: Array,
) -> HMCState:
    p = state.p + 0.5 * eps * state.d_log_prob
    q = state.q + eps * p
    (lp, det), dlogp = value_and_grad(q)
    p = p + 0.5 * eps * dlogp
    return HMCState(
        q=q, p=p, log_prob=lp, d_log_prob=dlogp, deterministics=det
    )


def persistent_ghmc(
    random_key: random.KeyArray,
    state: HMCState,
    u: Array,
    *,
    value_and_grad: Callable[[Array], Tuple[Array, Array]],
    eps: Array,
    alpha: Array,
    delta: Array,
) -> Tuple[HMCState, Array, SampleStats]:
    u = (u + 1.0 + delta) % 2.0 - 1.0

    # Jitter momentum
    n = random.normal(random_key, state.p.shape)
    p = state.p * jnp.sqrt(1 - alpha) + jnp.sqrt(alpha) * n
    init = state._replace(p=p)

    # Run integrator
    state_ = _leapfrog(value_and_grad, state, eps=eps)

    # Accept/reject
    diff = init.log_prob - state_.log_prob
    diff += -0.5 * jnp.sum(jnp.square(init.p))
    diff -= -0.5 * jnp.sum(jnp.square(state_.p))

    accept_prob = jnp.exp(diff)
    accept = jnp.log(jnp.abs(u)) < diff

    # Negate the initial momentum
    init = init._replace(p=-init.p)
    state = tree_map(lambda x, y: jnp.where(accept, x, y), state_, init)
    new_u = u * (~accept + accept / accept_prob)

    return (
        state,
        new_u,
        {
            "accept": accept,
            "accept_prob": accept_prob,
            "eps": eps,
            "alpha": alpha,
            "delta": delta,
            "u": new_u,
        },
    )


def _larget_eigenvalue_of_cov(x: Array, remove_mean: bool = True) -> Array:
    if remove_mean:
        x = x - jnp.mean(x, axis=0)
    trace_est = jnp.sum(jnp.square(x)) / x.shape[0]
    trace_sq_est = jnp.sum(jnp.square(x @ x.T)) / x.shape[0] ** 2
    return trace_sq_est / trace_est


@dataclass(frozen=True)
class MEADS(RedBlue):
    step_size_multiplier: Array = 0.5
    damping_slowdown: Array = 1.0
    diagonal_preconditioning: bool = True

    def init(
        self,
        log_prob_fn: WrappedLogProbFn,
        random_key: random.KeyArray,
        ensemble: FlatWalkerState,
    ) -> StepState:
        key1, key2 = random.split(random_key)

        augments = {} if ensemble.augments is None else ensemble.augments
        augments["u"] = random.uniform(
            key1, (ensemble.coordinates.shape[0],), minval=-1, maxval=1
        )
        augments["p"] = random.normal(key2, ensemble.coordinates.shape)
        dlogp, _ = jax.vmap(jax.grad(log_prob_fn, has_aux=True))(
            ensemble.coordinates
        )
        augments["d_log_prob"] = dlogp

        updated = ensemble._replace(augments=augments)
        return StepState(move_state={"iteration": 0}, walker_state=updated)

    def propose(
        self,
        log_prob_fn: WrappedLogProbFn,
        move_state: MoveState,
        key: random.KeyArray,
        target: FlatWalkerState,
        complementary: FlatWalkerState,
    ) -> Tuple[MoveState, FlatWalkerState, SampleStats]:
        assert move_state is not None
        assert target.augments is not None
        assert complementary.augments is not None

        # Apply preconditioning
        if self.diagonal_preconditioning:
            sigma = jnp.std(complementary.coordinates, axis=0)
        else:
            sigma = 1.0
        scaled_coords = complementary.coordinates / sigma
        scaled_grads = complementary.augments["d_log_prob"] * sigma

        # Step size
        max_eig_step = _larget_eigenvalue_of_cov(
            scaled_grads, remove_mean=False
        )
        eps = self.step_size_multiplier / jnp.sqrt(max_eig_step)
        eps = jnp.minimum(1.0, eps)

        # Damping
        max_eig_damp = _larget_eigenvalue_of_cov(scaled_coords)
        gamma = eps / jnp.sqrt(max_eig_damp)
        gamma = jnp.maximum(
            self.damping_slowdown / move_state["iteration"], gamma
        )
        alpha = 1 - jnp.exp(-2 * gamma)
        delta = 0.5 * alpha

        init = HMCState(
            q=target.coordinates,
            p=target.augments["p"],
            log_prob=target.log_probability,
            d_log_prob=target.augments["d_log_prob"],
            deterministics=target.deterministics,
        )
        step = jax.vmap(
            partial(
                persistent_ghmc,
                value_and_grad=jax.value_and_grad(log_prob_fn, has_aux=True),
                eps=eps * sigma,
                alpha=alpha,
                delta=delta,
            )
        )
        new_state, new_u, stats = step(
            random.split(key, target.coordinates.shape[0]),
            init,
            target.augments["u"],
        )

        updated = FlatWalkerState(
            coordinates=new_state.q,
            deterministics=new_state.deterministics,
            log_probability=new_state.log_prob,
            augments={
                "u": new_u,
                "p": new_state.p,
                "d_log_prob": new_state.d_log_prob,
            },
        )
        return (
            dict(move_state, iteration=move_state.pop("iteration") + 1),
            updated,
            stats,
        )
