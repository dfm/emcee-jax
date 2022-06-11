__all__ = ["stretch"]

from functools import partial, wraps
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from emcee_jax.types import (
    Array,
    InitFn,
    MoveFn,
    MoveState,
    SampleStats,
    StepFn,
    WalkerState,
    WrappedLogProbFn,
)

ProposalFn = Callable[[random.KeyArray, Array, Array], Tuple[Array, Array]]


class red_blue:
    def __init__(
        self, build_proposal: Callable[..., Tuple[InitFn, ProposalFn]]
    ) -> None:
        self.build_proposal = build_proposal

    def __call__(self, *args: Any, **kwargs: Any) -> MoveFn:
        @wraps(self.build_proposal)
        def move_impl(log_prob_fn: WrappedLogProbFn) -> Tuple[InitFn, StepFn]:
            init, proposal = self.build_proposal(*args, **kwargs)

            def step(
                state: MoveState,
                random_key: random.KeyArray,
                ensemble: WalkerState,
            ) -> Tuple[SampleStats, WalkerState]:
                key1, key2 = random.split(random_key)
                nwalkers, _ = ensemble.coordinates.shape
                mid = nwalkers // 2

                a = ensemble.coordinates[:mid]
                b = ensemble.coordinates[mid:]

                def half_step(
                    key: random.KeyArray,
                    s: Array,
                    c: Array,
                    det: Array,
                    lp: Array,
                ) -> Tuple[Array, Array, Array, Array, Array]:
                    key1, key2 = random.split(key)
                    q, f = proposal(key1, s, c)
                    nlp, ndet = jax.vmap(log_prob_fn)(q)
                    diff = f + nlp - lp
                    accept_prob = jnp.exp(diff)
                    accept = accept_prob > random.uniform(key2, shape=lp.shape)
                    return (
                        accept_prob,
                        accept,
                        jnp.where(accept[:, None], q, s),
                        jnp.where(accept[:, None], det, ndet),
                        jnp.where(accept, nlp, lp),
                    )

                ap1, acc1, a, det1, lp1 = half_step(
                    key1,
                    a,
                    b,
                    ensemble.deterministics[:mid],
                    ensemble.log_probability[:mid],
                )
                ap2, acc2, b, det2, lp2 = half_step(
                    key2,
                    b,
                    a,
                    ensemble.deterministics[mid:],
                    ensemble.log_probability[mid:],
                )

                return (
                    {
                        "accept_prob": jnp.concatenate((ap1, ap2)),
                        "accept": jnp.concatenate((acc1, acc2)),
                    },
                    WalkerState(
                        coordinates=jnp.concatenate((a, b), axis=0),
                        deterministics=jnp.concatenate((det1, det2), axis=0),
                        log_probability=jnp.concatenate((lp1, lp2), axis=0),
                    ),
                )

            return init, step

        return move_impl


@red_blue
def stretch(*, a: float = 2.0) -> Tuple[InitFn, ProposalFn]:
    def proposal(
        key: random.KeyArray, s: Array, c: Array
    ) -> Tuple[Array, Array]:
        ns, ndim = s.shape
        key1, key2 = random.split(key)
        u = random.uniform(key1, shape=(ns,))
        z = jnp.square((a - 1) * u + 1) / a
        c = random.choice(key2, c, shape=(ns,))
        q = c - (c - s) * z[..., None]
        return q, (ndim - 1) * jnp.log(z)

    return lambda _: None, proposal


@red_blue
def differential_evolution(
    *, sigma: float = 1.0e-5, gamma: Optional[float] = None
) -> Tuple[InitFn, ProposalFn]:
    def proposal(
        key: random.KeyArray, s: Array, c: Array
    ) -> Tuple[Array, Array]:
        ns, ndim = s.shape
        key1, key2 = jax.random.split(key)

        # This is a magic formula from the paper
        gamma0 = 2.38 / np.sqrt(2 * ndim) if gamma is None else gamma

        # These two slightly complicated lines are just to select two helper
        # walkers per target walker _without replacement_. This means that we'll
        # always get two different complementary walkers per target walker.
        choose2 = partial(jax.random.choice, a=c, replace=False, shape=(2,))
        helpers = jax.vmap(choose2)(jax.random.split(key1, ns))

        # Use the vector between helper walkers to update the target walker
        delta = jnp.squeeze(jnp.diff(helpers, axis=1))
        norm = jax.random.normal(key2, shape=(ns,))
        delta = (gamma0 + sigma * norm[:, None]) * delta

        return s + delta, jnp.zeros(ns)

    return lambda _: None, proposal
