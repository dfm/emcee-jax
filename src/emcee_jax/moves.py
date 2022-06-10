from functools import wraps
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import random

from emcee_jax.types import (
    Array,
    InitFn,
    MoveFn,
    MoveState,
    SamplerStats,
    StepFn,
    WalkerState,
    WrappedLogProbFn,
)

ProposalFn = Callable[[random.KeyArray, Array, Array], Tuple[Array, Array]]


def red_blue(
    build_proposal: Callable[..., Tuple[InitFn, ProposalFn]],
) -> MoveFn:
    @wraps(build_proposal)
    def move_impl(
        log_prob_fn: WrappedLogProbFn, *args: Any, **kwargs: Any
    ) -> Tuple[InitFn, StepFn]:
        init, proposal = build_proposal(*args, **kwargs)

        def step(
            state: MoveState,
            random_key: random.KeyArray,
            ensemble: WalkerState,
        ) -> Tuple[SamplerStats, WalkerState]:
            key1, key2 = random.split(random_key)
            nwalkers, _ = ensemble.coordinates.shape
            mid = nwalkers // 2

            a = ensemble.coordinates[:mid]
            b = ensemble.coordinates[mid:]

            def half_step(
                key: random.KeyArray, s: Array, c: Array, det: Array, lp: Array
            ) -> Tuple[Array, Array, Array, Array]:
                key1, key2 = random.split(key)
                q, f = proposal(key1, s, c)
                nlp, ndet = jax.vmap(log_prob_fn)(q)
                diff = f + nlp - lp
                accept = jnp.exp(diff) > random.uniform(key2, shape=lp.shape)
                return (
                    accept,
                    jnp.where(accept[:, None], q, s),
                    jnp.where(accept[:, None], det, ndet),
                    jnp.where(accept, nlp, lp),
                )

            acc1, a, det1, lp1 = half_step(
                key1,
                a,
                b,
                ensemble.deterministics[:mid],
                ensemble.log_probability[:mid],
            )
            acc2, b, det2, lp2 = half_step(
                key2,
                b,
                a,
                ensemble.deterministics[mid:],
                ensemble.log_probability[mid:],
            )

            return (
                {"accept": jnp.concatenate((acc1, acc2))},
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
