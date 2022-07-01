# mypy: ignore-errors

from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import jax
import jax.linear_util as lu
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax.tree_util import tree_map

from emcee_jax._src.moves.core import MoveState, RedBlue, StepState
from emcee_jax._src.types import Array, PyTree, SampleStats


class HMCState(NamedTuple):
    coordinates: Array
    momenta: Array
    log_probability: Array
    grad_log_probability: Array
    deterministics: Array


def leapfrog(
    log_prob_and_grad_fn: WrappedLogProbFn,
    state: HMCState,
    *,
    step_size: Array,
) -> HMCState:
    p = state.coordinates + 0.5 * step_size * state.grad_log_probability
    q = state.momenta + step_size * p
    (log_prob, det), dlogp = log_prob_and_grad_fn(q)
    p = p + 0.5 * step_size * dlogp
    return state._replace(
        coordinates=q,
        momenta=p,
        log_probability=log_prob,
        grad_log_probability=dlogp,
        deterministics=det,
    )


def hmc(
    log_prob_and_grad_fn: WrappedLogProbFn,
    random_key: random.KeyArray,
    state: HMCState,
    step_size: Array,
    num_steps: Array,
) -> Tuple[HMCState, SampleStats]:
    momenta_key, accept_key = random.split(random_key)
    norm = random.normal(momenta_key, state.momenta.shape)
    init = state._replace(momenta=norm)

    def step(_: Any, state: HMCState) -> HMCState:
        new_state = leapfrog(log_prob_and_grad_fn, state, step_size=step_size)
        return new_state

    proposed = jax.lax.fori_loop(0, num_steps, step, init)
    proposed = proposed._replace(momenta=-proposed.momenta)

    return mh_accept(init, proposed, key=accept_key)


def mh_accept(
    init: HMCState,
    prop: HMCState,
    *,
    key: Optional[random.KeyArray] = None,
    level: Optional[Array] = None,
) -> Tuple[HMCState, SampleStats]:
    prop_lp = prop.log_probability - 0.5 * jnp.sum(jnp.square(prop.momenta))
    init_lp = init.log_probability - 0.5 * jnp.sum(jnp.square(init.momenta))
    log_accept_prob = prop_lp - init_lp

    if level is None:
        assert key is not None
        u = random.uniform(key)
        accept = jnp.log(u) < log_accept_prob
        return tree_map(lambda x, y: jnp.where(accept, y, x), init, prop), {
            "accept": accept,
            "log_accept_prob": log_accept_prob,
        }

    raise NotImplementedError


class HMC(RedBlue):
    step_size: Array = 0.1
    num_steps: Array = 50
    max_step_size: Optional[Array] = None
    max_num_steps: Optional[Array] = None
    precondition: bool = False

    def init(
        self,
        log_prob_fn: WrappedLogProbFn,
        key: random.KeyArray,
        ensemble: FlatWalkerState,
    ) -> StepState:
        extras = {} if ensemble.extras is None else ensemble.extras
        extras["momenta"] = random.normal(key, ensemble.coordinates.shape)
        dlogp, _ = jax.vmap(jax.grad(log_prob_fn, has_aux=True))(
            ensemble.coordinates
        )
        extras["grad_log_probability"] = dlogp
        ensemble = ensemble._replace(extras=extras)
        return StepState(move_state={"iteration": 0}, walker_state=ensemble)

    def get_step_size(self, key: random.KeyArray) -> Array:
        if self.max_step_size is None:
            return self.step_size
        return jnp.exp(
            random.uniform(
                key,
                minval=jnp.log(self.step_size),
                maxval=jnp.log(self.max_step_size),
            )
        )

    def get_num_steps(self, key: random.KeyArray) -> Array:
        if self.max_num_steps is None:
            return jnp.asarray(self.num_steps, dtype=int)
        return jnp.floor(
            random.uniform(
                key,
                minval=self.num_steps,
                maxval=self.max_num_steps + 1,
            )
        ).astype(int)

    def propose(
        self,
        log_prob_fn: WrappedLogProbFn,
        move_state: MoveState,
        key: random.KeyArray,
        target: FlatWalkerState,
        complementary: FlatWalkerState,
        *,
        tune: bool,
    ) -> Tuple[MoveState, FlatWalkerState, SampleStats]:
        assert move_state is not None
        assert target.extras is not None
        assert complementary.extras is not None

        if self.precondition:
            cov = jnp.cov(complementary.coordinates, rowvar=0)
            ell = jnp.linalg.cholesky(cov)
            condition = partial(jsp.linalg.solve_triangular, ell, lower=True)
            coords = jax.vmap(condition)(target.coordinates)
            uncondition = lambda x: ell @ x
            wrapped_log_prob_fn = precondition_log_prob_fn(
                lu.wrap_init(log_prob_fn), uncondition
            ).call_wrapped
        else:
            coords = target.coordinates
            wrapped_log_prob_fn = log_prob_fn

        num_target = target.coordinates.shape[0]
        step_size_key, num_steps_key, step_key = random.split(key, 3)

        init = HMCState(
            coordinates=coords,
            momenta=target.extras["momenta"],
            log_probability=target.log_probability,
            grad_log_probability=target.extras["grad_log_probability"],
            deterministics=target.deterministics,
        )

        step_size = self.get_step_size(step_size_key)
        num_steps = self.get_num_steps(num_steps_key)
        if jnp.ndim(step_size) >= 1 or jnp.ndim(num_steps) >= 1:
            step_size = jnp.broadcast_to(step_size, num_target)
            num_steps = jnp.broadcast_to(step_size, num_target)
            step = jax.vmap(
                partial(
                    hmc, jax.value_and_grad(wrapped_log_prob_fn, has_aux=True)
                )
            )
            result, stats = step(
                random.split(step_key, num_target), init, step_size, num_steps
            )

        else:
            step = jax.vmap(
                partial(
                    hmc,
                    jax.value_and_grad(wrapped_log_prob_fn, has_aux=True),
                    step_size=step_size,
                    num_steps=num_steps,
                )
            )
            result, stats = step(random.split(step_key, num_target), init)

            step_size = jnp.broadcast_to(step_size, num_target)
            num_steps = jnp.broadcast_to(step_size, num_target)

        if self.precondition:
            coords = jax.vmap(uncondition)(result.coordinates)
        else:
            coords = result.coordinates

        updated = target._replace(
            coordinates=coords,
            deterministics=result.deterministics,
            log_probability=result.log_probability,
            extras={
                "momenta": result.momenta,
                "grad_log_probability": result.grad_log_probability,
            },
        )
        return (
            dict(move_state, iteration=move_state.pop("iteration") + 1),
            updated,
            dict(stats, step_size=step_size, num_steps=num_steps),
        )


@lu.transformation
def precondition_log_prob_fn(
    uncondition: Callable[[Array], Array], x: Array
) -> Generator[
    Tuple[Array, Union[PyTree, Dict[str, Any]]],
    Tuple[Array, Union[PyTree, Dict[str, Any]]],
    None,
]:
    result = yield (uncondition(x),), {}
    yield result


# def persistent_ghmc(
#     random_key: random.KeyArray,
#     state: HMCState,
#     u: Array,
#     *,
#     value_and_grad: Callable[[Array], Tuple[Array, Array]],
#     eps: Array,
#     alpha: Array,
#     delta: Array,
# ) -> Tuple[HMCState, Array, SampleStats]:
#     u = (u + 1.0 + delta) % 2.0 - 1.0

#     # Jitter momentum
#     n = random.normal(random_key, state.p.shape)
#     p = state.p * jnp.sqrt(1 - alpha) + jnp.sqrt(alpha) * n
#     init = state._replace(p=p)

#     # Run integrator
#     state_ = _leapfrog(value_and_grad, state, eps=eps)

#     # Accept/reject
#     diff = init.log_prob - state_.log_prob
#     diff += -0.5 * jnp.sum(jnp.square(init.p))
#     diff -= -0.5 * jnp.sum(jnp.square(state_.p))

#     accept_prob = jnp.exp(diff)
#     accept = jnp.log(jnp.abs(u)) < diff

#     # Negate the initial momentum
#     init = init._replace(p=-init.p)
#     state = tree_map(lambda x, y: jnp.where(accept, x, y), state_, init)
#     new_u = u * (~accept + accept / accept_prob)

#     return (
#         state,
#         new_u,
#         {
#             "accept": accept,
#             "accept_prob": accept_prob,
#             "eps": eps,
#             "alpha": alpha,
#             "delta": delta,
#             "u": new_u,
#         },
#     )


# MEADS: https://proceedings.mlr.press/v151/hoffman22a.html


# class HMCState(NamedTuple):
#     q: Array
#     p: Array
#     log_prob: Array
#     d_log_prob: Array
#     deterministics: Array


# def _leapfrog(
#     value_and_grad: Callable[[Array], Tuple[Array, Array]],
#     state: HMCState,
#     *,
#     eps: Array,
# ) -> HMCState:
#     p = state.p + 0.5 * eps * state.d_log_prob
#     q = state.q + eps * p
#     (lp, det), dlogp = value_and_grad(q)
#     p = p + 0.5 * eps * dlogp
#     return HMCState(
#         q=q, p=p, log_prob=lp, d_log_prob=dlogp, deterministics=det
#     )


# def persistent_ghmc(
#     random_key: random.KeyArray,
#     state: HMCState,
#     u: Array,
#     *,
#     value_and_grad: Callable[[Array], Tuple[Array, Array]],
#     eps: Array,
#     alpha: Array,
#     delta: Array,
# ) -> Tuple[HMCState, Array, SampleStats]:
#     u = (u + 1.0 + delta) % 2.0 - 1.0

#     # Jitter momentum
#     n = random.normal(random_key, state.p.shape)
#     p = state.p * jnp.sqrt(1 - alpha) + jnp.sqrt(alpha) * n
#     init = state._replace(p=p)

#     # Run integrator
#     state_ = _leapfrog(value_and_grad, state, eps=eps)

#     # Accept/reject
#     diff = init.log_prob - state_.log_prob
#     diff += -0.5 * jnp.sum(jnp.square(init.p))
#     diff -= -0.5 * jnp.sum(jnp.square(state_.p))

#     accept_prob = jnp.exp(diff)
#     accept = jnp.log(jnp.abs(u)) < diff

#     # Negate the initial momentum
#     init = init._replace(p=-init.p)
#     state = tree_map(lambda x, y: jnp.where(accept, x, y), state_, init)
#     new_u = u * (~accept + accept / accept_prob)

#     return (
#         state,
#         new_u,
#         {
#             "accept": accept,
#             "accept_prob": accept_prob,
#             "eps": eps,
#             "alpha": alpha,
#             "delta": delta,
#             "u": new_u,
#         },
#     )


# def _larget_eigenvalue_of_cov(x: Array, remove_mean: bool = True) -> Array:
#     if remove_mean:
#         x = x - jnp.mean(x, axis=0)
#     trace_est = jnp.sum(jnp.square(x)) / x.shape[0]
#     trace_sq_est = jnp.sum(jnp.square(x @ x.T)) / x.shape[0] ** 2
#     return trace_sq_est / trace_est


# @dataclass(frozen=True)
# class MEADS(RedBlue):
#     step_size_multiplier: Array = 0.5
#     damping_slowdown: Array = 1.0
#     diagonal_preconditioning: bool = True

#     def init(
#         self,
#         log_prob_fn: WrappedLogProbFn,
#         random_key: random.KeyArray,
#         ensemble: FlatWalkerState,
#     ) -> StepState:
#         key1, key2 = random.split(random_key)

#         augments = {} if ensemble.augments is None else ensemble.augments
#         augments["u"] = random.uniform(
#             key1, (ensemble.coordinates.shape[0],), minval=-1, maxval=1
#         )
#         augments["p"] = random.normal(key2, ensemble.coordinates.shape)
#         dlogp, _ = jax.vmap(jax.grad(log_prob_fn, has_aux=True))(
#             ensemble.coordinates
#         )
#         augments["d_log_prob"] = dlogp

#         updated = ensemble._replace(augments=augments)
#         return StepState(move_state={"iteration": 0}, walker_state=updated)

#     def propose(
#         self,
#         log_prob_fn: WrappedLogProbFn,
#         move_state: MoveState,
#         key: random.KeyArray,
#         target: FlatWalkerState,
#         complementary: FlatWalkerState,
#     ) -> Tuple[MoveState, FlatWalkerState, SampleStats]:
#         assert move_state is not None
#         assert target.augments is not None
#         assert complementary.augments is not None

#         # Apply preconditioning
#         if self.diagonal_preconditioning:
#             sigma = jnp.std(complementary.coordinates, axis=0)
#         else:
#             sigma = 1.0
#         scaled_coords = complementary.coordinates / sigma
#         scaled_grads = complementary.augments["d_log_prob"] * sigma

#         # Step size
#         max_eig_step = _larget_eigenvalue_of_cov(
#             scaled_grads, remove_mean=False
#         )
#         eps = self.step_size_multiplier / jnp.sqrt(max_eig_step)
#         eps = jnp.minimum(1.0, eps)

#         # Damping
#         max_eig_damp = _larget_eigenvalue_of_cov(scaled_coords)
#         gamma = eps / jnp.sqrt(max_eig_damp)
#         gamma = jnp.maximum(
#             self.damping_slowdown / move_state["iteration"], gamma
#         )
#         alpha = 1 - jnp.exp(-2 * gamma)
#         delta = 0.5 * alpha

#         init = HMCState(
#             q=target.coordinates,
#             p=target.augments["p"],
#             log_prob=target.log_probability,
#             d_log_prob=target.augments["d_log_prob"],
#             deterministics=target.deterministics,
#         )
#         step = jax.vmap(
#             partial(
#                 persistent_ghmc,
#                 value_and_grad=jax.value_and_grad(log_prob_fn, has_aux=True),
#                 eps=eps * sigma,
#                 alpha=alpha,
#                 delta=delta,
#             )
#         )
#         new_state, new_u, stats = step(
#             random.split(key, target.coordinates.shape[0]),
#             init,
#             target.augments["u"],
#         )

#         updated = FlatWalkerState(
#             coordinates=new_state.q,
#             deterministics=new_state.deterministics,
#             log_probability=new_state.log_prob,
#             augments={
#                 "u": new_u,
#                 "p": new_state.p,
#                 "d_log_prob": new_state.d_log_prob,
#             },
#         )
#         return (
#             dict(move_state, iteration=move_state.pop("iteration") + 1),
#             updated,
#             stats,
#         )
