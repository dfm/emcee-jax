from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Tuple, Union

import jax
import jax.linear_util as lu
import jax.numpy as jnp
from jax import device_get, random
from jax.tree_util import tree_flatten, tree_map

from emcee_jax._src.ensemble import Ensemble
from emcee_jax._src.log_prob_fn import LogProbFn, wrap_log_prob_fn
from emcee_jax._src.moves.core import Extras, Move, MoveState, Stretch
from emcee_jax._src.types import Array, PyTree, SampleStats

if TYPE_CHECKING:
    from arviz import InferenceData


class SamplerState(NamedTuple):
    move_state: MoveState
    ensemble: Ensemble
    extras: Extras


class Trace(NamedTuple):
    final_state: SamplerState
    samples: SamplerState
    sample_stats: SampleStats

    def to_inference_data(self, **kwargs: Any) -> "InferenceData":
        from arviz import InferenceData, dict_to_dataset

        import emcee_jax

        # Deal with different possible PyTree shapes
        samples = self.samples.ensemble.coordinates
        if not isinstance(samples, dict):
            flat, _ = tree_flatten(samples)
            samples = {f"param_{n}": x for n, x in enumerate(flat)}

        # Deterministics also live in samples
        deterministics = self.samples.ensemble.deterministics
        if deterministics is not None:
            if not isinstance(deterministics, dict):
                flat, _ = tree_flatten(deterministics)
                deterministics = {f"det_{n}": x for n, x in enumerate(flat)}
            for k in list(deterministics.keys()):
                if k in samples:
                    assert f"{k}_det" not in samples
                    deterministics[f"{k}_det"] = deterministics.pop(k)
            samples = dict(samples, **deterministics)

        # ArviZ has a different convention about axis locations. It wants (chain,
        # draw, ...) whereas we produce (draw, chain, ...).
        samples = tree_map(lambda x: jnp.swapaxes(x, 0, 1), samples)

        # Convert sample stats to ArviZ's preferred style
        sample_stats = dict(
            self.sample_stats, lp=self.samples.ensemble.log_probability
        )
        renames = [("accept_prob", "acceptance_rate")]
        for old, new in renames:
            if old in sample_stats:
                sample_stats[new] = sample_stats.pop(old)
        sample_stats = tree_map(lambda x: jnp.swapaxes(x, 0, 1), sample_stats)

        return InferenceData(
            posterior=dict_to_dataset(device_get(samples), library=emcee_jax),
            sample_stats=dict_to_dataset(
                device_get(sample_stats), library=emcee_jax
            ),
            **kwargs,
        )


@dataclass(frozen=True, init=False)
class EnsembleSampler:
    wrapped_log_prob_fn: lu.WrappedFun
    move: Move

    def __init__(
        self,
        log_prob_fn: LogProbFn,
        *,
        move: Optional[Move] = None,
        log_prob_args: Tuple[Any, ...] = (),
        log_prob_kwargs: Optional[Dict[str, Any]] = None,
    ):
        log_prob_kwargs = {} if log_prob_kwargs is None else log_prob_kwargs
        wrapped_log_prob_fn = wrap_log_prob_fn(
            log_prob_fn, *log_prob_args, **log_prob_kwargs
        )
        object.__setattr__(self, "wrapped_log_prob_fn", wrapped_log_prob_fn)

        move = Stretch() if move is None else move
        object.__setattr__(self, "move", move)

    def init(
        self,
        random_key: random.KeyArray,
        ensemble: Union[Ensemble, Array],
    ) -> SamplerState:
        initial_ensemble = Ensemble.init(self.wrapped_log_prob_fn, ensemble)
        move_state, extras = self.move.init(random_key, initial_ensemble)
        return SamplerState(move_state, initial_ensemble, extras)

    def step(
        self,
        random_key: random.KeyArray,
        state: SamplerState,
        *,
        tune: bool = False,
    ) -> Tuple[SamplerState, SampleStats]:
        new_state, stats = self.move.step(
            self.wrapped_log_prob_fn, random_key, *state, tune=tune
        )
        return SamplerState(*new_state), stats

    def sample(
        self,
        random_key: random.KeyArray,
        state: SamplerState,
        num_steps: int,
        *,
        tune: bool = False,
    ) -> Trace:
        def one_step(
            state: SamplerState, key: random.KeyArray
        ) -> Tuple[SamplerState, Tuple[SamplerState, SampleStats]]:
            state, stats = self.step(key, state, tune=tune)
            return state, (state, stats)

        keys = random.split(random_key, num_steps)
        final, (trace, stats) = jax.lax.scan(one_step, state, keys)
        return Trace(final, trace, stats)
