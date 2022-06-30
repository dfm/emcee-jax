__all__ = ["Trace", "trace_to_inference_data"]

from typing import TYPE_CHECKING, Any, NamedTuple

import jax.numpy as jnp
from jax import device_get
from jax.tree_util import tree_flatten, tree_map

import emcee_jax
from emcee_jax.types import PyTree, SampleStats, WalkerState

if TYPE_CHECKING:
    from arviz import InferenceData


class Trace(NamedTuple):
    final_state: WalkerState
    final_extras: PyTree
    samples: WalkerState
    extras: PyTree
    sample_stats: SampleStats

    def to_inference_data(self, **kwargs: Any) -> "InferenceData":
        return trace_to_inference_data(self, **kwargs)


def trace_to_inference_data(trace: Trace, **kwargs: Any) -> "InferenceData":
    from arviz import InferenceData, dict_to_dataset

    # Deal with different possible PyTree shapes
    samples = trace.samples.coordinates
    if not isinstance(samples, dict):
        flat, _ = tree_flatten(samples)
        samples = {f"param_{n}": x for n, x in enumerate(flat)}

    # Deterministics also live in samples
    deterministics = trace.samples.deterministics
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
    sample_stats = dict(trace.sample_stats, lp=trace.samples.log_probability)
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
