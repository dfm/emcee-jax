# mypy: ignore-errors

from itertools import product

import jax.numpy as jnp
import pytest
from jax import random, vmap
from jax.flatten_util import ravel_pytree
from scipy import stats

from emcee_jax import EnsembleSampler, moves


@pytest.mark.parametrize(
    "ndim,move",
    product(
        [1, 2],
        [
            moves.Stretch(),
            moves.DiffEvol(),
            moves.DiffEvolSlice(),
            moves.compose(moves.Stretch(), moves.DiffEvolSlice()),
        ],
    ),
)
def test_uniform(ndim, move, seed=1, num_walkers=32, num_steps=2_000):
    key = random.PRNGKey(seed)
    coords_key, init_key, sample_key = random.split(key, 3)
    coords = random.uniform(coords_key, shape=(num_walkers, ndim))
    sampler = EnsembleSampler(
        lambda x: jnp.sum(jnp.where((0 < x) & (x < 1), 0.0, -jnp.inf)),
        move=move,
    )
    state = sampler.init(init_key, coords)
    trace = sampler.sample(sample_key, state, num_steps)
    flat_samples = vmap(vmap(lambda x: ravel_pytree(x)[0]))(
        trace.samples.coordinates
    )
    assert flat_samples.shape == (num_steps, num_walkers, ndim)
    for n in range(ndim):
        _, pvalue = stats.kstest(
            flat_samples[::100, :, n].flatten(), "uniform"
        )
        assert pvalue > 0.01, n


@pytest.mark.parametrize(
    "pytree,move",
    product(
        [True, False],
        [moves.Stretch(), moves.DiffEvol(), moves.DiffEvolSlice()],
    ),
)
def test_normal(pytree, move, seed=1, num_walkers=32, num_steps=2_000):
    def log_prob(x):
        if pytree:
            x, y = x["x"], x["y"]
        else:
            x, y = x
        return -0.5 * (x**2 + y**2)

    key = random.PRNGKey(seed)
    coords_key, init_key, sample_key = random.split(key, 3)
    coords = random.normal(coords_key, shape=(num_walkers, 2))
    if pytree:
        coords = {"x": coords[:, 0], "y": coords[:, 1]}
    sampler = EnsembleSampler(log_prob, move=move)
    state = sampler.init(init_key, coords)
    trace = sampler.sample(sample_key, state, num_steps)
    flat_samples = vmap(vmap(lambda x: ravel_pytree(x)[0]))(
        trace.samples.coordinates
    )
    assert flat_samples.shape == (num_steps, num_walkers, 2)
    for n in range(2):
        _, pvalue = stats.kstest(flat_samples[::100, :, n].flatten(), "norm")
        assert pvalue > 0.01, n
