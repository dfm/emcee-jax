# mypy: ignore-errors

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import emcee_jax
from emcee_jax.host_callback import wrap_python_log_prob_fn


def build_rosenbrock(pytree_input=False, deterministics=False):
    def rosenbrock(theta, a1=100.0, a2=20.0):
        if pytree_input:
            x1, x2 = theta["x"], theta["y"]
        else:
            x1, x2 = theta
        log_prob = -(a1 * (x2 - x1**2) ** 2 + (1 - x1) ** 2) / a2

        if deterministics:
            return log_prob, {"some_number": x1 + jnp.sin(x2)}
        return log_prob

    return rosenbrock


def test_basic(seed=0, num_walkers=5, num_steps=21):
    log_prob = build_rosenbrock()
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
    coords = jax.random.normal(key1, shape=(num_walkers, 2))
    sample = emcee_jax.sampler(log_prob)
    trace = sample(key2, coords, steps=num_steps)
    assert trace.samples.deterministics is None
    assert trace.samples.coordinates.shape == (num_steps, num_walkers, 2)
    assert trace.samples.log_probability.shape == (num_steps, num_walkers)
    assert trace.sample_stats["accept"].shape == (num_steps, num_walkers)


def test_pytree_input(seed=0, num_walkers=5, num_steps=21):
    log_prob = build_rosenbrock(pytree_input=True)
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    coords = {
        "x": jax.random.normal(key1, shape=(num_walkers,)),
        "y": jax.random.normal(key2, shape=(num_walkers,)),
    }
    sample = emcee_jax.sampler(log_prob)
    trace = sample(key3, coords, steps=num_steps)
    shape = (num_steps, num_walkers)
    assert trace.samples.deterministics is None
    assert trace.samples.coordinates["x"].shape == shape
    assert trace.samples.log_probability.shape == shape
    assert trace.sample_stats["accept"].shape == shape


def test_deterministics(seed=0, num_walkers=5, num_steps=21):
    log_prob = build_rosenbrock(deterministics=True)
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
    coords = jax.random.normal(key1, shape=(num_walkers, 2))
    sample = emcee_jax.sampler(log_prob)
    trace = sample(key2, coords, steps=num_steps)
    shape = (num_steps, num_walkers)
    assert trace.samples.deterministics["some_number"].shape == shape
    assert trace.samples.coordinates.shape == (num_steps, num_walkers, 2)
    assert trace.samples.log_probability.shape == shape
    assert trace.sample_stats["accept"].shape == shape


def test_host_callback(seed=0, num_walkers=5, num_steps=21):
    import numpy as np

    @wrap_python_log_prob_fn
    def log_prob(theta, a1=100.0, a2=20.0):
        x1, x2 = theta
        return -(a1 * np.square(x2 - x1**2) + np.square(1 - x1)) / a2

    num_walkers, num_steps = 100, 1000
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
    coords = jax.random.normal(key1, shape=(num_walkers, 2))
    sample = emcee_jax.sampler(log_prob)
    trace = sample(key2, coords, steps=num_steps)
    assert trace.samples.deterministics is None
    assert trace.samples.coordinates.shape == (num_steps, num_walkers, 2)
    assert trace.samples.log_probability.shape == (num_steps, num_walkers)
    assert trace.sample_stats["accept"].shape == (num_steps, num_walkers)


def test_init_errors(seed=0, num_walkers=5, num_steps=21):
    def check_raises(log_prob):
        key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
        coords = jax.random.normal(key1, shape=(num_walkers, 2))
        sample = emcee_jax.sampler(log_prob)
        with pytest.raises(ValueError):
            sample(key2, coords, steps=num_steps)

    check_raises(lambda *_: None)
    check_raises(lambda *_: jnp.ones(2))
    check_raises(lambda *_: jnp.ones(4))
    check_raises(lambda *_: jnp.ones(5))


def test_to_inference_data_basic(seed=0, num_walkers=5, num_steps=21):
    pytest.importorskip("arviz")
    log_prob = build_rosenbrock()
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
    coords = jax.random.normal(key1, shape=(num_walkers, 2))
    sample = emcee_jax.sampler(log_prob)
    trace = sample(key2, coords, steps=num_steps)
    data = trace.to_inference_data()

    assert data.posterior.dims["chain"] == num_walkers
    assert data.posterior.dims["draw"] == num_steps
    np.testing.assert_allclose(
        np.swapaxes(data.posterior.param_0.values, 0, 1),
        trace.samples.coordinates,
    )

    assert data.sample_stats.dims["chain"] == num_walkers
    assert data.sample_stats.dims["draw"] == num_steps
    assert data.sample_stats.lp.values.shape == (num_walkers, num_steps)
    np.testing.assert_allclose(
        data.sample_stats.lp.values.T, trace.samples.log_probability
    )


def test_to_inference_data_full(seed=0, num_walkers=5, num_steps=21):
    pytest.importorskip("arviz")
    log_prob = build_rosenbrock(pytree_input=True, deterministics=True)
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    coords = {
        "x": jax.random.normal(key1, shape=(num_walkers,)),
        "y": jax.random.normal(key2, shape=(num_walkers,)),
    }
    sample = emcee_jax.sampler(log_prob)
    trace = sample(key3, coords, steps=num_steps)
    data = trace.to_inference_data()

    assert data.posterior.dims["chain"] == num_walkers
    assert data.posterior.dims["draw"] == num_steps
    np.testing.assert_allclose(
        data.posterior.x.values.T, trace.samples.coordinates["x"]
    )
    np.testing.assert_allclose(
        data.posterior.y.values.T, trace.samples.coordinates["y"]
    )
    np.testing.assert_allclose(
        data.posterior.some_number.values.T,
        trace.samples.deterministics["some_number"],
    )

    assert data.sample_stats.dims["chain"] == num_walkers
    assert data.sample_stats.dims["draw"] == num_steps
    assert data.sample_stats.lp.values.shape == (num_walkers, num_steps)
    np.testing.assert_allclose(
        data.sample_stats.lp.values.T, trace.samples.log_probability
    )
