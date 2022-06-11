# mypy: ignore-errors

import jax
import jax.numpy as jnp
import numpy as np

from emcee_jax.host_callback import wrap_python_log_prob_fn


def test_host_callback_vmap(seed=0):
    func = wrap_python_log_prob_fn(lambda x: -0.5 * np.sum(np.square(x)))

    arg = jax.random.normal(jax.random.PRNGKey(seed), (10, 3))
    expected = jnp.stack([func(row) for row in arg])
    computed = jax.vmap(func)(arg)

    np.testing.assert_allclose(computed, expected, rtol=1e-6)


def test_host_callback_vmap_pytree(seed=0):
    func_py = wrap_python_log_prob_fn(
        lambda x: np.sum(np.square(x["x"])) + x["y"]
    )
    func_jax = lambda x: jnp.sum(jnp.square(x["x"])) + x["y"]

    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
    arg = {
        "x": jax.random.normal(key1, (10, 3)),
        "y": jax.random.normal(key2, (10,)),
    }
    expected = jax.vmap(func_jax)(arg)
    computed = jax.vmap(func_py)(arg)

    np.testing.assert_allclose(computed, expected, rtol=1e-6)
