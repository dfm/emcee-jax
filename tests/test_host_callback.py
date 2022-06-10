# mypy: ignore-errors

import jax
import jax.numpy as jnp
import numpy as np

from emcee_jax.host_callback import wrap_python_log_prob_fn


def test_host_callback_vmap(seed=0):
    func = wrap_python_log_prob_fn(
        lambda x, **kwargs: -0.5 * np.sum(np.square(x))
    )

    arg = jax.random.normal(jax.random.PRNGKey(seed), (10, 3))
    expected = jnp.stack([func(row) for row in arg])
    computed = jax.vmap(func)(arg)

    np.testing.assert_allclose(computed, expected)
