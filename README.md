# emcee-jax

An experiement.

A simple example:

```python
>>> import jax
>>> from emcee_jax.sampler import build_sampler
>>>
>>> def log_prob(theta, a1=100.0, a2=20.0):
...     x1, x2 = theta
...     return -(a1 * (x2 - x1**2)**2 + (1 - x1)**2) / a2
...
>>> num_walkers, num_steps = 100, 1000
>>> key1, key2 = jax.random.split(jax.random.PRNGKey(0))
>>> coords = jax.random.normal(key1, shape=(num_walkers, 2))
>>> sample = build_sampler(log_prob)
>>> trace = sample(key2, coords, steps=num_steps)

```

An example using PyTrees as input coordinates:

```python
>>> import jax
>>> from emcee_jax.sampler import build_sampler
>>>
>>> def log_prob(theta, a1=100.0, a2=20.0):
...     x1, x2 = theta["x"], theta["y"]
...     return -(a1 * (x2 - x1**2)**2 + (1 - x1)**2) / a2
...
>>> num_walkers, num_steps = 100, 1000
>>> key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)
>>> coords = {
...     "x": jax.random.normal(key1, shape=(num_walkers,)),
...     "y": jax.random.normal(key2, shape=(num_walkers,)),
... }
>>> sample = build_sampler(log_prob)
>>> trace = sample(key3, coords, steps=num_steps)

```

An example that includes deterministics:

```python
>>> import jax
>>> from emcee_jax.sampler import build_sampler
>>>
>>> def log_prob(theta, a1=100.0, a2=20.0):
...     x1, x2 = theta
...     some_number = x1 + jax.numpy.sin(x2)
...     log_prob_value = -(a1 * (x2 - x1**2)**2 + (1 - x1)**2) / a2
...
...     # This second argument can be any PyTree
...     return log_prob_value, {"some_number": some_number}
...
>>> num_walkers, num_steps = 100, 1000
>>> key1, key2 = jax.random.split(jax.random.PRNGKey(0))
>>> coords = jax.random.normal(key1, shape=(num_walkers, 2))
>>> sample = build_sampler(log_prob)
>>> trace = sample(key2, coords, steps=num_steps)

```

You can even use pure-Python log probability functions:

```python
>>> import jax
>>> import numpy as np
>>> from emcee_jax.sampler import build_sampler
>>> from emcee_jax.host_callback import wrap_python_log_prob_fn
>>>
>>> # A log prob function that uses numpy, not jax.numpy inside
>>> @wrap_python_log_prob_fn
... def log_prob(theta, a1=100.0, a2=20.0):
...     x1, x2 = theta
...     return -(a1 * np.square(x2 - x1**2) + np.square(1 - x1)) / a2
...
>>> num_walkers, num_steps = 100, 1000
>>> key1, key2 = jax.random.split(jax.random.PRNGKey(0))
>>> coords = jax.random.normal(key1, shape=(num_walkers, 2))
>>> sample = build_sampler(log_prob)
>>> trace = sample(key2, coords, steps=num_steps)

```
