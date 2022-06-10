# emcee-jax

An experiement.

An example:

```python
import jax
from emcee_jax.sampler import build_sampler

def log_prob(theta, a1=100.0, a2=20.0):
    return -(a1 * (theta["y"] - theta["x"]**2)**2 + (1 - theta["x"])**2) / a2

key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)
coords = {
    "x": jax.random.normal(key1, shape=(1000,)),
    "y": jax.random.normal(key2, shape=(1000,)),
}
sample = build_sampler(log_prob)
results, stats = sample(key3, coords, steps=10_000)
```
