# emcee-jax

An experiement.

An example:

```python
import jax
from emcee_jax.sampler import build_sampler

def log_prob(x, a1=100.0, a2=20.0):
    return -(a1 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2) / a2

coords = jax.random.normal(jax.random.PRNGKey(0), shape=(1000, 2))
sample = build_sampler(log_prob)
results, stats = sample(jax.random.PRNGKey(1), coords, steps=10_000)
```
