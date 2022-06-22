__all__ = ["apply_accept"]

from functools import partial
from typing import TypeVar

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from emcee_jax.types import Array, PyTree

T = TypeVar("T", PyTree, Array)


def apply_accept(accept: Array, target: T, other: T) -> T:
    accepter = jax.vmap(lambda a, x, y: jnp.where(a, y, x))
    accepter = partial(accepter, accept)
    return tree_map(accepter, target, other)
