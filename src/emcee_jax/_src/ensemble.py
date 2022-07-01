from typing import NamedTuple, Tuple, Union

import jax
import jax.linear_util as lu
import numpy as np
from jax.tree_util import tree_leaves

from emcee_jax._src.types import Array, PyTree


class Ensemble(NamedTuple):
    coordinates: PyTree
    deterministics: PyTree
    log_probability: Array

    @classmethod
    def init(
        cls, log_prob_fn: lu.WrappedFun, ensemble: Union["Ensemble", PyTree]
    ) -> "Ensemble":
        if isinstance(ensemble, cls):
            return ensemble
        fn = jax.vmap(log_prob_fn.call_wrapped)
        log_probability, deterministics = fn(ensemble)
        return cls(
            coordinates=ensemble,
            deterministics=deterministics,
            log_probability=log_probability,
        )


def get_ensemble_shape(ensemble: PyTree) -> Tuple[int, int]:
    leaves = tree_leaves(ensemble)
    if not len(leaves):
        raise ValueError("The ensemble is empty")
    if len(leaves) == 1 and leaves[0].ndim <= 1:
        raise ValueError(
            "An ensemble must have at least 2 dimensions; "
            "did you provide just a single walker coordinate?"
        )
    leading, rest = zip(
        *((x.shape[0], int(np.prod(x.shape[1:]))) for x in leaves)
    )
    if any(s != leading[0] for s in leading):
        raise ValueError(
            f"All leaves must have the same leading dimension; got {leading}"
        )
    return leading[0], sum(rest)
