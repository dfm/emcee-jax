from typing import Any, Callable, Dict, NamedTuple, Tuple

from jax.random import KeyArray

Array = Any
Params = Any
State = Any
Stats = Dict[str, Any]


class Walker(NamedTuple):
    coords: Params
    log_probability: Array


UnravelFn = Callable[[Array], Params]

InitFn = Callable[[Array], State]
StepFn = Callable[[State, KeyArray, Array, Array], Tuple[Stats, Array, Array]]
MoveFn = Callable[..., Tuple[InitFn, StepFn]]

LogProbFn = Callable[..., Array]
WrappedLogProbFn = Callable[[Array], Array]
