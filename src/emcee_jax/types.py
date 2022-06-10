from typing import Any, Callable, Dict, NamedTuple, Tuple, Union

from jax.random import KeyArray

PyTree = Any
Array = Any
Params = Union[PyTree, Array]
Deterministics = Union[PyTree, Array]
MoveState = Any
SamplerStats = Dict[str, Any]


class WalkerState(NamedTuple):
    coordinates: Params
    deterministics: Deterministics
    log_probability: Array


UnravelFn = Callable[[Array], Params]

InitFn = Callable[[Array], MoveState]
StepFn = Callable[
    [MoveState, KeyArray, WalkerState], Tuple[SamplerStats, WalkerState]
]
MoveFn = Callable[..., Tuple[InitFn, StepFn]]

LogProbFn = Callable[..., Array]
WrappedLogProbFn = Callable[[Array], Array]


class Trace(NamedTuple):
    final_state: WalkerState
    samples: WalkerState
    stats: SamplerStats
