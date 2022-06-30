from typing import Any, Callable, Dict, Iterable, NamedTuple, Tuple, Union

Array = Any
PyTree = Union[Array, Iterable[Array], Dict[Any, Array], NamedTuple]
SampleStats = Dict[str, Array]


class WalkerState(NamedTuple):
    coordinates: PyTree
    deterministics: PyTree
    log_probability: Array


class FlatWalkerState(NamedTuple):
    coordinates: Array
    deterministics: Array
    log_probability: Array


LogProbFn = Callable[..., Union[Array, Tuple[Array, PyTree]]]
WrappedLogProbFn = Callable[[Array], Tuple[Array, Array]]