from typing import Any, Dict, Iterable, NamedTuple, Union

Array = Any
PyTree = Union[Array, Iterable[Array], Dict[Any, Array], NamedTuple]
SampleStats = Dict[str, Array]
Extras = PyTree
