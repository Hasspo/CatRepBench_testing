from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, runtime_checkable

import pandas as pd

from genbench.data.schema import TabularSchema


@dataclass
class RepresentationState:
    """
    Minimal serializable state for a representation.

    NOTE:
      - `params` must be JSON-serializable if you want JSON persistence.
    """
    name: str
    params: Dict[str, Any]


@runtime_checkable
class BaseRepresentation(Protocol):
    """
    Representation = "how to map mixed tabular df into model-friendly df".

    Contract:
      - DataFrame in -> DataFrame out
      - may require fit (build vocab, stats, etc.)
      - may be invertible (decode back to original columns)

    We keep it DataFrame->DataFrame on purpose so it can be used inside TransformPipeline
    via a thin wrapper transform (see transforms/categorical.py).
    """

    name: str

    def requires_fit(self) -> bool:
        ...

    def is_invertible(self) -> bool:
        ...

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "BaseRepresentation":
        ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def get_state(self) -> RepresentationState:
        ...

    @classmethod
    def from_state(cls, state: RepresentationState) -> "BaseRepresentation":
        ...
