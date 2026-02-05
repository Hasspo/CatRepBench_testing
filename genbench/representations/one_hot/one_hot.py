from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.representations.base import RepresentationState


def _safe_str(x: object) -> str:
    # Stable string for categories (handles NaN/None)
    if x is None:
        return "__NONE__"
    if isinstance(x, float) and pd.isna(x):
        return "__NAN__"
    return str(x)


@dataclass
class OneHotRepresentation:
    """
    One-hot representation for categorical columns.

    Output:
      - Keeps continuous + discrete columns as-is.
      - Replaces each categorical column with one-hot columns: "{col}__{category}"

    Options:
      - unk_token: bucket for unseen categories at transform-time
      - include_unk: whether to include an UNK column in output (recommended True for robustness)
      - drop_original_categoricals: drop original categorical columns (True by default)

    Inverse transform:
      - Reconstructs each categorical column by argmax over its one-hot group.
      - If all zeros -> unk_token (or None if include_unk=False).
    """

    name: str = "one_hot_representation"
    unk_token: str = "__UNK__"
    include_unk: bool = True
    drop_original_categoricals: bool = True

    # fitted state
    fitted_: bool = False
    categorical_cols_: List[str] = field(default_factory=list)
    vocab_: Dict[str, List[str]] = field(default_factory=dict)          # col -> list of category strings
    out_cols_: Dict[str, List[str]] = field(default_factory=dict)       # col -> list of one-hot column names

    def requires_fit(self) -> bool:
        return True

    def is_invertible(self) -> bool:
        return True

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "OneHotRepresentation":
        cat_cols = list(schema.categorical_cols)
        self.categorical_cols_ = cat_cols

        vocab: Dict[str, List[str]] = {}
        out_cols: Dict[str, List[str]] = {}

        for c in cat_cols:
            # Build vocab on TRAIN ONLY
            s = df[c].map(_safe_str)
            cats = list(pd.unique(s.dropna()))
            # deterministic ordering
            cats = sorted(cats)

            if self.include_unk and self.unk_token not in cats:
                cats = cats + [self.unk_token]

            vocab[c] = cats
            out_cols[c] = [f"{c}__{cat}" for cat in cats]

        self.vocab_ = vocab
        self.out_cols_ = out_cols
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("OneHotRepresentation must be fitted before transform().")

        out = df.copy()

        for c in self.categorical_cols_:
            if c not in out.columns:
                raise KeyError(f"Categorical column '{c}' not found in DataFrame.")

            s = out[c].map(_safe_str)

            # Map unseen categories -> UNK (or keep as-is if include_unk=False)
            if self.include_unk:
                known = set(self.vocab_[c])
                s = s.where(s.isin(known), other=self.unk_token)

            # Create one-hot columns (0/1 ints)
            cats = self.vocab_[c]
            for cat in cats:
                col_name = f"{c}__{cat}"
                out[col_name] = (s == cat).astype("int64")

            if self.drop_original_categoricals:
                out = out.drop(columns=[c])

        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("OneHotRepresentation must be fitted before inverse_transform().")

        out = df.copy()

        for c in self.categorical_cols_:
            oh_cols = self.out_cols_[c]
            missing = [cc for cc in oh_cols if cc not in out.columns]
            if missing:
                raise KeyError(f"Missing one-hot columns for '{c}': {missing}")

            # argmax across one-hot group
            mat = out[oh_cols].to_numpy()
            idx = mat.argmax(axis=1)

            cats = self.vocab_[c]
            recovered = [cats[i] if 0 <= i < len(cats) else self.unk_token for i in idx]

            out[c] = recovered

            # Drop one-hot columns and keep original categorical
            out = out.drop(columns=oh_cols)

        return out

    def get_state(self) -> RepresentationState:
        return RepresentationState(
            name=self.name,
            params={
                "unk_token": self.unk_token,
                "include_unk": self.include_unk,
                "drop_original_categoricals": self.drop_original_categoricals,
                "fitted": self.fitted_,
                "categorical_cols": self.categorical_cols_,
                "vocab": self.vocab_,
                "out_cols": self.out_cols_,
            },
        )

    @classmethod
    def from_state(cls, state: RepresentationState) -> "OneHotRepresentation":
        obj = cls(
            unk_token=str(state.params.get("unk_token", "__UNK__")),
            include_unk=bool(state.params.get("include_unk", True)),
            drop_original_categoricals=bool(state.params.get("drop_original_categoricals", True)),
        )
        obj.fitted_ = bool(state.params.get("fitted", False))
        obj.categorical_cols_ = list(state.params.get("categorical_cols", []))
        obj.vocab_ = dict(state.params.get("vocab", {}))
        obj.out_cols_ = dict(state.params.get("out_cols", {}))
        return obj
