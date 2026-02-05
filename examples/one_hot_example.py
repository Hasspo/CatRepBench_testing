

from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_openml

from genbench.data.schema import TabularSchema
from genbench.data.datamodule import TabularDataModule
from genbench.data.splits import SplitConfigKFold, SplitConfigHoldout

from genbench.transforms.pipeline import TransformPipeline
from genbench.transforms.missing import DropMissingRows
from genbench.transforms.continuous import ContinuousStandardScaler
from genbench.transforms.categorical import CategoricalRepresentationTransform


def load_adult_from_sklearn() -> pd.DataFrame:
    # Adult dataset is a classic mixed-type tabular dataset (categorical + numeric)
    bunch = fetch_openml(name="adult", version=2, as_frame=True)
    df = bunch.frame.copy()

    # In OpenML "adult" target column is typically named "class"
    # Ensure it's present:
    if "class" not in df.columns:
        raise RuntimeError(f"Unexpected columns: {df.columns.tolist()[:20]} ...")

    return df


def main() -> None:
    # 1) Download dataset from sklearn (OpenML)
    df = load_adult_from_sklearn()
    print("Downloaded df shape:", df.shape)
    print("Columns:", df.columns.tolist()[:12], "...")

    target_col = "class"

    # 2) Infer schema automatically (mixed-type)
    #    - object/category/bool -> categorical
    #    - int/integer-like -> discrete (depending on heuristics)
    #    - float -> continuous
    schema = TabularSchema.infer_from_dataframe(
        df,
        target_col=target_col,
        id_col=None,
        # heuristics knobs (optional):
        treat_bool_as_categorical=True,
        discrete_max_unique=50,
    )
    print("\nInferred schema:")
    print("  continuous:", schema.continuous_cols)
    print("  discrete  :", schema.discrete_cols)
    print("  categorical:", schema.categorical_cols)

    # 3) Build preprocessing pipeline
    transforms = TransformPipeline(
        transforms=[
            DropMissingRows(),            # global safe step (no fit required)
            ContinuousStandardScaler(),   # fit on TRAIN only per split
            CategoricalRepresentationTransform(
                representation_name="one_hot_representation",
                representation_kwargs={
                    "include_unk": True,
                    "drop_original_categoricals": True,
                },
            ),
        ]
    )

    # 4) Create DataModule (applies global missing handling before splitting)
    dm = TabularDataModule(
        df=df,
        schema=schema,
        transforms=transforms,
        reset_index=True,
    )
    print("\nAfter global cleaning:")
    print("  n_samples:", dm.n_samples)

