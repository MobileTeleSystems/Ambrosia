import os

import pandas as pd
import pytest

from ambrosia.preprocessing import Preprocessor

store_path: str = "tests/configs/preprocessor_config.json"


@pytest.mark.smoke()
def test_init(data_nonlin_var):
    """
    Instantiation of preprocessor class
    """
    preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    preprocessor.data()
    preprocessor.data(copy=True)


@pytest.mark.smoke()
def test_cuped_sequential(data_nonlin_var):
    """
    Test sequential cuped + robust
    """
    preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    transformed: pd.DataFrame = (
        preprocessor.robust("target", alpha=0.005)
        .cuped("target", "feature_1", transformed_name="target_1")
        .cuped("target_1", "feature_2", transformed_name="target_2")
        .cuped("target_2", "feature_3", transformed_name="target_3")
        .data()
    )


@pytest.mark.smoke()
def test_full_sequential(data_nonlin_var):
    """
    Test available transformations sequentially.
    """
    preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    (
        preprocessor.robust("feature_1", alpha=0.01, tail="right")
        .iqr(["feature_2", "feature_3"])
        .iqr(["feature_1"])
        .log("feature_1")
        .boxcox(["feature_2", "feature_3"])
        .cuped("target", "feature_3", transformed_name="target_cuped")
        .multicuped("target", ["feature_1", "feature_2"], transformed_name="target_multicuped")
    )


@pytest.mark.unit()
def test_load_store_methods(data_nonlin_var):
    """
    Test load and store methods of Preprocessor for the number of transformations.
    """
    preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    (
        preprocessor.robust("feature_1", alpha=0.01, tail="right")
        .iqr(["feature_1"])
        .log("feature_1")
        .boxcox(["feature_2", "feature_3"])
        .cuped("target", "feature_3", transformed_name="target_cuped")
        .multicuped("target", ["feature_1", "feature_2"], transformed_name="target_multicuped")
    )
    preprocessor.store_transformations(store_path)
    loaded_preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    loaded_preprocessor.load_transformations(store_path)
    os.remove(store_path)
    print(preprocessor.transformations())
    for transformer, loaded_transformer in zip(preprocessor.transformations(), loaded_preprocessor.transformations()):
        assert transformer.get_params_dict() == loaded_transformer.get_params_dict()


@pytest.mark.unit()
def test_transform_from_config(data_nonlin_var):
    """
    Test load and store methods of Preprocessor for the number of transformations.
    """
    preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    transformed: pd.DataFrame = (
        preprocessor.robust("feature_1", alpha=0.01, tail="right")
        .iqr(["feature_2", "feature_3"])
        .iqr(["feature_1"])
        .log("feature_1")
        .boxcox(["feature_2", "feature_3"])
        .cuped("target", "feature_3", transformed_name="target_cuped_1")
        .cuped("target", "feature_2", transformed_name="target_cuped_2")
        .multicuped("target", ["feature_1", "feature_2"], transformed_name="target_multicuped")
        .data()
    )
    preprocessor.store_transformations(store_path)
    loaded_preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    transformed_by_config: pd.DataFrame = loaded_preprocessor.transform_from_config(store_path)
    os.remove(store_path)
    assert (transformed == transformed_by_config).all(None)
