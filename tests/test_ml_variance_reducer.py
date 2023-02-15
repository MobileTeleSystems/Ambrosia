import os

import numpy as np
import pandas as pd
import pytest

from ambrosia.preprocessing import MLVarianceReducer

STORAGE_PATHS = {
    "config_store_path": "tests/configs/mlvar_reducer_config.json",
    "model_store_path": "tests/configs/mlvar_reducer_model.pkl",
}


@pytest.mark.smoke
def test_instance():
    """
    Check that simple instance works
    """
    transf = MLVarianceReducer(verbose=False)


@pytest.mark.unit
@pytest.mark.parametrize(
    "columns",
    [
        ["feature_1"],
        ["feature_2"],
        ["feature_3"],
        ["feature_1", "feature_2"],
        ["feature_1", "feature_3"],
        ["feature_2", "feature_3"],
    ],
)
def test_ml_reduce_variance(data_nonlin_var, columns):
    """
    Test, that class reduce variance
    """
    transformer = MLVarianceReducer(verbose=False)
    result: pd.DataFrame = transformer.fit_transform(data_nonlin_var, "target", columns, "target_hat")
    var_y: float = np.var(result.target)
    var_hat: float = np.var(result.target_hat)
    assert var_y >= var_hat


@pytest.mark.unit
def test_store_load_catboost(data_nonlin_var):
    """
    Test store/load parameters for boosting model
    """
    columns = ["feature_1", "feature_2"]
    transformer = MLVarianceReducer(model="boosting", verbose=False)
    transformer.fit(data_nonlin_var, "target", columns, "target_hat")
    transformer.store_params(STORAGE_PATHS["config_store_path"], STORAGE_PATHS["model_store_path"])
    other_transformer = MLVarianceReducer(verbose=False)
    other_transformer.load_params(STORAGE_PATHS["config_store_path"], STORAGE_PATHS["model_store_path"])
    os.remove(STORAGE_PATHS["config_store_path"])
    os.remove(STORAGE_PATHS["model_store_path"])
    data1: pd.DataFrame = transformer.transform(data_nonlin_var, inplace=False)
    data2: pd.DataFrame = other_transformer.transform(data_nonlin_var, inplace=False)
    assert np.allclose(data1.target_hat, data2.target_hat, atol=1e-3)
