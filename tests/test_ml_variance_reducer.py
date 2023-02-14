import os

import numpy as np
import pandas as pd
import pytest

from ambrosia.preprocessing import MLVarianceReducer


@pytest.mark.smoke
def test_instance(data_nonlin_var):
    """
    Check that simple instance works
    """
    transf = MLVarianceReducer(data_nonlin_var, "target", verbose=False)


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
    transformer = MLVarianceReducer(data_nonlin_var, "target", verbose=False)
    result: pd.DataFrame = transformer.fit_transform(columns, name="target_hat")
    var_y: float = np.var(result.target)
    var_hat: float = np.var(result.target_hat)
    assert var_y >= var_hat


@pytest.mark.unit
def test_store_load_catboost(data_nonlin_var):
    """
    Test store/load parameters for boosting
    """
    columns = ["feature_1", "feature_2"]
    transformer = MLVarianceReducer(data_nonlin_var, "target", verbose=False)
    transformer.fit(columns)
    transformer.store_params("tests/params.json")
    other_transformer = MLVarianceReducer(data_nonlin_var, "target", verbose=False)
    other_transformer.load_params("tests/params.json")
    os.remove("tests/params.json")
    data1: pd.DataFrame = transformer.transform(columns, inplace=False, name="target_hat")
    data2: pd.DataFrame = other_transformer.transform(columns, inplace=False, name="target_hat")
    assert np.allclose(data1.target_hat, data2.target_hat, atol=1e-3)
