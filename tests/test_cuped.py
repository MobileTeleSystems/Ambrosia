import os

import numpy as np
import pandas as pd
import pytest

from ambrosia.preprocessing import Cuped, MultiCuped


@pytest.mark.smoke
def test_instance(data_variance_lin):
    """
    Check that simple instance without args work
    """
    transf = Cuped(data_variance_lin, "target")


@pytest.mark.unit
@pytest.mark.parametrize("covariate_column", ["feature_1", "feature_2", "feature_3"])
def test_cuped_decrease_var(covariate_column, data_variance_lin):
    transformer = Cuped(data_variance_lin, target_column="target", verbose=False)
    result: pd.DataFrame = transformer.fit_transform(covariate_column, name="target_hat")
    var_y: float = np.var(result.target)
    var_hat: float = np.var(result.target_hat)
    assert var_y >= var_hat


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
def test_multi_cuped(columns, data_variance_lin):
    """
    Check Multi CUPED decrease Variance
    """
    transformer = MultiCuped(data_variance_lin, target_column="target", verbose=False)
    result: pd.DataFrame = transformer.fit_transform(columns, name="target_hat")
    var_y: float = np.var(result.target)
    var_hat: float = np.var(result.target_hat)
    assert var_y >= var_hat


@pytest.mark.unit
@pytest.mark.parametrize("column", ["feature_1", "feature_2", "feature_3"])
def test_equal_multi_simple(column, data_variance_lin):
    """
    Multi_1 = Simple
    """
    transformer_cuped = Cuped(data_variance_lin, target_column="target", verbose=False)
    transformer_multi = MultiCuped(data_variance_lin, target_column="target", verbose=False)
    transformer_cuped.fit_transform(column)
    transformer_multi.fit_transform([column])
    assert np.isclose(transformer_cuped.theta, transformer_multi.theta[0][0], atol=0.0001)


@pytest.mark.unit
@pytest.mark.parametrize("Model, factor", [(Cuped, "feature_1"), (MultiCuped, ["feature_1"])])
def test_load_store_params(Model, factor, data_variance_lin):
    """
    Test load and store functions for cuped and multi cuped functions
    """
    cuped = Model(data_variance_lin, "target")
    cuped.fit(factor)
    cuped.store_params("params.json")
    params = cuped.get_params_dict()
    cuped.load_params("params.json")
    os.remove("params.json")
    assert params == cuped.get_params_dict()
