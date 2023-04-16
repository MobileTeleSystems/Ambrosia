import os

import numpy as np
import pandas as pd
import pytest

from ambrosia.preprocessing import Cuped, MultiCuped

store_path: str = "tests/configs/cuped_config.json"


@pytest.mark.smoke
def test_cuped_instance():
    """
    Check that Cuped constructor works.
    """
    cuped = Cuped()


@pytest.mark.smoke
def test_multicuped_instance():
    """
    Check that MultiCuped constructor works.
    """
    cuped = MultiCuped()


@pytest.mark.unit
@pytest.mark.parametrize("covariate_column", ["feature_1", "feature_2", "feature_3"])
def test_cuped_decrease_var(covariate_column, data_variance_lin):
    """
    Check that CUPED technique decreases variance.
    """
    transformer = Cuped(verbose=False)
    result: pd.DataFrame = transformer.fit_transform(
        data_variance_lin, target_column="target", covariate_column=covariate_column, transformed_name="target_hat"
    )
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
    Check that Multi CUPED decreases variance.
    """
    transformer = MultiCuped(verbose=False)
    result: pd.DataFrame = transformer.fit_transform(
        data_variance_lin, target_column="target", covariate_columns=columns, transformed_name="target_hat"
    )
    var_y: float = np.var(result.target)
    var_hat: float = np.var(result.target_hat)
    assert var_y >= var_hat


@pytest.mark.unit
@pytest.mark.parametrize("column", ["feature_1", "feature_2", "feature_3"])
def test_equal_multi_simple(column, data_variance_lin):
    """
    Check that Multi CUPED result based on single covariate column is equal to the simple CUPED.
    """
    transformer_cuped = Cuped(verbose=False)
    transformer_multi = MultiCuped(verbose=False)
    transformer_cuped.fit_transform(data_variance_lin, "target", column)
    transformer_multi.fit_transform(data_variance_lin, "target", column)
    assert np.isclose(
        transformer_cuped.params[transformer_cuped.THETA_NAME],
        transformer_multi.params[transformer_cuped.THETA_NAME][0][0],
        atol=0.0001,
    )


@pytest.mark.unit
@pytest.mark.parametrize("Model, factor", [(Cuped, "feature_1"), (MultiCuped, ["feature_1"])])
def test_load_store_params(Model, factor, data_variance_lin):
    """
    Test load and store functions for Cuped and MultiCuped functions.
    """
    cuped = Model(verbose=False)
    transformed: pd.DataFrame = cuped.fit_transform(data_variance_lin, "target", factor)
    cuped.store_params(store_path)
    loaded_cuped = Model(verbose=True)
    loaded_cuped.load_params(store_path)
    os.remove(store_path)
    loaded_transformed: pd.DataFrame = loaded_cuped.transform(data_variance_lin)
    assert (transformed == loaded_transformed).all(None)
