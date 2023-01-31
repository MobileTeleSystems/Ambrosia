import os
from typing import List

import numpy as np
import pytest

from ambrosia.preprocessing import BoxCoxTransformer, LogTransformer

STORAGE_PATHS = {
    "boxcox": "tests/configs/boxcox_config.json",
    "logarithm": "tests/configs/log_config.json",
}


@pytest.mark.smoke
def test_boxcox_constructor():
    """
    Check BoxCoxTransformer constructor works.
    """
    boxcox = BoxCoxTransformer()


@pytest.mark.unit
@pytest.mark.parametrize(
    "column_names, transf_name",
    [
        ("feature_1", "boxcox_feature_1"),
        (["feature_2"], "boxcox_feature_2"),
        (["feature_3", "feature_2", "feature_1"], "boxcox_feature_3"),
    ],
)
def test_boxcox(column_names, transf_name, data_nonlin_var, robust_moments):
    """
    Test BoxCoxTransformer transformations via resulted moments.
    """
    robust = BoxCoxTransformer()
    transformed_data = robust.fit_transform(data_nonlin_var, column_names, inplace=False)
    if isinstance(column_names, List):
        column_names: str = column_names[0]
    col_mean: float = transformed_data[column_names].mean()
    col_std: float = transformed_data[column_names].std()
    moments: np.ndarray = robust_moments.loc[transf_name].values
    assert np.isclose(np.array([col_mean, col_std]), moments, atol=0.001).all()


@pytest.mark.unit
def test_boxcox_load_store(data_nonlin_var, robust_moments):
    """
    Test BoxCoxTransformer save and load methods.
    """
    store_path = STORAGE_PATHS["boxcox"]
    boxcox = BoxCoxTransformer()
    boxcox.fit(data_nonlin_var, ["feature_2", "feature_1"])
    boxcox.store_params(store_path)
    loaded_boxcox = BoxCoxTransformer()
    loaded_boxcox.load_params(store_path)
    os.remove(store_path)
    transformed_data = loaded_boxcox.transform(data_nonlin_var, inplace=False)
    col_mean, col_std = transformed_data["feature_2"].mean(), transformed_data["feature_2"].std()
    moments: np.ndarray = robust_moments.loc["boxcox_feature_2"].values
    assert np.isclose(np.array([col_mean, col_std]), moments, atol=0.001).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "column_names",
    [
        ("feature_1"),
        (["feature_2"]),
        (["feature_3", "feature_2", "feature_1"]),
    ],
)
def test_boxcox_inverse(column_names, data_nonlin_var):
    """
    Test BoxCoxTransformer forward and inverse transformations.
    """
    robust = BoxCoxTransformer()
    robust.fit(data_nonlin_var, column_names)
    transformed_data = robust.transform(data_nonlin_var, inplace=False)
    transformed_data = robust.inverse_transform(transformed_data, inplace=False)
    assert np.isclose(transformed_data.values, data_nonlin_var.values, atol=0.000001).all()


@pytest.mark.smoke
def test_log_constructor():
    """
    Check LogTransformer constructor works.
    """
    log = LogTransformer()


@pytest.mark.unit
@pytest.mark.parametrize(
    "column_names, transf_name",
    [
        ("feature_1", "log_feature_1"),
        (["feature_2"], "log_feature_2"),
        (["feature_3", "feature_2", "feature_1"], "log_feature_3"),
    ],
)
def test_logarithm(column_names, transf_name, data_nonlin_var, robust_moments):
    """
    Test LogTransformer forward and inverse transformations.
    """
    robust = LogTransformer()
    transformed_data = robust.fit_transform(data_nonlin_var, column_names, inplace=False)
    if isinstance(column_names, List):
        column_names: str = column_names[0]
    col_mean: float = transformed_data[column_names].mean()
    col_std: float = transformed_data[column_names].std()
    moments: np.ndarray = robust_moments.loc[transf_name].values
    assert np.isclose(np.array([col_mean, col_std]), moments, atol=0.00001).all()


@pytest.mark.unit
def test_logarithm_load_store(data_nonlin_var, robust_moments):
    """
    Test LogTransformer save and load methods.
    """
    store_path = STORAGE_PATHS["logarithm"]
    log = LogTransformer()
    log.fit(data_nonlin_var, ["feature_2", "feature_1"])
    log.store_params(store_path)
    loaded_log = LogTransformer()
    loaded_log.load_params(store_path)
    os.remove(store_path)
    transformed_data = loaded_log.transform(data_nonlin_var, inplace=False)
    col_mean, col_std = transformed_data["feature_2"].mean(), transformed_data["feature_2"].std()
    moments: np.ndarray = robust_moments.loc["log_feature_2"].values
    assert np.isclose(np.array([col_mean, col_std]), moments, atol=0.001).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "column_names",
    [
        ("feature_1"),
        (["feature_2"]),
        (["feature_3", "feature_2", "feature_1"]),
    ],
)
def test_logarithm_inverse(column_names, data_nonlin_var):
    """
    Test LogTransformer transformations via resulted moments.
    """
    robust = LogTransformer()
    robust.fit(data_nonlin_var, column_names)
    transformed_data = robust.transform(data_nonlin_var, inplace=False)
    transformed_data = robust.inverse_transform(transformed_data, inplace=False)
    assert np.isclose(transformed_data.values, data_nonlin_var.values, atol=0.000001).all()
