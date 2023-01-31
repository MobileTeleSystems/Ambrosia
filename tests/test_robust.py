import os
from typing import List

import numpy as np
import pytest

from ambrosia.preprocessing import BoxCoxTransformer, IQRPreprocessor, LogTransformer, RobustPreprocessor

STORAGE_PATHS = {
    "robust": "tests/configs/robust_config.json",
    "iqr": "tests/configs/iqr_config.json",
    "boxcox": "tests/configs/boxcox_config.json",
    "logarithm": "tests/configs/log_config.json",
}


@pytest.mark.smoke
def test_robust_constructor():
    """
    Check RobustPreprocessor constructor works.
    """
    robust = RobustPreprocessor(verbose=False)


@pytest.mark.unit
@pytest.mark.parametrize(
    "tail, column_names, alpha, transf_name",
    [
        ("both", "feature_1", 0.05, "robust_both_0.05_feature_1"),
        ("both", ["feature_2"], 0.05, "robust_both_0.05_feature_2"),
        ("both", ["feature_2", "feature_1"], [0.05, 0.05], "robust_both_0.05_feature_2after1"),
        ("left", "feature_1", 0.1, "robust_left_0.1_feature_1"),
        ("right", "feature_1", 0.1, "robust_right_0.1_feature_1"),
    ],
)
def test_robust(tail, column_names, alpha, transf_name, data_nonlin_var, robust_moments):
    """
    Test RobustPreprocessor quantile transformations via resulted moments.
    """
    robust = RobustPreprocessor(verbose=False)
    transformed_data = robust.fit_transform(data_nonlin_var, column_names, alpha, tail, inplace=False)
    if isinstance(column_names, List):
        column_names: str = column_names[0]
    col_mean: float = transformed_data[column_names].mean()
    col_std: float = transformed_data[column_names].std()
    moments: np.ndarray = robust_moments.loc[transf_name].values
    assert np.isclose(np.array([col_mean, col_std]), moments, atol=0.000001).all()


@pytest.mark.unit
def test_robust_load_store(data_nonlin_var, robust_moments):
    """
    Test RobustPreprocessor save and load methods.
    """
    store_path = STORAGE_PATHS["robust"]
    robust = RobustPreprocessor(verbose=False)
    robust.fit(data_nonlin_var, ["feature_2", "feature_1"], 0.05)
    robust.store_params(store_path)
    loaded_robust = RobustPreprocessor(verbose=False)
    loaded_robust.load_params(store_path)
    os.remove(store_path)
    transformed_data = loaded_robust.transform(data_nonlin_var, inplace=False)
    col_mean: float = transformed_data["feature_2"].mean()
    col_std: float = transformed_data["feature_2"].std()
    moments: np.ndarray = robust_moments.loc["robust_both_0.05_feature_2after1"].values
    assert np.isclose(np.array([col_mean, col_std]), moments, atol=0.000001).all()


@pytest.mark.smoke
def test_iqr_constructor():
    """
    Check IQRPreprocessor constructor works.
    """
    iqr = IQRPreprocessor(verbose=False)


@pytest.mark.unit
@pytest.mark.parametrize(
    "column_names, transf_name",
    [
        ("feature_1", "iqr_feature_1"),
        (["feature_2"], "iqr_feature_2"),
        (["feature_2", "feature_1"], "iqr_feature_2after1"),
    ],
)
def test_iqr(column_names, transf_name, data_nonlin_var, robust_moments):
    """
    Test IQRPreprocessor transformations via resulted moments.
    """
    iqr = IQRPreprocessor(verbose=False)
    transformed_data = iqr.fit_transform(data_nonlin_var, column_names, inplace=False)
    if isinstance(column_names, List):
        column_names: str = column_names[0]
    col_mean: float = transformed_data[column_names].mean()
    col_std: float = transformed_data[column_names].std()
    moments: np.ndarray = robust_moments.loc[transf_name].values
    assert np.isclose(np.array([col_mean, col_std]), moments, atol=0.000001).all()


@pytest.mark.unit
def test_iqr_load_store(data_nonlin_var, robust_moments):
    """
    Test IQRPreprocessor save and load methods.
    """
    store_path = STORAGE_PATHS["iqr"]
    iqr = IQRPreprocessor(verbose=False)
    iqr.fit(data_nonlin_var, ["feature_2", "feature_1"])
    iqr.store_params(store_path)
    loaded_iqr = IQRPreprocessor(verbose=False)
    loaded_iqr.load_params(store_path)
    os.remove(store_path)
    transformed_data = loaded_iqr.transform(data_nonlin_var, inplace=False)
    col_mean, col_std = transformed_data["feature_2"].mean(), transformed_data["feature_2"].std()
    moments: np.ndarray = robust_moments.loc["iqr_feature_2after1"].values
    assert np.isclose(np.array([col_mean, col_std]), moments, atol=0.000001).all()


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
