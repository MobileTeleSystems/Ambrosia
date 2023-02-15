import os

import pytest

from ambrosia.preprocessing import AggregatePreprocessor


@pytest.mark.smoke()
def test_inst():
    aggregator = AggregatePreprocessor()


@pytest.mark.unit()
def test_aggregation_by_agg_params(data_for_agg):
    """
    Check that aggregation with agg_params dict works.
    """
    aggregator = AggregatePreprocessor()
    groupby_columns = ["id", "gender"]
    agg_params = {
        "watched": "mean",
        "sessions": "median",
        "gender": "last",
        "platform": "first",
    }
    result = aggregator.fit_transform(
        dataframe=data_for_agg,
        groupby_columns=groupby_columns,
        agg_params=agg_params,
    )


@pytest.mark.unit()
def test_agg_params_priority(data_for_agg):
    """
    Check that agg_params has higher priority over default aggregation methods.
    """
    aggregator = AggregatePreprocessor(
        categorial_method="mode",
        real_method="mean",
    )
    groupby_columns = ["id"]
    agg_params = {
        "sessions": "median",
        "gender": "last",
    }
    aggregator.fit(
        dataframe=data_for_agg,
        groupby_columns=groupby_columns,
        agg_params=agg_params,
        real_cols=["sessions"],
        categorial_cols="gender",
    )
    assert aggregator.agg_params == agg_params


@pytest.mark.unit()
def test_aggregation_by_week(data_for_agg):
    """
    Check that table aggregation decrease amount of rows by 7 times (week agg).
    """
    aggregator = AggregatePreprocessor()
    res = aggregator.fit_transform(
        dataframe=data_for_agg,
        groupby_columns="id",
        real_cols=["watched", "sessions"],
        categorial_cols=["gender", "platform"],
    )
    assert res.shape[0] * 7 == data_for_agg.shape[0]


@pytest.mark.unit
def test_aggregate_load_store(data_for_agg):
    """
    Test AggregatePreprocessor save and load methods.
    """
    store_path = "tests/configs/aggregate_config.json"
    aggregator = AggregatePreprocessor()
    aggregator.fit(
        dataframe=data_for_agg,
        groupby_columns="id",
        real_cols=["watched", "sessions"],
        categorial_cols=["gender", "platform"],
    )
    aggregator.store_params(store_path)
    loaded_aggregator = AggregatePreprocessor()
    loaded_aggregator.load_params(store_path)
    os.remove(store_path)
    loaded_aggregator.transform(data_for_agg)
