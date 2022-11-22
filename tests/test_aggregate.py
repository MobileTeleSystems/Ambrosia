import pytest

from ambrosia.preprocessing import AggregatePreprocessor


@pytest.mark.smoke()
def test_inst():
    transformer = AggregatePreprocessor()


@pytest.mark.unit()
def test_aggregation_by_week(data_for_agg):
    """
    Check, that aggregation decrease amount of rows by 7 times
    """
    transformer = AggregatePreprocessor()
    res = transformer.run(
        data_for_agg, groupby_columns="id", real_cols=["watched", "sessions"], categorial_cols=["gender", "platform"]
    )
    assert res.shape[0] * 7 == data_for_agg.shape[0]
