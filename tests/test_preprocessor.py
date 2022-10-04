import pytest

from ambrozia.preprocessing import Preprocessor


@pytest.mark.smoke()
def test_init(data_notlin_var):
    """
    Instantiation of preprocessor class
    """
    transformer = Preprocessor(data_notlin_var, verbose=False)
    transformer.data()
    transformer.data(copy=True)


@pytest.mark.smoke()
def test_cuped_sequential(data_notlin_var):
    """
    Test sequential cuped + robust
    """
    transformer = Preprocessor(data_notlin_var, verbose=False)
    transformed = (
        transformer.robust("target", alpha=0.005)
        .cuped("target", "feature_1", name="target_1")
        .cuped("target_1", "feature_2", name="target_2")
        .cuped("target_2", "feature_3", name="target_3")
        .data()
    )
