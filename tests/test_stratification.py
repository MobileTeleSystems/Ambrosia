import numpy as np
import pytest

from ambrozia.tools.ab_abstract_component import EmptyStratValue
from ambrozia.tools.stratification import Stratification


@pytest.mark.smoke()
def test_instance():
    """
    Check that simple instance without args work
    """
    strat = Stratification()
    strat = Stratification(threshold=5)


@pytest.mark.smoke()
def test_fit(stratification_table):
    """
    Fit method tests
    """
    strat = Stratification()
    assert not strat.is_trained()
    strat.fit(stratification_table, columns=["gender", "retention"])
    assert strat.dataframe.equals(stratification_table)
    # Test No columns
    strat.fit(stratification_table)
    assert list(strat.strats.keys()) == [EmptyStratValue.NO_STRATIFICATION]
    assert strat.is_trained()


@pytest.mark.unit()
def test_strat_sizes(stratificator):
    """
    Stratification group sizes
    """
    sizes = stratificator.strat_sizes()
    results = {("Female", 0): 112, ("Female", 1): 173, ("Male", 0): 304, ("Male", 1): 411}
    assert sizes == results


@pytest.mark.unit()
@pytest.mark.parametrize(
    "ids, column, answer",
    [
        (np.arange(0, 1000, 43), None, pytest.lazy_fixture("answer_ids_strat")),
        (pytest.lazy_fixture("id_for_b_strat"), "id", pytest.lazy_fixture("answer_ids_strat_column")),
    ],
)
def test_test_ids(ids, column, answer, stratificator):
    """
    Test test group ids with stratification
    """
    assert stratificator.get_test_inds(ids, column) == answer


@pytest.mark.unit()
@pytest.mark.parametrize("group_size", [100, 500, 1000])
def test_groups_size(group_size, stratificator):
    """
    Test group sizes for stratification
    """
    amount_of_strats: int = len(stratificator.strats)
    sizes = stratificator.get_group_sizes(group_size)
    total_size: int = 0
    for _, size in sizes.items():
        total_size += size
    assert total_size - amount_of_strats <= group_size
    assert total_size + amount_of_strats >= group_size
