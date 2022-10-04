from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from ambrozia.tester import Tester, test
from ambrozia.tools.stat_criteria import TtestIndCriterion, TtestRelCriterion


@pytest.mark.smoke
def test_instance():
    """
    Check that simple instance without args work
    """
    tester = Tester()


@pytest.mark.smoke
def test_constructors(results_ltv_retention_conversions):
    """
    Test different constructors
    """
    # Only table
    tester = Tester(dataframe=results_ltv_retention_conversions, column_groups="group")
    # Use metrics
    tester = Tester(
        dataframe=results_ltv_retention_conversions, metrics=["retention", "conversions"], column_groups="group"
    )
    tester = Tester(metrics="ltv")


@pytest.mark.smoke
@pytest.mark.parametrize("effect_type", ["relative", "absolute"])
@pytest.mark.parametrize("as_table", [False, True])
def test_correct_type(effect_type, as_table, tester_on_ltv_retention):
    """
    Check, that method run is callable and return correct type
    """
    types = [List, pd.DataFrame]
    assert isinstance(tester_on_ltv_retention.run(effect_type, as_table=as_table), types[as_table])


@pytest.mark.unit
@pytest.mark.parametrize("effect_type", ["relative", "absolute"])
@pytest.mark.parametrize("method", ["theory", "empiric"])
def test_every_type_run(effect_type, method, tester_on_ltv_retention):
    """
    Use cortesian product of all params to check, that all posible combinations are working
    """
    result = tester_on_ltv_retention.run(effect_type=effect_type, method=method, as_table=False)
    assert result[0]["effect"] > 0
    assert result[2]["effect"] < 0


def check_pvalue_for_interval(interval: Tuple, pvalue: float, alpha: float, check_value: float = 0) -> bool:
    """
    Check, that check_value in interval <=> pvalue <= alpha
    """
    if interval[0] <= check_value and interval[1] >= check_value and pvalue <= alpha:
        return False
    elif (interval[0] > check_value or interval[1] < check_value) and pvalue > alpha:
        return False
    else:
        return True


@pytest.mark.unit
@pytest.mark.parametrize("method", ["theory", "binary", "empiric"])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("metrics", ["retention", "conversions"])
@pytest.mark.parametrize("criterion", ["ttest", "ttest_rel"])
def test_coinf_interval_absolute(method, alpha, metrics, criterion, tester_on_ltv_retention):
    """
    Test that confidence interval contains 0 <=> pvalue < alpha
    """
    result = tester_on_ltv_retention.run(
        "absolute", method=method, criterion=criterion, first_errors=alpha, metrics=metrics, as_table=False
    )[0]
    interval = result["confidence_interval"]
    pvalue = result["pvalue"]
    assert check_pvalue_for_interval(interval, pvalue, alpha, 0)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["theory", "empiric"])
@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.1])
@pytest.mark.parametrize("metrics", ["retention", "conversions", "ltv"])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_coinf_interval_relative(method, alpha, metrics, alternative, tester_on_ltv_retention):
    """
    Test that confidence interval contains 1 <=> pvalue <= alpha
    """
    result = tester_on_ltv_retention.run(
        "relative",
        method=method,
        first_errors=alpha,
        metrics=metrics,
        as_table=False,
        alternative=alternative,
    )[0]
    interval = result["confidence_interval"]
    pvalue = result["pvalue"]
    assert check_pvalue_for_interval(interval, pvalue, alpha, 0)


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.1])
@pytest.mark.parametrize("metrics", ["retention", "conversions"])
@pytest.mark.parametrize("interval_type", ["wald", "yule", "newcombe", "yule_modif", "jeffrey", "recenter"])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_coinf_interval_bin_abs(alpha, metrics, interval_type, alternative, tester_on_ltv_retention):
    """
    Test that confidence interval contains 0 <=> pvalue <= alpha
    For binary method and different interval approaches
    For absolute effect
    """
    result = tester_on_ltv_retention.run(
        "absolute",
        method="binary",
        first_errors=alpha,
        metrics=metrics,
        interval_type=interval_type,
        alternative=alternative,
        as_table=False,
    )[0]
    interval = result["confidence_interval"]
    pvalue = result["pvalue"]
    assert check_pvalue_for_interval(interval, pvalue, alpha)


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.1])
@pytest.mark.parametrize("metrics", ["retention", "conversions"])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_coinf_interval_bin_rel(alpha, metrics, alternative, tester_on_ltv_retention):
    """
    Test that confidence interval contains 0 <=> pvalue <= alpha
    For binary method and different interval approaches
    For relative effect
    """
    result = tester_on_ltv_retention.run(
        "relative",
        method="binary",
        first_errors=alpha,
        metrics=metrics,
        alternative=alternative,
        as_table=False,
    )[0]
    interval = result["confidence_interval"]
    pvalue = result["pvalue"]
    assert check_pvalue_for_interval(interval, pvalue, alpha)


@pytest.mark.unit
@pytest.mark.parametrize("criterion", ["ttest", "ttest_rel"])
@pytest.mark.parametrize("effect_type", ["absolute", "relative"])
@pytest.mark.parametrize("method", ["theory", "binary"])
@pytest.mark.parametrize("alpha", [0.01, 0.05])
@pytest.mark.parametrize("metrics", ["retention", "conversions"])
def test_standalone_test_function(
    criterion, effect_type, method, alpha, metrics, tester_on_ltv_retention, results_ltv_retention_conversions
):
    """
    Test standalone test function gives same result as Tester class.
    """
    if method == "binary" and effect_type == "relative":
        return

    function_result = test(
        effect_type,
        method,
        dataframe=results_ltv_retention_conversions,
        metrics=metrics,
        criterion=criterion,
        column_groups="group",
        first_errors=alpha,
        as_table=False,
    )
    class_result = tester_on_ltv_retention.run(
        effect_type, method, metrics=metrics, first_errors=alpha, criterion=criterion, as_table=False
    )
    assert function_result == class_result


@pytest.mark.parametrize("effect_type", ["absolute", "relative"])
def test_criteria_ttest_different(effect_type):
    """
    Test criteria classes
    """
    group_a = np.array([1, 2, 3, 4, 5])
    group_b = np.array([2, 3, 4, 7, 10])
    ttest_ind = TtestIndCriterion()
    ttest_rel = TtestRelCriterion()
    assert ttest_ind.calculate_pvalue(group_a, group_b, effect_type=effect_type) != ttest_rel.calculate_pvalue(
        group_a, group_b, effect_type=effect_type
    )
    assert ttest_ind.calculate_conf_interval(
        group_a, group_b, effect_type=effect_type
    ) != ttest_rel.calculate_conf_interval(group_a, group_b, effect_type=effect_type)


@pytest.mark.parametrize("criterion", ["ttest", "ttest_rel", "mw", "wilcoxon"])
def test_kwargs_passing_theory(criterion, tester_on_ltv_retention):
    """
    Test passing key word argument to run method
    """
    old_pvalue = tester_on_ltv_retention.run(criterion=criterion, as_table=False)[2]["pvalue"]
    alternative_pvalue = tester_on_ltv_retention.run(criterion=criterion, as_table=False, alternative="greater")[2][
        "pvalue"
    ]
    assert old_pvalue >= alternative_pvalue


@pytest.mark.parametrize("interval_type", ["yule", "yule_modif", "newcombe", "jeffrey", "agresti"])
def test_kwargs_passing_binary(interval_type, tester_on_ltv_retention):
    """
    Test passing key word argument to run method for binary metrics
    """
    wald_interval = tester_on_ltv_retention.run("absolute", "binary", metrics="retention", as_table=False)[0][
        "confidence_interval"
    ]
    other_interval = tester_on_ltv_retention.run(
        "absolute", "binary", metrics="retention", interval_type=interval_type, as_table=False
    )[0]["confidence_interval"]
    assert wald_interval != other_interval


def get_ci_pvalue(tester_on_ltv_retention, alternative: str, idx: int = 0, **run_kwargs):
    """
    Get pvalue and confidence intervals for alternative
    """
    res_table = tester_on_ltv_retention.run(alternative=alternative, **run_kwargs)
    pvalue = res_table[idx]["pvalue"]
    confidence_interval = res_table[idx]["confidence_interval"]
    return pvalue, confidence_interval


def calc_intervals_pvalue(tester_on_ltv_retention, idx: int = 0, **run_kwargs) -> bool:
    """
    Calc pvalue and intervals
    """
    pvalue_center, int_center = get_ci_pvalue(tester_on_ltv_retention, "two-sided", idx, **run_kwargs)
    pvalue_gr, int_gr = get_ci_pvalue(tester_on_ltv_retention, "greater", idx, **run_kwargs)
    pvalue_less, int_less = get_ci_pvalue(tester_on_ltv_retention, "less", idx, **run_kwargs)
    return pvalue_center, int_center, pvalue_gr, int_gr, pvalue_less, int_less


def check_bound_intervals(int_center, int_less, int_gr, left_bound: float = -np.inf, right_bound: float = np.inf):
    """
    Check bound of intervals for different alternatives
    """
    assert int_gr[0] == left_bound
    assert int_less[1] == right_bound
    assert int_gr[1] < int_center[1]
    assert int_less[0] > int_center[0]


@pytest.mark.parametrize("effect_type", ["absolute"])
@pytest.mark.parametrize("interval_type", ["wald", "yule", "newcombe", "yule_modif", "jeffrey", "recenter"])
def test_alternative_change_binary(effect_type, interval_type, tester_on_ltv_retention):
    """
    Test changes in pvalue and confidence interval for binary method
    """
    pvalue_center, int_center, pvalue_gr, int_gr, pvalue_less, int_less = calc_intervals_pvalue(
        tester_on_ltv_retention, effect_type=effect_type, method="binary", metrics="retention", as_table=False
    )
    # mean retention A - 0.303
    # mean retention B - 0.399
    assert pvalue_less < pvalue_gr
    assert pvalue_center < pvalue_gr
    # Check intervals
    check_bound_intervals(int_center, int_less, int_gr, -1, 1)


@pytest.mark.parametrize("effect_type", ["absolute", "relative"])
@pytest.mark.parametrize("criterion", ["ttest", "ttest_rel"])
def test_alternative_change_th(effect_type, criterion, tester_on_ltv_retention):
    """
    Test changes in pvalue and confidence interval for theory method
    """
    pvalue_center, int_center, pvalue_gr, int_gr, pvalue_less, int_less = calc_intervals_pvalue(
        tester_on_ltv_retention,
        effect_type=effect_type,
        criterion=criterion,
        method="theory",
        metrics="ltv",
        as_table=False,
    )
    # Mean(group_a) > Mean(group_b) in this table
    assert pvalue_less > pvalue_gr
    assert pvalue_center > pvalue_gr
    # Check intervals
    check_bound_intervals(int_center, int_less, int_gr)
