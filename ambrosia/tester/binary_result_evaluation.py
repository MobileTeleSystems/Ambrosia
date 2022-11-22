#  Copyright 2022 MTS (Mobile Telesystems)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Methods for calculating test results for binary metrics.

Module contains functions that help to deal
with test results evaluation of binary metrics.

Mainly these functions are used in `Tester` core class.
"""
from typing import Tuple

import numpy as np
import scipy.stats as sps

import ambrosia.tools.bin_intervals as bin_pkg
import ambrosia.tools.pvalue_tools as pvalue_pkg
from ambrosia import types


def binary_absolute_result(
    group_a: types.GroupType, group_b: types.GroupType, alpha: np.ndarray, **kwargs
) -> types._SubResultType:
    """
    Function to calculate absolute test result for binary metrics.

    Parameters
    ----------
    group_a : types.GroupType
        Array containing the binary metrics for group A.
    group_b : types.GroupType
        Array containing the binary metrics for group B.
    alpha: np.ndarray
        Array of the first type errors.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    result_dict : _SubResultType
        Dict with absolute test result, computed for binary metrics.
    """
    success_a: int = group_a.sum()
    success_b: int = group_b.sum()
    trials_a: int = len(group_a)
    trials_b: int = len(group_b)
    pvalue: float = bin_pkg.BinomTwoSampleCI.calculate_pvalue(
        a_success=success_a, b_success=success_b, a_trials=trials_a, b_trials=trials_b, **kwargs
    )
    point_effect: float = np.mean(group_b) - np.mean(group_a)
    conf_intervals = bin_pkg.BinomTwoSampleCI.confidence_interval(
        a_success=success_a,
        b_success=success_b,
        a_trials=trials_a,
        b_trials=trials_b,
        confidence_level=1 - alpha,
        **kwargs,
    )
    conf_intervals = list(zip(conf_intervals[0], conf_intervals[1]))
    return {
        "first_type_error": alpha,
        "pvalue": pvalue,
        "effect": point_effect,
        "confidence_interval": conf_intervals,
    }


def binary_relative_confidence_interval(
    confidence_level: float, group_a: types.GroupType, group_b: types.GroupType, alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Function for constructing a relative confidence intervals.

    Uses log-tranformation https://en.wikipedia.org/wiki/Relative_risk.

    Parameters
    ----------
    confidence_level : np.ndarray
        Array of nominal coverages of confidence intervals.
    group_a : types.GroupType
        Array containing the binary metrics for group A.
    group_b : types.GroupType
        Array containing the binary metrics for group B.

    Returns
    -------
    intervals_bounds : Tuple[np.ndarray, np.ndarray]
        Two arrays containing left and right bounds of binary
        relative confidence intervals.
    """
    pvalue_pkg.check_alternative(alternative)
    confidence_level = 1 - pvalue_pkg.corrected_alpha(1 - confidence_level, alternative)
    p_a: float = np.mean(group_a)
    p_b: float = np.mean(group_b)
    estimation_fraction: float = p_b / p_a
    a_size: int = len(group_a)
    b_size: int = len(group_b)
    se_rr: float = np.sqrt((1 - p_a) / (p_a * a_size) + (1 - p_b) / (p_b * b_size))
    quantiles: np.ndarray = sps.norm.ppf((1 + confidence_level) / 2)
    left: np.ndarray = np.log(estimation_fraction) - quantiles * se_rr
    right: np.ndarray = np.log(estimation_fraction) + quantiles * se_rr
    return pvalue_pkg.choose_from_bounds(np.exp(left) - 1, np.exp(right) - 1, alternative)


def binary_relative_result(
    group_a: types.GroupType, group_b: types.GroupType, alpha: np.ndarray, alternative: str = "two-sided"
) -> types._SubResultType:
    """
    Function calculates relative result for binary metrics
    Using log-tranformation https://en.wikipedia.org/wiki/Relative_risk
    """
    point_effect = np.mean(group_b) / np.mean(group_a) - 1
    function_interval = binary_relative_confidence_interval
    intervals = function_interval(1 - alpha, group_a, group_b, alternative=alternative)
    confidence_intervals = [(left, right) for left, right in zip(*intervals)]
    pvalue: float = pvalue_pkg.calculate_pvalue_by_interval(
        function_interval, 0, group_a=group_a, group_b=group_b, alternative=alternative
    )
    return {
        "first_type_error": alpha,
        "pvalue": pvalue,
        "effect": point_effect,
        "confidence_interval": confidence_intervals,
    }
