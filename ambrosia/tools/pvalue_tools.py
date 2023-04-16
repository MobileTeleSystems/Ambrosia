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

from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
import scipy.stats as sps

from ambrosia import types

ADMISSIBLE_TRANSFORMATIONS: List[str] = ["fraction"]


def calculate_point_effect_by_delta_method(
    mean_a: float,
    mean_b: float,
    var_group_a: float,
    var_group_b: float,  # pylint: disable=W0613
    covariance_ab: float,
    mean_size: int,  # Same size for both groups, TODO to change?
    transformation: str,
) -> float:
    """
    Calculate pvalue for continuous transformation using Delta method.

    Applying the Delta Method in Metric Analytics: A Practical Guide with Novel Ideas.
    Alex Deng, Ulf Knoblich, Jiannan Lu. 2018
    """
    if transformation == "fraction":
        fraction_estimation: float = mean_b / mean_a
        bias_correction: float = (mean_b / mean_a**3) * (var_group_a / mean_size) - (1.0 / mean_a**2) * (
            covariance_ab / mean_size
        )
        point_estimate = fraction_estimation - 1 + bias_correction
    else:
        raise ValueError(f"Got unknown random variable transformation: {ADMISSIBLE_TRANSFORMATIONS}")
    return point_estimate


def calc_statistic_for_delta_method(
    mean_a: float, mean_b: float, var_group_a: float, var_group_b: float, covariance_ab: float, size: int
) -> float:
    """
    Helps calculate statistic after delta method transformation
    """
    return np.sqrt(
        np.abs(
            (
                var_group_b / (mean_a**2)
                - 2 * covariance_ab * mean_b / mean_a**3
                + var_group_a * (mean_b**2) / mean_a**4
            )
            / size
        )
    )


def calculate_pvalue_by_delta_method(
    mean_a: float,
    mean_b: float,
    var_group_a: float,
    var_group_b: float,
    covariance_ab: float,
    mean_size: int,  # Same size for both groups, TODO to change?
    transformation: str,
    alternative: str = "two-sided",
) -> float:
    """
    Calculate pvalue for continuous transformation using Delta method.

    Applying the Delta Method in Metric Analytics: A Practical Guide with Novel Ideas.
    Alex Deng, Ulf Knoblich, Jiannan Lu. 2018

    Arguments
    ---------
    mean_a : float
        Metrics mean in group A
    mean_b : np.ndarray
        Metrics mean in group B
    variance_group_b
        Metrics variance in group B
    mean_size: int
        Mean size of two groups
    transformation : str
        Continuous transformation of random variable
    alternative : str, default: ``two-sided``
        Alternative for static criteria - two-sided, less, greater
        Less means, that mean in first group less, than mean in second group
    Returns
    -------
    pvalue : float
        P-value based on the first order Taylor expansion of continuous transformation
    """
    if transformation == "fraction":
        point_estimate = calculate_point_effect_by_delta_method(
            mean_a, mean_b, var_group_a, var_group_b, covariance_ab, mean_size, transformation
        )
        fraction_se: float = calc_statistic_for_delta_method(
            mean_a, mean_b, var_group_a, var_group_b, covariance_ab, mean_size
        )
        statistic: float = point_estimate / fraction_se
    else:
        raise ValueError(f"Got unknown random variable transformation: {ADMISSIBLE_TRANSFORMATIONS}")

    if alternative == "less":
        pvalue: float = sps.norm.cdf(statistic)
    elif alternative == "greater":
        pvalue: float = sps.norm.sf(statistic)
    elif alternative == "two-sided":
        pvalue: float = 2 * min(sps.norm.cdf(statistic), sps.norm.sf(statistic))
    else:
        raise ValueError(f"Incorrect alternative value - {alternative}, choose from two-sided, less, greater")
    return pvalue


def check_alternative(alternative: str) -> None:
    """
    Check correctness of alternative value
    """
    valid_alternatives: List[str] = ["two-sided", "less", "greater"]
    problem = f"Incorrect alternative value - {alternative}, choose from two-sided, less, greater"
    if alternative not in valid_alternatives:
        raise ValueError(problem)


def corrected_alpha(alpha: Union[float, np.ndarray], alternative: str) -> np.ndarray:
    """
    Corrects alpha according to alternative
    """
    if isinstance(alpha, float):
        return alpha if alternative == "two-sided" else min(2 * alpha, 1)
    else:
        return alpha if alternative == "two-sided" else np.minimum(2 * alpha, np.ones(alpha.shape[0]))


def choose_from_bounds(
    left_ci: np.ndarray,
    right_ci: np.ndarray,
    alternative: str = "two-sided",
    right_bound: float = np.inf,
    left_bound: float = -np.inf,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Choose left and right bounds according to alternative
    """
    cond_many: bool = isinstance(left_ci, Iterable)
    amount: int = len(left_ci) if cond_many else 1
    if alternative == "greater":
        right_ci = np.ones(amount) * right_bound if cond_many else right_bound
    if alternative == "less":
        left_ci = np.ones(amount) * left_bound if cond_many else left_bound
    return left_ci, right_ci


def calculate_intervals_by_delta_method(
    mean_a: float,
    mean_b: float,
    var_group_a: float,
    var_group_b: float,
    covariance_ab: float,
    mean_size: int,  # Same size for both groups, TODO to change?
    transformation: str,
    alpha: np.ndarray = np.array([0.05]),
    alternative: str = "two-sided",
) -> types.ManyIntervalType:
    """
    Computation of confidence intervals for each I type error bound (alpha) using Delta method.

    Applying the Delta Method in Metric Analytics: A Practical Guide with Novel Ideas.
    Alex Deng, Ulf Knoblich, Jiannan Lu. 2018

    Parameters
    ----------
    mean_a : float
        Metrics mean in group A
    mean_b : np.ndarray
        Metrics mean in group B
    variance_group_b
        Metrics variance in group B
    mean_size: int
        Mean size of two groups
    transformation : str
        Continuous transformation of random variable
    alpha : np.ndarray, default: ``np.array([0.05])``
        Array of I type errors bounds
    alternative : str, default ``two-sided``
        Alternative for static criteria - two-sided, less, greater
        Less means, that mean in first group less, than mean in second group

    Returns
    -------
    (left_bounds, right_bounds) : types.ManyIntervalType
        Confidence intervals based on the first order Taylor expansion
        of continuous transformation
    """
    if transformation == "fraction":
        point_estimate = calculate_point_effect_by_delta_method(
            mean_a, mean_b, var_group_a, var_group_b, covariance_ab, mean_size, transformation
        )
        fraction_se: float = calc_statistic_for_delta_method(
            mean_a, mean_b, var_group_a, var_group_b, covariance_ab, mean_size
        )
        correct_alpha: float = corrected_alpha(alpha, alternative)
        quantiles: np.ndarray = sps.norm.ppf(1 - correct_alpha / 2)
        shift: np.ndarray = quantiles * fraction_se
        left_bounds: np.ndarray = point_estimate - shift
        right_bounds: np.ndarray = point_estimate + shift
        left_bounds, right_bounds = choose_from_bounds(left_bounds, right_bounds, alternative)
    else:
        raise ValueError(f"Got unknown random variable transformation: {ADMISSIBLE_TRANSFORMATIONS}")
    conf_intervals: List[Tuple] = list(zip(left_bounds, right_bounds))
    return conf_intervals


def calculate_pvalue_by_interval(
    interval_function: Callable, criterion_value_label: float = 0, precision: float = 10e-7, **kwargs
) -> float:
    """
    Calculate pvalue for confidence interval.
    pvalue(x) = inf_a {a | x \\in S_a }
    S_a = {x | 0 not in interval(x) }.

    Parameters
    ----------
    interval_function : Callable
        Function returns confidence interval using kwargs
        Function must have argument confidence_level !!!
    criterion_value_label : float, default: ``0``
        This number indicates whether the null hypothesis should
        be rejected if it falls within the interval.
    precision : float, default: ``0.0000001``
        Precision for binary search solution.

    Returns
    -------
    pvalue : float
        P-value of the interval-induced criterion.
    """
    left: float = 0
    right: float = 1
    while right - left > precision:
        middle: float = (left + right) / 2
        conf_interval: Tuple[float, float] = interval_function(confidence_level=1 - middle, **kwargs)
        if conf_interval[0] <= criterion_value_label <= conf_interval[1]:
            left = middle
        else:
            right = middle
    return (left + right) / 2
