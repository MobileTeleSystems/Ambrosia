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

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as sps
from scipy.stats import norm

import ambrosia.tools.pvalue_tools as pvalue_pkg
from ambrosia import types

FIRST_TYPE_ERROR: float = 0.05
SECOND_TYPE_ERROR: float = 0.2
EFFECT_DEFAULT: float = 1.01
ROUND_DIGITS_TABLE: int = 3
ROUND_DIGITS_PERCENT: int = 1


def get_stats(values: Iterable[float], ddof: int = 1) -> Tuple[float, float]:
    return np.mean(values), np.std(values, ddof=ddof)  # 1 for ubbiased estimation


def get_table_stats(data: pd.DataFrame, column: types.ColumnNameType) -> Tuple[float, float]:
    return get_stats(data[column].values)


def get_sample_size(mean, std, eff=EFFECT_DEFAULT, alpha=FIRST_TYPE_ERROR, beta=SECOND_TYPE_ERROR):
    """
    Calculate minimum sample size to catch effect with fixed errors.

    Parameters
    ----------
    mean : float
        Sample mean
    std : float
        Sample standard deviation
    eff : float, default: ``1.01``
        Effect for which we calculate sample size
    alpha : float, default: ``0.05``
        First type error
    beta : float, default: ``0.2``
        Second type error

    Returns
    -------
    sample_size : int
        Minimal sample size

    """
    t_alpha = abs(norm.ppf(alpha / 2, loc=0, scale=1))
    t_beta = norm.ppf(1 - beta, loc=0, scale=1)

    mu_diff_squared = (mean - mean * eff) ** 2
    z_scores_sum_squared = (t_alpha + t_beta) ** 2
    disp_sum = 2 * (std**2)
    sample_size = int(np.ceil(z_scores_sum_squared * disp_sum / mu_diff_squared))
    return sample_size


def get_minimal_determinable_effect(mean, std, sample_size, alpha=FIRST_TYPE_ERROR, beta=SECOND_TYPE_ERROR):
    """
    Calculate power for given minimum detectable effect and group size.

    Parameters
    ----------
    mean : float
        Sample mean
    std : float
        Sample standard deviation
    sample_size : int
        Size of sample
    alpha : float, default: ``0.05``
        First type error
    beta : float, default: ``0.2``
        Second type error

    Returns
    -------
    mde : float
        Minimal effect which we can find

    """
    t_alpha = abs(norm.ppf(alpha / 2, loc=0, scale=1))
    t_beta = norm.ppf(1 - beta, loc=0, scale=1)

    z_scores_sum = t_alpha + t_beta
    disp_sum_sqrt = ((std**2) + (std**2)) ** 0.5
    mde = z_scores_sum * disp_sum_sqrt / (mean * np.sqrt(sample_size))
    return mde


def get_power(mean: float, std: float, sample_size: int, effect: float, alpha: float = FIRST_TYPE_ERROR) -> float:
    """
    Calculate minimum detectable effect which we can find.

    Parameters
    ----------
    mean : float
        Sample mean
    std : float
        Sample standard deviation
    sample_size : int
        Size of sample
    effect : float
        Second type error
    alpha : float, default: ``1.01``
        First type error

    Returns
    -------
    power : float
        Power effect with fixed size and effect
    """
    absolute_effect: float = mean * (1 - effect)
    point: float = norm.ppf(1 - alpha / 2) + np.sqrt(sample_size / 2) / std * absolute_effect
    second_error: float = norm.cdf(point)
    return 1 - second_error


def get_table_sample_size(mean, std, effects, first_errors, second_errors):
    """
    Create table of sample sizes for different effects and errors.

    Parameters
    ----------
    mean : float
        Sample mean
    std : float
        Sample standard deviation.
    effects : List
        List of effects which we want to catch.
        e.x.: [1.01, 1.02, 1.05]
    first_errors : List
        1st and 2nd type errors.
        e.x.: [0.01, 0.05, 0.1]
    second_errors : List
        1st and 2nd type errors.
        e.x.: [0.01, 0.05, 0.1]

    Returns
    -------
    df_results : pandas df
        Table with minimal sample sizes for each effect and error from input data.
    """
    multiindex = pd.MultiIndex.from_tuples([(eff,) for eff in effects], names=["effect"])
    multicols = pd.MultiIndex.from_tuples(
        [(f"({err_one}; {err_two})",) for err_one in first_errors for err_two in second_errors], names=["errors"]
    )
    df_results = pd.DataFrame(index=multiindex, columns=multicols)

    for eff in effects:
        for first_err in first_errors:
            for second_err in second_errors:
                err = f"({first_err}; {second_err})"
                df_results.loc[(eff,), (err,)] = get_sample_size(
                    mean=mean, std=std, eff=eff, alpha=first_err, beta=second_err
                )
    df_results.index = pd.MultiIndex(
        levels=[[f"{np.round((x - 1) * 100, ROUND_DIGITS_PERCENT)}%" for x in effects]],
        codes=[np.arange(len(effects))],
        names=["effects"],
    )
    return df_results


def design_groups_size(
    dataframe: pd.DataFrame,
    column: types.ColumnNameType,
    effects: Iterable[float],
    first_errors: Iterable[float],
    second_errors: Iterable[float],
) -> pd.DataFrame:
    """
    Get table for designing samples size for experiment.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Table for designing the experiment
    column: Column name type
        Column, containg metric for designing
    effects: Iterable of floats
        List of effects which we want to catch.
        e.x.: [1.01, 1.02, 1.05]
    first_errors: Iterable of floats
        1st and 2nd type errors.
        e.x.: [0.01, 0.05, 0.1]
    second_errors: Iterable of floats
        1st and 2nd type errors.
        e.x.: [0.01, 0.05, 0.1]

    Returns
    -------
    df_results : pd.DataFrame
        Table with minimal sample sizes for each effect and error from input data.
    """
    mean, std = get_table_stats(dataframe, column)
    return get_table_sample_size(mean, std, effects, first_errors, second_errors)


def get_minimal_effects_table(
    mean: float,
    std: float,
    sample_sizes: Iterable[int],
    first_errors: Iterable[float],
    second_errors: Iterable[float],
    as_numeric: bool = False,
) -> pd.DataFrame:
    """
    Create table of effects for different sample sizes and errors.

    Parameters
    ----------
    mean : float
        Sample mean
    std : float
        Sample standard deviation.
    sample_sizes : List
        List of sample sizes which we want to check.
        e.x.: [100, 200, 1000]
    first_errors : List
        1st and 2nd type errors.
        e.x.: [0.01, 0.05, 0.1]
    second_errors : List
        1st and 2nd type errors.
        e.x.: [0.01, 0.05, 0.1]
    as_numeric: bool, default False
        Whether to return a number or a string with percentages

    Returns
    -------
    df_results : pd.DataFrame
        Table with minimal effects for each sample size and error from input data.
    """
    multiindex = pd.MultiIndex.from_tuples([(size,) for size in sample_sizes], names=["sample_size"])
    multicols = pd.MultiIndex.from_tuples(
        [(f"({err_one}; {err_two})",) for err_one in first_errors for err_two in second_errors], names=["errors"]
    )
    df_results = pd.DataFrame(index=multiindex, columns=multicols)
    for sample_size in sample_sizes:
        for first_err in first_errors:
            for second_err in second_errors:
                err = f"({first_err}; {second_err})"
                effect = get_minimal_determinable_effect(
                    mean=mean, std=std, sample_size=sample_size, alpha=first_err, beta=second_err
                )
                str_effect = str(np.round(effect * 100, ROUND_DIGITS_PERCENT)) + "%"
                if as_numeric:
                    df_results.loc[(sample_size,), (err,)] = round(effect, ROUND_DIGITS_TABLE) + 1
                else:
                    df_results.loc[(sample_size,), (err,)] = str_effect
    df_results.index = pd.MultiIndex(
        levels=[sample_sizes],
        codes=[np.arange(len(sample_sizes))],
        names=["sample_sizes"],
    )
    return df_results


def design_effect(
    dataframe: pd.DataFrame,
    column: types.ColumnNameType,
    sample_sizes: Iterable[int],
    first_errors: Iterable[float],
    second_errors: Iterable[float],
    as_numeric: bool = False,
) -> pd.DataFrame:
    """
    Create table of effects for different sample sizes and errors.

    Parameters
    ----------
    dataframe : pandas Data Frame
        Table for data to be designed
    column : Column name type
        Column of metric to be designed
    sample_sizes : List
        List of sample sizes which we want to check.
        e.x.: [100, 200, 1000]
    first_errors : List
        1st and 2nd type errors.
        e.x.: [0.01, 0.05, 0.1]
    second_errors : List
        1st and 2nd type errors.
        e.x.: [0.01, 0.05, 0.1]
    as_numeric: bool, default False
        Whether to return a number or a string with percentages

    Returns
    -------
    df_results : pandas df
        Table with minimal effects for each sample size and error from input data.
    """
    mean, std = get_table_stats(dataframe, column)
    return get_minimal_effects_table(mean, std, sample_sizes, first_errors, second_errors, as_numeric)


def get_power_table(
    mean: float,
    std: float,
    sample_sizes: Iterable[int],
    effects: Iterable[float],
    first_errors: Iterable[float] = (FIRST_TYPE_ERROR,),
    as_numeric: bool = False,
) -> pd.DataFrame:
    """
    Create table of power for different sample sizes and effects.

    Parameters
    ----------
    mean : float
        Sample mean
    std : float
        Sample standard deviation.
    sample_sizes : Iterable[int]
        List of sample sizes which we want to check.
        e.x.: [100, 200, 1000]
    effects : Iterable[float]
        Iterable object with
        e.x.: [1.01, 1.02, 1.05]
    first_errors : Iterable, default: ``(0.05,)``
        1st and 2nd type errors.
        e.x.: [0.01, 0.05, 0.1]
    as_numeric: bool, default False
        Whether to return a number or a string with percentages

    Returns
    -------
    df_results : pandas df
        Table with  sample sizes for each effect and error from input data.
    """
    effects_str = [str(round((effect - 1) * 100, ROUND_DIGITS_PERCENT)) + "%" for effect in effects]
    multiindex = pd.MultiIndex.from_tuples(
        [(first_error, effect_str) for first_error in first_errors for effect_str in effects_str],
        names=["First type error", "Effect"],
    )
    powers: List[np.ndarray] = []
    for effect in effects:
        for first_err in first_errors:
            power: np.ndarray = get_power(
                mean=mean, std=std, sample_size=np.array(sample_sizes), effect=effect, alpha=first_err
            )
            if as_numeric:
                power = [np.round(p, ROUND_DIGITS_TABLE) for p in power]
            else:
                power = [str(np.round(p * 100, ROUND_DIGITS_PERCENT)) + "%" for p in power]
            powers.append(power)
    df_results = pd.DataFrame(
        np.vstack(powers),
        columns=sample_sizes,
        index=multiindex,
    )
    df_results.index.name = "Errors and Effects"
    df_results.columns.name = "sample sizes"
    return df_results


def design_power(
    dataframe: pd.DataFrame,
    column: types.ColumnNameType,
    sample_sizes: Iterable[int],
    effects: Iterable[float],
    first_errors: Iterable[float] = (FIRST_TYPE_ERROR,),
    as_numeric: bool = False,
) -> pd.DataFrame:
    """
    Create table of power for different sample sizes and effects.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Table for data to be designed
    column : Column name type
        Sample standard deviation.
    sample_sizes : Iterable[int]
        List of sample sizes which we want to check.
        e.x.: [100, 200, 1000]
    effects : Iterable[float]
        Iterable object with
        e.x.: [1.01, 1.02, 1.05]
    first_errors : Iterable[float], default: ``(0.05,)``
        1st and 2nd type errors.
        e.x.: [0.01, 0.05, 0.1]
    as_numeric: bool, default False
        Whether to return a number or a string with percentages

    Returns
    -------
    df_results : pandas df
        Table with power for each effect and samples from input data.
    """
    mean, std = get_table_stats(dataframe, column)
    return get_power_table(mean, std, sample_sizes, effects, first_errors, as_numeric=as_numeric)


def get_ttest_info(group_a: np.ndarray, group_b: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute standart error and quatiles for ttest (Welch)
    https://en.wikipedia.org/wiki/Student%27s_t-test

    Arguments
    ---------
    group_a : np.ndarray
        Group A values
    group_b : np.ndarray
        Group B values
    alpha : np.ndarray
        First type error values

    Returns:
    -------
    quantiles, compound_se : np.ndarray, float
        Quantiles of T_dim corresponding each alpha
        compound_se - Standart error used in t-test
    """
    variance_group_a: float = group_a.var(ddof=1)
    variance_group_b: float = group_b.var(ddof=1)
    compound_se: float = np.sqrt(variance_group_a / len(group_a) + variance_group_b / len(group_b))
    denominator: float = (variance_group_a / len(group_a)) ** 2 / (len(group_a) - 1) + (
        variance_group_b / len(group_b)
    ) ** 2 / (len(group_b) - 1)
    dim: float = compound_se**2 / denominator
    quantiles: np.ndarray = sps.t.ppf(1 - alpha / 2, df=dim)
    return quantiles, compound_se


def apply_delta_method_by_stats(
    size: int,
    mean_group_a: float,
    var_group_a: float,
    mean_group_b: float,
    var_group_b: float,
    cov_groups: float = 0,
    transformation: str = "fraction",
    alpha: np.ndarray = np.array([FIRST_TYPE_ERROR]),
    alternative: str = "two_sided",
) -> Tuple[types.ManyIntervalType, float]:
    """
    Computation of pvalue and confidence intervals for each I type error bound (alpha)
    using Delta-method by statistics.

    Arguments
    ---------
    size: int
        Size of both groups
    mean_group_a: float
        Mean of metrics from group A
    var_group_a: float
        Consistent estimation of variation (ddof = 0) of metrics from group A
    mean_group_b: float
        Mean of metrics from group B
    var_group_b: float
        Consistent estimation of variation (ddof = 0) of metrics from group B
    cov_groups: float, default ``0``
        Covariation between groups for dependent samples
    transformation : str, default: ``fraction``
        Continuous transformation of random variable
    alpha : np.ndarray, default: ``np.array([0.05])``
        Lists of I type errors bounds
    alternative : str, default: ``two-sided``
        Alternative for static criteria - two-sided, less, greater
        Less means, that mean in first group less, than mean in second group.
    Returns
    -------
    (intervals, pvalue) : Tuple[types.ManyIntervalType, float]
        Confidence intervals with given sufficient level
        And
        Pvalue for corresponding criterion
            H0: mean(A) / mean(B) = 1
                vs
            H1: mean(A) / mean(B) <> 1
    """
    admissible_transformations: List[str] = ["fraction"]
    if transformation in admissible_transformations:
        conf_intervals = pvalue_pkg.calculate_intervals_by_delta_method(
            mean_group_a, mean_group_b, var_group_a, var_group_b, cov_groups, size, transformation, alpha, alternative
        )
        pvalue = pvalue_pkg.calculate_pvalue_by_delta_method(
            mean_group_a, mean_group_b, var_group_a, var_group_b, cov_groups, size, transformation, alternative
        )
    else:
        raise ValueError(f'Choose method from {", ".join(admissible_transformations)}, got {transformation}')
    return conf_intervals, pvalue


def apply_delta_method(
    group_a: np.ndarray,
    group_b: np.ndarray,
    transformation: str,
    alpha: np.ndarray = np.array([FIRST_TYPE_ERROR]),
    dependent: bool = False,
    alternative: str = "two-sided",
) -> Tuple[types.ManyIntervalType, float]:
    """
    Computation of pvalue and confidence intervals for each I type error bound
    (alpha) using Delta-method.

    Parameters
    ----------
    group_a : np.ndarray
        Metrics from group A
    group_b : np.ndarray
        Metrics from group B
    transformation : str
        Continuous transformation of random variable
    alpha : np.ndarray, default: ``np.array([0.05])``
        Lists of I type errors bounds
    alternative : str, default: ``two-sided``
        Alternative for static criteria - two-sided, less, greater
        Less means, that mean in first group less, than mean in second group.

    Returns
    -------
    (intervals, pvalue) : Tuple[types.ManyIntervalType, float]
        Confidence intervals with given sufficient level
        And
        Pvalue for corresponding criterion
            H0: mean(A) / mean(B) = 1
                vs
            H1: mean(A) / mean(B) <> 1
    """
    mean_size: int = (group_a.shape[0] + group_b.shape[0]) // 2
    mean_a, std_group_a = get_stats(group_a, ddof=0)
    mean_b, std_group_b = get_stats(group_b, ddof=0)
    covariance_ab: float = np.cov(group_a, group_b)[0][1] if dependent else 0
    return apply_delta_method_by_stats(
        mean_size,
        mean_a,
        std_group_a**2,
        mean_b,
        std_group_b**2,
        covariance_ab,
        transformation,
        alpha,
        alternative=alternative,
    )
