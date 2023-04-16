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

from math import asin
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
import scipy.stats as sps
import statsmodels.stats as stats
import statsmodels.stats.api as sms

import ambrosia.tools.pvalue_tools as pvalue_pkg
from ambrosia import types
from ambrosia.tools.configs import Alternatives

from . import EFFECT_COL_NAME, FIRST_TYPE_ERROR_COL_NAME, GROUP_SIZE_COL_NAME, STAT_ERRORS_COL_NAME

FIRST_TYPE_ERROR: float = 0.05
SECOND_TYPE_ERROR: float = 0.2
EFFECT_DEFAULT: float = 1.01
ROUND_DIGITS_TABLE: int = 3
ROUND_DIGITS_PERCENT: int = 1


def get_stats(values: Iterable[float], ddof: int = 1) -> Tuple[float, float]:
    """
    Calculate the mean and standard value for a list of values.
    """
    return np.mean(values), np.std(values, ddof=ddof)  # 1 for unbiased estimation


def get_table_stats(data: pd.DataFrame, column: types.ColumnNameType) -> Tuple[float, float]:
    """
    Calculate the mean and standard value a data frame column.
    """
    return get_stats(data[column].values)


def check_encode_alternative(alternative: str) -> str:
    """
    Check the correctness of the alternative and encode for use in statsmodels api.
    """
    alternatives: Set = {"two-sided", "greater", "less"}
    statsmodels_alternatives_encoding: Dict = {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}
    if alternative not in alternatives:
        raise ValueError(f"Alternative must be one of '{alternatives}'.")
    else:
        return statsmodels_alternatives_encoding[alternative]


def unbiased_to_sufficient(std: float, size: int) -> float:
    """
    Transforms unbiased estimation of standard deviation to sufficient
    (ddof = 1) => (ddof = 0)
    """
    return std * np.sqrt((size - 1) / size)


def check_target_type(
    dataframe: pd.DataFrame,
    column: types.ColumnNameType,
):
    """
    Check type of target: binary / non-binary.
    """
    unique_vals_count = dataframe[column].nunique()
    if unique_vals_count > 2:
        return "non-binary"
    unique_vals = set(dataframe[column].unique())
    if unique_vals == {0, 1} or unique_vals in {0, 1}:
        return "binary"
    return "non-binary"


def stabilize_effect(
    eff: float, mean: float, std: float, target_type: str = "binary", stabilizing_method: str = "asin"
):
    """
    Evaluate stabilized effect for solve_power method.
    """
    if target_type == "non-binary":
        return (eff - 1) * mean / std
    elif target_type == "binary":
        if stabilizing_method == "asin":
            return 2 * (asin((mean * eff) ** 0.5) - asin(mean**0.5))
        elif stabilizing_method == "norm":
            return mean * (eff - 1) / ((mean * (1 - mean)) ** 0.5)
        else:
            raise Exception("Invalid stabilizing_method")
    else:
        raise Exception("Invalid target_type")


def destabilize_effect(
    eff: float, mean: float, std: float, target_type: str = "binary", stabilizing_method: str = "asin"
):
    """
    Evaluate destabilized effect from solve_power method (statsmodels).
    """
    if target_type == "non-binary":
        return eff * std / mean
    elif target_type == "binary":
        if stabilizing_method == "asin":
            return np.sin((eff + 2 * asin(mean**0.5)) / 2) ** 2 / mean - 1
        elif stabilizing_method == "norm":
            return eff * (mean * (1 - mean)) ** 0.5 / mean
        else:
            raise Exception("Invalid stabilizing_method")
    else:
        raise Exception("Invalid target_type")


def get_sample_size(
    mean: float,
    std: float,
    eff: float = EFFECT_DEFAULT,
    alpha: float = FIRST_TYPE_ERROR,
    beta: float = SECOND_TYPE_ERROR,
    target_type: str = "non-binary",
    groups_ratio: float = 1.0,
    alternative: str = "two-sided",
    stabilizing_method: str = "asin",
):
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
    target_type : str, default: ``non-binary``
        Type of target metric: binary or non-binary
    groups_ratio : float, default: ``1.0``
        Ratio between two groups.
    alternative : str, default: ``"two-sided"``
        Alternative hypothesis, can be ``"two-sided"``, ``"greater"``
        or ``"less"``.
        ``"greater"`` - if effect is positive.
        ``"less"`` - if effect is negative.
    stabilizing_method : str, default: ``"asin"``
        Effect trasformation. Can be ``"asin"`` and ``"norm"``.
        For non-binary metrics: only ``"norm"`` is accceptable.
        For binary metrics: ``"norm"`` and ``"asin"``, but ``"asin"``
        is more robust and accurate.

    Returns
    -------
    sample_size : int
        Minimal sample size

    """
    alternative: str = check_encode_alternative(alternative)
    power_class = stats.power.TTestIndPower() if target_type == "non-binary" else stats.power.NormalIndPower()
    stabilized_effect = stabilize_effect(
        eff=eff, mean=mean, std=std, target_type=target_type, stabilizing_method=stabilizing_method
    )
    sample_size = power_class.solve_power(
        effect_size=stabilized_effect,
        nobs1=None,
        alpha=alpha,
        power=1 - beta,
        ratio=groups_ratio,
        alternative=alternative,
    )
    return int(np.ceil(sample_size))


def get_minimal_determinable_effect(
    mean: float,
    std: float,
    sample_size: int,
    alpha: float = FIRST_TYPE_ERROR,
    beta: float = SECOND_TYPE_ERROR,
    target_type: str = "non-binary",
    groups_ratio: float = 1.0,
    alternative: str = "two-sided",
    stabilizing_method: str = "asin",
):
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
    target_type : str, default: ``non-binary``
        Type of target metric: binary or non-binary
    groups_ratio : float, default: ``1.0``
        Ratio between two groups.
    alternative : str, default: ``"two-sided"``
        Alternative hypothesis, can be ``"two-sided"``, ``"greater"``
        or ``"less"``.
        ``"greater"`` - if effect is positive.
        ``"less"`` - if effect is negative.
    stabilizing_method : str, default: ``"asin"``
        Effect trasformation. Can be ``"asin"`` and ``"norm"``.
        For non-binary metrics: only ``"norm"`` is accceptable.
        For binary metrics: ``"norm"`` and ``"asin"``, but ``"asin"``
        is more robust and accurate.

    Returns
    -------
    mde : float
        Minimal effect which we can find

    """
    alternative: str = check_encode_alternative(alternative)
    power_class = stats.power.TTestIndPower() if target_type == "non-binary" else stats.power.NormalIndPower()
    stabilized_mde = power_class.solve_power(
        effect_size=None,
        nobs1=sample_size,
        alpha=alpha,
        power=1 - beta,
        ratio=groups_ratio,
        alternative=alternative,
    )
    mde = destabilize_effect(stabilized_mde, mean, std, target_type, stabilizing_method)
    return mde


def get_power(
    mean: float,
    std: float,
    sample_size: int,
    effect: float,
    alpha: float = FIRST_TYPE_ERROR,
    target_type: str = "non-binary",
    groups_ratio: float = 1.0,
    alternative: str = "two-sided",
    stabilizing_method: str = "asin",
) -> float:
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
    target_type : str, default: ``non-binary``
        Type of target metric: binary or non-binary
    groups_ratio : float, default: ``1.0``
        Ratio between two groups.
    alternative : str, default: ``"two-sided"``
        Alternative hypothesis, can be ``"two-sided"``, ``"greater"``
        or ``"less"``.
        ``"greater"`` - if effect is positive.
        ``"less"`` - if effect is negative.
    stabilizing_method : str, default: ``"asin"``
        Effect trasformation. Can be ``"asin"`` and ``"norm"``.
        For non-binary metrics: only ``"norm"`` is accceptable.
        For binary metrics: ``"norm"`` and ``"asin"``, but ``"asin"``
        is more robust and accurate.

    Returns
    -------
    power : float
        Power effect with fixed size and effect
    """
    alternative: str = check_encode_alternative(alternative)
    power_class = stats.power.TTestIndPower() if target_type == "non-binary" else stats.power.NormalIndPower()
    stabilized_effect = stabilize_effect(
        eff=effect, mean=mean, std=std, target_type=target_type, stabilizing_method=stabilizing_method
    )
    power = power_class.solve_power(
        effect_size=stabilized_effect,
        nobs1=sample_size,
        alpha=alpha,
        power=None,
        ratio=groups_ratio,
        alternative=alternative,
    )
    return power


def get_table_sample_size(
    mean: float,
    std: float,
    effects: types.EffectType,
    first_errors: types.StatErrorType = (0.05,),
    second_errors: types.StatErrorType = (0.2,),
    target_type: str = "non-binary",
    groups_ratio: float = 1.0,
    alternative: str = "two-sided",
    stabilizing_method: str = "asin",
):
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
    target_type : str, default: ``non-binary``
        Type of target metric: binary or non-binary
    groups_ratio : float, default: ``1.0``
        Ratio between two groups.
    alternative : str, default: ``"two-sided"``
        Alternative hypothesis, can be ``"two-sided"``, ``"greater"``
        or ``"less"``.
        ``"greater"`` - if effect is positive.
        ``"less"`` - if effect is negative.
    stabilizing_method : str, default: ``"asin"``
        Effect trasformation. Can be ``"asin"`` and ``"norm"``.
        For non-binary metrics: only ``"norm"`` is accceptable.
        For binary metrics: ``"norm"`` and ``"asin"``, but ``"asin"``
        is more robust and accurate.

    Returns
    -------
    df_results : pd.DataFrame
        Table with minimal sample sizes for each effect and error from input data.
    """
    multiindex = pd.MultiIndex.from_tuples([(eff,) for eff in effects], names=[EFFECT_COL_NAME])
    multicols = pd.MultiIndex.from_tuples(
        [(f"({err_one}; {err_two})",) for err_one in first_errors for err_two in second_errors],
        names=[STAT_ERRORS_COL_NAME],
    )
    df_results = pd.DataFrame(index=multiindex, columns=multicols)

    for eff in effects:
        for first_err in first_errors:
            for second_err in second_errors:
                err = f"({first_err}; {second_err})"
                df_results.loc[(eff,), (err,)] = get_sample_size(
                    mean=mean,
                    std=std,
                    eff=eff,
                    alpha=first_err,
                    beta=second_err,
                    target_type=target_type,
                    groups_ratio=groups_ratio,
                    alternative=alternative,
                    stabilizing_method=stabilizing_method,
                )
    df_results.index = pd.MultiIndex(
        levels=[[f"{np.round((x - 1) * 100, ROUND_DIGITS_PERCENT)}%" for x in effects]],
        codes=[np.arange(len(effects))],
        names=[EFFECT_COL_NAME],
    )
    return df_results


def design_groups_size(
    dataframe: pd.DataFrame,
    column: types.ColumnNameType,
    effects: Iterable[float],
    first_errors: Iterable[float],
    second_errors: Iterable[float],
    groups_ratio: float = 1.0,
    alternative: str = "two-sided",
    stabilizing_method: str = "asin",
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
    groups_ratio : float, default: ``1.0``
        Ratio between two groups.
    alternative : str, default: ``"two-sided"``
        Alternative hypothesis, can be ``"two-sided"``, ``"greater"``
        or ``"less"``.
        ``"greater"`` - if effect is positive.
        ``"less"`` - if effect is negative.
    stabilizing_method : str, default: ``"asin"``
        Effect trasformation. Can be ``"asin"`` and ``"norm"``.
        For non-binary metrics: only ``"norm"`` is accceptable.
        For binary metrics: ``"norm"`` and ``"asin"``, but ``"asin"``
        is more robust and accurate.

    Returns
    -------
    df_results : pd.DataFrame
        Table with minimal sample sizes for each effect and error from input data.
    """
    target_type = check_target_type(dataframe, column)
    mean, std = get_table_stats(dataframe, column)
    return get_table_sample_size(
        mean, std, effects, first_errors, second_errors, target_type, groups_ratio, alternative, stabilizing_method
    )


def get_minimal_effects_table(
    mean: float,
    std: float,
    sample_sizes: Iterable[int],
    first_errors: Iterable[float],
    second_errors: Iterable[float],
    as_numeric: bool = False,
    target_type: str = "non-binary",
    groups_ratio: float = 1.0,
    alternative: str = "two-sided",
    stabilizing_method: str = "asin",
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
    as_numeric : bool, default False
        Whether to return a number or a string with percentages
    target_type : str, default: ``non-binary``
        Type of target metric: binary or non-binary
    groups_ratio : float, default: ``1.0``
        Ratio between two groups.
    alternative : str, default: ``"two-sided"``
        Alternative hypothesis, can be ``"two-sided"``, ``"greater"``
        or ``"less"``.
        ``"greater"`` - if effect is positive.
        ``"less"`` - if effect is negative.
    stabilizing_method : str, default: ``"asin"``
        Effect trasformation. Can be ``"asin"`` and ``"norm"``.
        For non-binary metrics: only ``"norm"`` is accceptable.
        For binary metrics: ``"norm"`` and ``"asin"``, but ``"asin"``
        is more robust and accurate.

    Returns
    -------
    df_results : pd.DataFrame
        Table with minimal effects for each sample size and error from input data.
    """
    multiindex = pd.MultiIndex.from_tuples([(size,) for size in sample_sizes], names=[GROUP_SIZE_COL_NAME])
    multicols = pd.MultiIndex.from_tuples(
        [(f"({err_one}; {err_two})",) for err_one in first_errors for err_two in second_errors],
        names=[STAT_ERRORS_COL_NAME],
    )
    df_results = pd.DataFrame(index=multiindex, columns=multicols)
    for sample_size in sample_sizes:
        for first_err in first_errors:
            for second_err in second_errors:
                err = f"({first_err}; {second_err})"
                effect = get_minimal_determinable_effect(
                    mean=mean,
                    std=std,
                    sample_size=sample_size,
                    alpha=first_err,
                    beta=second_err,
                    target_type=target_type,
                    groups_ratio=groups_ratio,
                    alternative=alternative,
                    stabilizing_method=stabilizing_method,
                )
                if as_numeric:
                    df_results.loc[(sample_size,), (err,)] = round(effect, ROUND_DIGITS_TABLE) + 1
                else:
                    df_results.loc[(sample_size,), (err,)] = str(np.round(effect * 100, ROUND_DIGITS_PERCENT)) + "%"
    df_results.index = pd.MultiIndex(
        levels=[sample_sizes],
        codes=[np.arange(len(sample_sizes))],
        names=[GROUP_SIZE_COL_NAME],
    )
    return df_results


def design_effect(
    dataframe: pd.DataFrame,
    column: types.ColumnNameType,
    sample_sizes: Iterable[int],
    first_errors: Iterable[float],
    second_errors: Iterable[float],
    as_numeric: bool = False,
    groups_ratio: float = 1.0,
    alternative: str = "two-sided",
    stabilizing_method: str = "asin",
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
    as_numeric : bool, default False
        Whether to return a number or a string with percentages
    groups_ratio : float, default: ``1.0``
        Ratio between two groups.
    alternative : str, default: ``"two-sided"``
        Alternative hypothesis, can be ``"two-sided"``, ``"greater"``
        or ``"less"``.
        ``"greater"`` - if effect is positive.
        ``"less"`` - if effect is negative.
    stabilizing_method : str, default: ``"asin"``
        Effect trasformation. Can be ``"asin"`` and ``"norm"``.
        For non-binary metrics: only ``"norm"`` is accceptable.
        For binary metrics: ``"norm"`` and ``"asin"``, but ``"asin"``
        is more robust and accurate.

    Returns
    -------
    df_results : pd.DataFrame
        Table with minimal effects for each sample size and error from input data.
    """
    target_type = check_target_type(dataframe, column)
    mean, std = get_table_stats(dataframe, column)
    return get_minimal_effects_table(
        mean,
        std,
        sample_sizes,
        first_errors,
        second_errors,
        as_numeric,
        target_type,
        groups_ratio,
        alternative,
        stabilizing_method,
    )


def get_power_table(
    mean: float,
    std: float,
    sample_sizes: Iterable[int],
    effects: Iterable[float],
    first_errors: Iterable[float] = (FIRST_TYPE_ERROR,),
    as_numeric: bool = False,
    target_type: str = "non-binary",
    groups_ratio: float = 1.0,
    alternative: str = "two-sided",
    stabilizing_method: str = "asin",
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
    as_numeric : bool, default False
        Whether to return a number or a string with percentages
    target_type : str, default: ``non-binary``
        Type of target metric: binary or non-binary
    groups_ratio : float, default: ``1.0``
        Ratio between two groups.
    alternative : str, default: ``"two-sided"``
        Alternative hypothesis, can be ``"two-sided"``, ``"greater"``
        or ``"less"``.
        ``"greater"`` - if effect is positive.
        ``"less"`` - if effect is negative.
    stabilizing_method : str, default: ``"asin"``
        Effect trasformation. Can be ``"asin"`` and ``"norm"``.
        For non-binary metrics: only ``"norm"`` is accceptable.
        For binary metrics: ``"norm"`` and ``"asin"``, but ``"asin"``
        is more robust and accurate.

    Returns
    -------
    df_results : pd.DataFrame
        Table with  sample sizes for each effect and error from input data.
    """
    effects_str = [str(round((effect - 1) * 100, ROUND_DIGITS_PERCENT)) + "%" for effect in effects]
    multiindex = pd.MultiIndex.from_tuples(
        [(first_error, effect_str) for first_error in first_errors for effect_str in effects_str],
        names=[FIRST_TYPE_ERROR_COL_NAME, EFFECT_COL_NAME],
    )
    powers: List[np.ndarray] = []
    for first_err in first_errors:
        for effect in effects:
            power: np.ndarray = get_power(
                mean=mean,
                std=std,
                sample_size=np.array(sample_sizes),
                effect=effect,
                alpha=first_err,
                target_type=target_type,
                groups_ratio=groups_ratio,
                alternative=alternative,
                stabilizing_method=stabilizing_method,
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
    df_results.columns.name = "Group sizes"
    return df_results


def design_power(
    dataframe: pd.DataFrame,
    column: types.ColumnNameType,
    sample_sizes: Iterable[int],
    effects: Iterable[float],
    first_errors: Iterable[float] = (FIRST_TYPE_ERROR,),
    as_numeric: bool = False,
    groups_ratio: float = 1.0,
    alternative: str = "two-sided",
    stabilizing_method: str = "asin",
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
    as_numeric : bool, default False
        Whether to return a number or a string with percentages
    groups_ratio : float, default: ``1.0``
        Ratio between two groups.
    alternative : str, default: ``"two-sided"``
        Alternative hypothesis, can be ``"two-sided"``, ``"greater"``
        or ``"less"``.
        ``"greater"`` - if effect is positive.
        ``"less"`` - if effect is negative.
    stabilizing_method : str, default: ``"asin"``
        Effect trasformation. Can be ``"asin"`` and ``"norm"``.
        For non-binary metrics: only ``"norm"`` is accceptable.
        For binary metrics: ``"norm"`` and ``"asin"``, but ``"asin"``
        is more robust and accurate.

    Returns
    -------
    df_results : pd.DataFrame
        Table with power for each effect and samples from input data.
    """
    target_type = check_target_type(dataframe, column)
    mean, std = get_table_stats(dataframe, column)
    return get_power_table(
        mean,
        std,
        sample_sizes,
        effects,
        first_errors,
        as_numeric,
        target_type,
        groups_ratio,
        alternative,
        stabilizing_method,
    )


def get_ttest_info_from_stats(
    var_a: float, var_b: float, n_obs_a: int, n_obs_b: int, alpha: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns quantiles and standard deviation of Ttest criterion statistic
    """
    compound_se: float = np.sqrt(var_a / n_obs_a + var_b / n_obs_b)
    denominator: float = (var_a / n_obs_a) ** 2 / (n_obs_a - 1) + (var_b / n_obs_b) ** 2 / (n_obs_b - 1)
    dim: float = compound_se**2 / denominator
    quantiles: np.ndarray = sps.t.ppf(1 - alpha / 2, df=dim)
    return quantiles, compound_se


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
    return get_ttest_info_from_stats(variance_group_a, variance_group_b, len(group_a), len(group_b), alpha)


def apply_delta_method_by_stats(
    size: int,
    mean_group_a: float,
    var_group_a: float,
    mean_group_b: float,
    var_group_b: float,
    cov_groups: float = 0,
    transformation: str = "fraction",
    alpha: np.ndarray = np.array([FIRST_TYPE_ERROR]),
    alternative: str = "two-sided",
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


def ttest_1samp_from_stats(
    mean: float, std: float, n_obs: int, alternative: str = Alternatives.ts.value
) -> Tuple[float, float]:
    """
    Implementation of ttest_1samp for stats, not from observations

    Parameters
    ----------
    mean: float
        Mean of samples
    std: float
        Standart deviation (consistent estimation)
    n_obs: int
        Amount of observations
    alternative: str
        One of two-sided, less, greater

    Returns
    -------
    (statistic, pvalue): Tuple[float, float]
    Statistic of criterion and pvalue
    """
    statistic: float = mean / std * np.sqrt(n_obs)

    if alternative == Alternatives.gr.value:
        pvalue: float = sps.t.sf(statistic, df=n_obs - 1)
    elif alternative == Alternatives.less.value:
        pvalue: float = sps.t.cdf(statistic, df=n_obs - 1)
    elif alternative == Alternatives.ts.value:
        pvalue: float = sps.t.cdf(-np.abs(statistic), df=n_obs - 1) + sps.t.sf(np.abs(statistic), df=n_obs - 1)
    else:
        Alternatives.raise_if_value_incorrect_enum(alternative)
    return statistic, pvalue
