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

from typing import Iterable, Tuple

import pandas as pd
import pyspark.sql.functions as funcs
import scipy.stats as sps

import ambrosia.tools.theoretical_tools as theory_pkg
from ambrosia import types


def get_stats_from_table(dataframe: types.SparkDataFrame, column: types.ColumnNameType) -> Tuple[float, float]:
    """
    Get table for designing samples size for experiment.
    """
    stats = dataframe.select(
        funcs.mean(funcs.col(column)).alias("mean"), funcs.stddev(funcs.col(column)).alias("std")
    ).collect()
    mean = stats[0]["mean"]
    std = stats[0]["std"]
    return mean, std


def design_groups_size(
    dataframe: types.SparkDataFrame,
    column: types.ColumnNameType,
    effects: Iterable[float],
    second_errors: Iterable[float],
    first_errors: Iterable[float] = (0.05,),
) -> pd.DataFrame:
    """
    Create pandas dataframe with designed groups sizes for metric based on data
    in spark dataframe.

    Results in returned dataframe depend on desired effects, first and second type errors,
    and statistic of metric from given spark table.

    Parameters
    ----------
    dataframe : types.SparkDataFrame
        Table for designing experiment
    column : types.ColumnNameType
        Column, containg metric for designing
    effects : Iterable[float]
        List of effects which we want to catch.
        e.x.: [1.01, 1.02, 1.05]
    second_errors : Iterable[float]
        1st and 2nd type errors.
        e.x.: [0.01, 0.05, 0.1]
    first_errors : Iterable[float], default: ``(0.05,)``
        1st and 2nd type errors.
        e.x.: [0.01, 0.05, 0.1]

    Returns
    -------
    df_results : pd.DataFrame
        Table with minimal sample sizes for each effect and errors pair based on input data.
    """
    mean, std = get_stats_from_table(dataframe, column)
    return theory_pkg.get_table_sample_size(mean, std, effects, first_errors, second_errors)


def design_effect(
    dataframe: types.SparkDataFrame,
    column: types.ColumnNameType,
    sample_sizes: Iterable[int],
    second_errors: Iterable[float],
    first_errors: Iterable[float] = (0.05,),
) -> pd.DataFrame:
    """
    Create pandas dataframe with designed effects for metric based on data in spark dataframe.
    Results in returned dataframe depend on sample sizes, first and second type errors,
    and statistic of metric from given spark table.

    Parameters
    ----------
    dataframe : types.SparkDataFrame
        Spark table with data for experiment design
    column : types.ColumnNameType
        Column, containg metric for designing
    sample_sizes : Iterable[int]
        List of sample sizes which we want to check.
        e.x.: [100, 200, 1000]
    second_errors : Iterable[float]
        2nd type errors.
        e.x.: [0.01, 0.05, 0.1]
    first_errors : Iterable[float], default: ``(0.05,)``
        1st type errors.
        e.x.: [0.01, 0.05, 0.1]

    Returns
    -------
    df_results : pd.DataFrame
        Table with minimal effects for each sample size and errors pair based on input data.
    """
    mean, std = get_stats_from_table(dataframe, column)
    return theory_pkg.get_minimal_effects_table(mean, std, sample_sizes, first_errors, second_errors)


def design_power(
    dataframe: types.SparkDataFrame,
    column: types.ColumnNameType,
    sample_sizes: Iterable[int],
    effects: Iterable[float],
    first_errors: Iterable[float] = (0.05,),
) -> pd.DataFrame:
    """
    Create pandas dataframe with designed power for metric based on data in spark dataframe.
    Results in returned dataframe depend on sample sizes, disered effects, first type errors,
    and statistic of metric from given spark table.

    Parameters
    ----------
    dataframe : types.SparkDataFrame
        Spark table with data for experiment design
    column : types.ColumnNameType
        Column, containg metric for designing
    sample_sizes : Iterable[int]
        List of sample sizes which we want to check.
        e.x.: [100, 200, 1000]
    effects : Iterable[float]
        Iterable object with
        e.x.: [1.01, 1.02, 1.05]
    first_errors : Iterable[float], default: ``(0.05,)``
        1st and 2nd type errors.
        e.x.: [0.01, 0.05, 0.1]

    Returns
    -------
    df_results : pd.DataFrame
        Table with power for each sample size, effect and first type error based on input data.
    """
    mean, std = get_stats_from_table(dataframe, column)
    return theory_pkg.get_power_table(mean, std, sample_sizes, effects, first_errors)


def ttest_spark(
    first_group: types.SparkDataFrame, second_group: types.SparkDataFrame, column: types.ColumnNameType
) -> Tuple[float, float]:
    """
    T-test for independent groups.

    Parameters
    ----------
    first_group : Spark Data Frame
        Data Frame for first group
    first_group : Spark Data Frame
        Data Frame for second group
    column : Column Type
        Column to be tested

    Returns
    -------
    statistic, pvalue : Tuple[float, float]
        T-test result
    """
    mean_1, std_1 = get_stats_from_table(first_group, column)
    mean_2, std_2 = get_stats_from_table(second_group, column)
    n_obs_1: int = first_group.count()
    n_obs_2: int = second_group.count()
    return sps.ttest_ind_from_stats(mean_1, std_1, n_obs_1, mean_2, std_2, n_obs_2)
