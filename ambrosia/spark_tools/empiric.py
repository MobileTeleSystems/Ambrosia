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

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm

import ambrosia.spark_tools.theory as th_pkg
import ambrosia.tools._lib._bootstrap_tools as solver_pkg
import ambrosia.tools._lib._selection_aide as select_pkg
from ambrosia import types
from ambrosia.tools.import_tools import spark_installed

if spark_installed():
    import pyspark.sql.functions as spark_functions

BOOSTRAP_BASE_CONST: int = 10
RANDOM_SAMPLE_SEED: int = 42
FIRST_TYPE_ERROR: float = 0.05
THREADS_BOOTSTRAP: int = 2  # Creates a significant reduction in runtime
N_JOBS_MULTIPROCESS: int = 1
ACCEPTED_CRITERIA: List[str] = ["ttest"]
BOOTSTRAP_BACKEND: str = "threading"
ROUND_DIGITS_PERCENT: int = 1


def inject_effect(dataframe: types.SparkDataFrame, column: types.ColumnNameType, effect: float) -> types.SparkDataFrame:
    """
    Injects effect to column of given dataframe and returns injected one.

    Injection conducts via adding mean * delta_relative_effect.

    Parameters
    ----------
    dataframe : Spark dataframe
        Table with column, where efect will be injected
    column : Column type
        Column which will be used
    effect : float
        Value of effect, for example for 20% percent growth pass 1.2

    Returns
    -------
    effected_dataframe : Spark dataframe
        Table with changed column
    """
    multiplicator: float = effect - 1
    current_mean, _ = th_pkg.get_stats_from_table(dataframe, column)
    return dataframe.withColumn(column, spark_functions.col(column) + current_mean * multiplicator)


def evaluate_criterion(
    dataframe: types.SparkDataFrame,
    column: types.ColumnNameType,
    effect: float,
    group_size: int,
    alpha: float = FIRST_TYPE_ERROR,
    criterion: str = ACCEPTED_CRITERIA[0],
) -> int:
    """
    Evaluate criterion, returns 0 if H1 is not rejected, 1 otherwise.

    Check hypotesis for given column, if effect injected for given sample size.
    H0: means in two groups equals, H1: means are not equal

    Parameters
    ----------
    dataframe : Spark dataframe
        Table with column, which wiil be used
    column : Column type
        Column with metric, which will be used in criterion
    effect : float
        Value of effect to be tested, for example, for 20% effect pass 1.2
    group_size : int
        Size for each of two groups, which wiil be sampled
    alpha : float
        Bound for first type error, will be used as pvalue <= alpha
    criterion : str
        Name of criterion, default ttest, see list acceptable criteria

    Returns
    -------
    is_rejected : bool
        Is H1 correct
    """
    total_size: int = dataframe.count()
    if group_size * 2 > total_size:
        err_msg: str = "Total sampled values more than table size"
        raise ValueError(err_msg)

    part: float = 2 * group_size / total_size
    data_a, data_b = dataframe.sample(part).randomSplit([0.5, 0.5], seed=RANDOM_SAMPLE_SEED)
    data_b = inject_effect(data_b, column, effect)

    if criterion == "ttest":
        _, pvalue = th_pkg.ttest_spark(data_a, data_b, column)
    else:
        err_msg: str = f"Choose criterion from {ACCEPTED_CRITERIA}"
        raise ValueError(err_msg)
    return pvalue < alpha


def calc_empiric_power(
    dataframe: types.SparkDataFrame,
    column: types.ColumnNameType,
    effect: float,
    group_size: int,
    first_error: float = FIRST_TYPE_ERROR,
    bootstrap_size: int = BOOSTRAP_BASE_CONST,
    criterion: str = ACCEPTED_CRITERIA[0],
    threads: int = THREADS_BOOTSTRAP,
) -> float:
    """
    Calculate empiric power of criterion via thread pool.

    Parameters
    ----------
    dataframe : Spark dataframe
        Table with column, which wiil be used
    column : Column type
        Column with metric, which will be used in criterion
    effect : float
        Value of effect to be tested, for example, for 20% effect pass 1.2
    group_size: int
        Size for each of two groups, which wiil be sampled
    first_error : float
        Bound for first type error, will be used as pvalue <= first_error
    bootstrap_size : int
        Amount of groups to be sampled
    criterion : str
        Name of criterion, default ttest, see list acceptable criteria
    threads : int
        Amount of threads used in thread pool

    Returns
    -------
    empirical_power : float
        Empirical power, calculated as frequecy of rejected hypotesis
    """
    if threads > 1:
        with parallel_backend(BOOTSTRAP_BACKEND, n_jobs=threads):
            exp_results = Parallel(verbose=False)(
                delayed(evaluate_criterion)(
                    dataframe=dataframe,
                    column=column,
                    effect=effect,
                    group_size=group_size,
                    alpha=first_error,
                    criterion=criterion,
                )
                for _ in range(bootstrap_size)
            )
    else:
        exp_results = []
        for _ in range(bootstrap_size):
            exp_results.append(
                evaluate_criterion(
                    dataframe=dataframe,
                    column=column,
                    effect=effect,
                    alpha=first_error,
                    group_size=group_size,
                    criterion=criterion,
                )
            )
    return np.mean(exp_results)


def get_table_power(
    dataframe: types.SparkDataFrame,
    metrics: Iterable[types.ColumnNameType],
    effects: Iterable[float],
    group_sizes: Iterable[int],
    alphas: Iterable[float],
    bootstrap_size: int = BOOSTRAP_BASE_CONST,
    threads: int = THREADS_BOOTSTRAP,
    use_tqdm: bool = True,
    as_numeric: bool = False,
) -> types.DesignerResult:
    """
    Calculate table of criterion empirical power with rows effects and columns sizes.

    Parameters
    ----------
    dataframe : Spark dataframe
        Table with column, which wiil be used
    metrics : Iterable of column type
        Iterable set of columns for designing
    effects : Iterable[float]
        List of effects which we want to check
    group_sizes : Iterable[int]
        List of group sizes which we want to check
    alphas : Iterable[float]
        1st type error bound, passed as list, for example [0.05]
    bootstrap_size: int, default: ``10``
        Amount of pairs of groups A/B to be sampled for estimation power
    threads : int
        Amount of threads for thread pool
    use_tqdm : bool
        Whether to use progress bar
    as_numeric : bool, default False
        Whether to return a number or a string with percentages

    Returns
    -------
    report : Union[pd.DataFrame, Dict[str, pd.DataFrame]
        Tables with sizes for group A, group B, effects, erros and metrics names
        Effects for indices
        Group sizes for columns
        table for each metric: dict[metric name] = table with mde(effects)
        or one table, if only one metric passed
    """
    if len(alphas) > 1:
        raise ValueError("For power table you can pass only one first error bound")
    result = pd.DataFrame(columns=group_sizes, index=effects)
    iterate_params = tqdm(zip(effects, group_sizes)) if use_tqdm else zip(effects, group_sizes)
    for column in metrics:
        for effect, group_size in iterate_params:
            power = calc_empiric_power(
                dataframe=dataframe,
                column=column,
                effect=effect,
                group_size=group_size,
                first_error=alphas[0],
                bootstrap_size=bootstrap_size,
                threads=threads,
            )
            if as_numeric:
                result.loc[effect, group_size] = power
            else:
                result.loc[effect, group_size] = (round(power * 100, ROUND_DIGITS_PERCENT)).astype(str) + "%"
    result.columns.name = "sample sizes"
    result.index.name = "effect"
    return result


def optimize_group_size(
    dataframe: types.SparkDataFrame,
    column: types.ColumnNameType,
    effect: float,
    beta: float,
    first_error: float = FIRST_TYPE_ERROR,
    bootstrap_size: int = BOOSTRAP_BASE_CONST,
    threads: int = THREADS_BOOTSTRAP,
) -> int:
    """
    Optimize group size for fixed effect and errors using empiric solution.
    Spark requests are made using thread pool.

    Parameters
    ----------
    dataframe : Spark table
        Table for designing experiment
    column : Column type
        Column, containg metric for designing
    effect : float
        Size of group both groups
    beta: float
        2nd type error bound
    first_error : float, default: ``(0.05,)``
        1st type error bound
    bootstrap_size : int, default: ``10``
        Amount of pairs of groups A/B to be sampled for estimation power
    threads : int
        Amount of threads for thread pool

    Returns
    -------
    optimal_size : int
        Groups sizes calculted via empiric power optimization
    """
    power: float = 1 - beta
    solver = solver_pkg.EmpiricSizeSolution(calc_empiric_power, power, ["group_size"])
    return solver.calc_binary(
        dataframe=dataframe,
        column=column,
        effect=effect,
        first_error=first_error,
        bootstrap_size=bootstrap_size,
        threads=threads,
    )


def get_table_size(
    dataframe: types.SparkDataFrame,
    metrics: Iterable[types.ColumnNameType],
    effects: Iterable[float],
    betas: Iterable[float],
    alphas: Iterable[float],
    bootstrap_size: int = BOOSTRAP_BASE_CONST,
    threads: int = THREADS_BOOTSTRAP,
    n_jobs: int = N_JOBS_MULTIPROCESS,
    use_tqdm: bool = True,
    as_numeric: bool = False,
) -> types.DesignerResult:
    """
    Find sizes by variating other params for many columns using thread pool for power estimation.

    Parameters
    ----------
    dataframe : Spark table
        Table for designing experiment
    metrics : Iterable of column type
        Iterable set of columns for designing
    effects : Iterable[float]
        List of group size which we want to check
    betas : Iterable[float]
        2nd type error bounds
    alpha : Iterable[float]
        1st type error bounds
    bootstrap_size : int, default: ``10``
        Amount of pairs of groups A/B to be sampled for estimation power
    threads : int
        Amount of threads for thread pool
    n_jobs : int
        Amount of jobs for metrics variating
    use_tqdm : bool
        Whether to use progress bar
    as_numeric : bool, default False
        Whether to return a number or a string with percentages

    Returns
    -------
    report : Union[pd.DataFrame, Dict[str, pd.DataFrame]
        Tables with sizes for group A, group B, effects, erros and metrics names
        Effects for indices
        (alpha(1 type error), beta(2 type error)) for columns
        table for each metric: dict[metric name] = table with mde(effects),
        or one table if one metric passed
    """
    params: Dict[str, Iterable[Any]] = {"effect": effects, "beta": betas, "first_error": alphas}
    results: Dict[types.ColumnNameType, pd.DataFrame] = {}
    for column_name in metrics:
        selector = select_pkg.Selector(
            optimize_group_size,
            params,
            n_jobs,
            use_tqdm,
            dataframe=dataframe,
            column=column_name,
            bootstrap_size=bootstrap_size,
            threads=threads,
        )
        results[column_name] = selector.get_table_size(as_numeric)
    return results[metrics[0]] if len(results) == 1 else results


def optimize_effect(
    dataframe: types.SparkDataFrame,
    column: types.ColumnNameType,
    group_size: int,
    beta: float,
    first_error: float = FIRST_TYPE_ERROR,
    bootstrap_size: int = BOOSTRAP_BASE_CONST,
    threads: int = THREADS_BOOTSTRAP,
) -> int:
    """
    Optimize effect for fixed size and errors using empiric solution.
    Spark requests are made using thread pool.

    Parameters
    ----------
    dataframe : Spark table
        Table for designing experiment
    column : Column type
        Column, containg metric for designing
    group_size : int
        Size of group both groups
    beta: float
        2nd type error bound
    first_error : float, default: ``(0.05,)``
        1st type error bound
    bootstrap_size : int, default: ``10``
        Amount of pairs of groups A/B to be sampled for estimation power
    threads : int
        Amount of threads for thread pool
    """
    power: float = 1 - beta
    solver = solver_pkg.EmpiricEffectSolution(calc_empiric_power, power, "effect")
    return solver.calc_binary(
        dataframe=dataframe,
        column=column,
        group_size=group_size,
        first_error=first_error,
        bootstrap_size=bootstrap_size,
        threads=threads,
    )


def get_table_effect(
    dataframe: types.SparkDataFrame,
    metrics: Iterable[types.ColumnNameType],
    group_sizes: Iterable[int],
    betas: Iterable[float],
    alphas: Iterable[float],
    bootstrap_size: int = BOOSTRAP_BASE_CONST,
    threads: int = THREADS_BOOTSTRAP,
    n_jobs: int = N_JOBS_MULTIPROCESS,
    use_tqdm: bool = True,
    as_numeric: bool = False,
) -> types.DesignerResult:
    """
    Find effects by variating other params for many columns using thread pool for power estimation.

    Parameters
    ----------
    dataframe : Spark table
        Table for designing experiment
    metrics : Iterable of column type
        Iterable set of columns for designing
    group_sizes : Iterable[int]
        List of group size which we want to check
    betas : Iterable[float]
        2nd type error bounds
    alphas : Iterable[float]
        1st type error bounds
    bootstrap_size : int, default: ``10``
        Amount of pairs of groups A/B to be sampled for estimation power
    threads : int
        Amount of threads for thread pool
    n_jobs : int
        Amount of jobs for metrics variating
    use_tqdm : bool
        Whether to use progress bar
    as_numeric : bool, default False
        Whether to return a number or a string with percentages

    Returns
    -------
    report : Union[pd.DataFrame, Dict[str, pd.DataFrame]
        Tables with sizes for group A, group B, effects, erros and metrics names
        Group sizes for indices
        (alpha(1 type error), beta(2 type error)) for columns
        table for each metric: dict[metric name] = table with mde(effects),
        or one table if one metric passed
    """
    params: Dict[str, Iterable[Any]] = {"group_size": group_sizes, "beta": betas, "first_error": alphas}
    results: Dict[types.ColumnNameType, pd.DataFrame] = {}
    for column_name in metrics:
        selector = select_pkg.Selector(
            optimize_effect,
            params,
            n_jobs,
            use_tqdm,
            dataframe=dataframe,
            column=column_name,
            bootstrap_size=bootstrap_size,
            threads=threads,
        )
        results[column_name] = selector.get_table_effect(as_numeric)
    return results[metrics[0]] if len(results) == 1 else results
