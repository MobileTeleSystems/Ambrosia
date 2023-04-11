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

from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe
from joblib import Parallel, delayed, parallel_backend

import ambrosia.tools._lib._tools_aide as aid_pkg
import ambrosia.tools.empirical_tools as emp_pkg
from ambrosia import types
from ambrosia.tools import back_tools

from . import EFFECT_COL_NAME, FIRST_TYPE_ERROR_COL_NAME, GROUP_SIZE_COL_NAME, STAT_ERRORS_COL_NAME

ROUND_DIGITS_TABLE: int = 3
ROUND_DIGITS_PERCENT: int = 1


def bootstrap_over_statistical_population(
    dataframe: pd.DataFrame,
    metrics: List,
    sample_size_a: int,
    sample_size_b: int,
    effect: float,
    alpha: float,
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    n_jobs: bool = 1,
    verbose: bool = False,
    **kwargs,
) -> List[Optional[float]]:
    """
    Evaluate errors of the experiment setup for each metric and set of parameters.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Given datframe.
    metrics : List
        List of metrics - columns of ``dataframe``.
    sample_size_a : int
        Size of group A.
    sample_size_b : int
        Size of group B.
    effect : float
        Expected effect in group B for the modeling.
    alpha : float
        First type error bound, 1 - alpha: correctness.
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion to apply.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed : int, optional
        A seed for the deterministic outcome of random processes.
    n_jobs : int, default: ``1``
        Amount of threads/workers for parallel.
    verbose : bool, default: ``False``
        Whether use logging.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    result_list : List[Optional[float]]
        List of calculated correctness/power for bootstraped groups
        [corretness_0/power_0, correctness_1/power_1 ... corretness_k/power_k]
    """
    seed_sequence: np.ndarray = back_tools.create_seed_sequence(len(metrics), random_seed)
    iterator = zip(metrics, seed_sequence)
    result_list = []
    for metric, seed in iterator:
        metric_vals = dataframe[metric].values.astype("float32")  ## to discuss
        sampled_metric_vals = np.random.default_rng(seed).choice(
            metric_vals, size=(sample_size_a + sample_size_b, bs_samples), replace=True
        )
        modified_metric_vals = emp_pkg.inject_effect(
            sampled_metric_vals, sample_size_a, effect, modeling_method=injection_method, random_seed=seed
        )
        power = emp_pkg.eval_error(
            modified_metric_vals,
            sample_size_a,
            alpha,
            mode=criterion,
            random_seed=seed,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs,
        )
        result_list.append(power)
    return result_list


def get_errors(
    dataframe: pd.DataFrame,
    metrics: List,
    sample_sizes_a: Iterable[int],
    sample_sizes_b: Iterable[int],
    effects: Iterable[float],
    alphas: Iterable[float],
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    n_jobs: int = 1,
    verbose: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Get I/II type errors estimation for the parameters of experiment.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Given datframe with data for experiment.
    metrics : List
        Columns of ``dataframe`` for calculating errors.
    sample_sizes_a : Iterable[int]
        List of A group sizes.
    sample_sizes_b : Iterable[int]
        List of B group sizes.
    effect : float
        Expected effect in group B for the modeling.
    alpha : float
        First type error bound, 1 - alpha: correctness.
    beta : float
        Second type error, 1 - beta - power of criterion.
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed : int, optional
        A seed for the deterministic outcome of random processes.
    n_jobs : int, default: ``1``
        Amount of threads/workers for parallel.
    verbose : bool, default: ``False``
        Whether use logging or not.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    parameters, empirical_errors : Tuple[Tuple, np.ndarray]
        Parameters - Cartesian product of all sample sizes, effects, alphas
        empirical_errors - Array of lists [corretness_0, power_0,
        correctness_1, power_1 ... corretness_k, power_k].
    """
    parameters = tuple(product(zip(sample_sizes_a, sample_sizes_b), effects, alphas))
    seed_sequence: np.ndarray = back_tools.create_seed_sequence(len(parameters), random_seed)
    iterator = zip(parameters, seed_sequence)
    handled_params: Dict = back_tools.handle_nested_multiprocessing(
        n_jobs,
        criterion,
        bootstrap_over_statistical_population,
        desc="Empirical errors calculation",
        total=len(parameters),
        **kwargs,
    )
    with handled_params["progress_bar"]:
        with parallel_backend(n_jobs=handled_params["n_jobs"], backend="loky"):
            empirical_errors = Parallel(verbose=verbose)(
                delayed(handled_params["parallel_func"])(
                    dataframe=dataframe,
                    metrics=metrics,
                    sample_size_a=params[0][0],
                    sample_size_b=params[0][1],
                    effect=params[1],
                    alpha=params[2],
                    bs_samples=bs_samples,
                    criterion=criterion,
                    injection_method=injection_method,
                    random_seed=seed,
                    n_jobs=handled_params["nested_n_jobs"],
                    verbose=verbose,
                    **handled_params["kwargs"],
                )
                for params, seed in iterator
            )
    return parameters, empirical_errors


def get_empirical_table_power(
    dataframe: pd.DataFrame,
    metrics: List,
    sample_sizes_a: Iterable[int],
    sample_sizes_b: Optional[Iterable[int]],
    effects: Iterable[int],
    alphas: Iterable[int],
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    n_jobs: int = 1,
    as_numeric: bool = False,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Create table with calculated power for given data and set of parameters.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Given datframe with data for experiment.
    metrics : str
        Columns of ``dataframe`` for calculating errors.
    sample_sizes_a : Iterable[int]
        List of A group sizes.
    sample_sizes_b : Optional[Iterable[int]]
        List of B group sizes, if None it will copy group A sizes.
    effects : Iterable[float]
        Expected effects in group B.
    alphas : Iterable[float]
        First type error bounds, 1 - alpha: correctness
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed : int, optional
        A seed for the deterministic outcome of random processes.
    parallel : bool, default: ``False``
        Whether to use parallel computing.
    n_jobs : int, default: ``1``
        Amount of threads/workers for parallel.
    as_numeric : bool, default False
        Whether to return a number or a string with percentages.
    verbose : bool, default: ``False``
        Whether use logging or not.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    report : pd.DataFrame
        Table with sizes for group A, group B, effects, erros and metrics names
        |A size | B size | effect | alpha | name_correctness| name_power| ...
    """
    if isinstance(sample_sizes_a, List):
        sample_sizes_a = np.array(sample_sizes_a)
    if sample_sizes_b is None:
        sample_sizes_b = np.copy(sample_sizes_a)
    elif isinstance(sample_sizes_b, List):
        sample_sizes_b = np.array(sample_sizes_b)
    parameters, emprical_errors = get_errors(
        dataframe=dataframe,
        metrics=metrics,
        sample_sizes_a=sample_sizes_a,
        sample_sizes_b=sample_sizes_b,
        effects=effects,
        alphas=alphas,
        bs_samples=bs_samples,
        criterion=criterion,
        injection_method=injection_method,
        random_seed=random_seed,
        n_jobs=n_jobs,
        verbose=verbose,
        **kwargs,
    )
    reports = {}
    for num, metric in enumerate(metrics):
        metric_errors: List = [errors_subset[num] for errors_subset in emprical_errors]
        report = pd.DataFrame(
            list(parameters), columns=[GROUP_SIZE_COL_NAME, EFFECT_COL_NAME, FIRST_TYPE_ERROR_COL_NAME]
        ).join(pd.DataFrame(metric_errors, columns=[STAT_ERRORS_COL_NAME]))
        report[EFFECT_COL_NAME] = (round((report[EFFECT_COL_NAME] - 1) * 100, ROUND_DIGITS_PERCENT)).astype(str) + "%"
        report = report.pivot_table(
            index=[FIRST_TYPE_ERROR_COL_NAME, EFFECT_COL_NAME],
            columns=GROUP_SIZE_COL_NAME,
            values=STAT_ERRORS_COL_NAME,
            sort=False,
        )
        report = report[list(zip(sample_sizes_a, sample_sizes_b))]
        if as_numeric:
            report = report.applymap(lambda x: round(x, ROUND_DIGITS_TABLE))
        else:
            report = report.applymap(lambda x: str(round(x * 100, ROUND_DIGITS_PERCENT)) + "%")
        if len(metrics) == 1:
            reports = report
        else:
            reports[metric] = report
    return reports


def optimize_group_size(
    dataframe: pd.DataFrame,
    metric: str,
    effect: float,
    alpha: float,
    power: float,
    groups_ratio: float = 1.0,
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    evals: int = 50,
    epsilon: float = 0.001,
    solution: str = "binary",
    n_jobs: int = 1,
    verbose: bool = False,
    **kwargs,
) -> int:
    """
    Find optimal group size for each metric on given parameters set.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Given datframe with data for experiment.
    metric : str
        Column of ``dataframe`` to calculate criterion.
    effect : float
        Expected effect in group B for the modeling.
    alpha : float
        First type error bound, 1 - alpha: correctness.
    power : float
        Desired power of criterion.
    groups_ratio : float, default: ``1.0``
        Ratio between groups A and B.
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed : int, optional
        A seed for the deterministic outcome of random processes.
    evals : int, default: ``50``
        Evals number for optimization.
    epsilon : float, default: ``0.001``
        Precision for power estimation solution.
    soltuion : str, default: ``"binary"``
        Optimizer method.
    n_jobs : int, default: ``1``
        Amount of threads/workers for parallel.
    verbose : bool, default: ``False``
        Whether use logging or not.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    optimal_group_size : int
        Optimal size for groups A/B to reach sufficient power and confidence
        levels.
    """
    solutions_names: List[str] = ["hyperopt", "binary"]
    power_calculation: Callable = bootstrap_over_statistical_population

    def objective(params: Dict) -> float:
        group_size_a = int(params["group_size"])
        group_size_b = int(groups_ratio * params["group_size"])
        power_emp = power_calculation(
            dataframe=dataframe,
            metrics=[metric],
            sample_size_a=group_size_a,
            sample_size_b=group_size_b,
            effect=effect,
            alpha=alpha,
            bs_samples=bs_samples,
            criterion=criterion,
            injection_method=injection_method,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs,
        )
        delta = abs(power_emp - (power + epsilon))
        return delta

    upper_bound_degree: int = aid_pkg.helper_bin_search_upper_bound_size(
        power_calculation,
        power,
        ["sample_size_a", "sample_size_b"],
        groups_ratio,
        dataframe=dataframe,
        metrics=[metric],
        effect=effect,
        alpha=alpha,
        bs_samples=bs_samples,
        criterion=criterion,
        injection_method=injection_method,
        random_seed=random_seed,
        n_jobs=n_jobs,
        verbose=verbose,
        **kwargs,
    )
    if solution == "hyperopt":
        lower_bound_degree: int = max(0, upper_bound_degree // 2)
        # log(2) for reduction to binary logarithm
        space = {
            "group_size": hp.qloguniform(
                "group_size", lower_bound_degree * np.log(2), upper_bound_degree * np.log(2), 1
            )
        }
        best = fmin(objective, space, algo=tpe.suggest, max_evals=evals, verbose=False)
        optimal_group_size = int(best["group_size"])
    elif solution == "binary":
        upper_bound: int = 2**upper_bound_degree
        total_size: int = int((1 + groups_ratio) * upper_bound)
        bootstraped_samples: types.BootstrapedSamplesType = aid_pkg.__helper_generate_bootstrap_samples(
            dataframe=dataframe,
            metrics=[metric],
            total_size=total_size,
            bootstrap_size=bs_samples,
            random_seed=random_seed,
        )

        modified_samples: types.BootstrapedSamplesType = aid_pkg.__helper_inject_effect(
            sampled_metrics=bootstraped_samples,
            sample_size_a=upper_bound,
            effect=effect,
            injection_method=injection_method,
            random_seed=random_seed,
        )
        optimal_group_size: int = aid_pkg.helper_binary_search_optimal_size(
            aid_pkg.__helper_get_power_for_bootstraped,
            power,
            upper_bound,
            ["sample_size"],
            modified_samples=modified_samples,
            bound_size=upper_bound,
            alpha=alpha,
            groups_ratio=groups_ratio,
            criterion=criterion,
            random_seed=random_seed,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs,
        )
    else:
        raise ValueError(f'Choose soltuion from {", ".join(solutions_names)}')
    return optimal_group_size


def calculate_group_size(
    dataframe: pd.DataFrame,
    metrics: List,
    effect: float,
    alpha: float,
    beta: float,
    groups_ratio: float = 1.0,
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    evals: int = 50,
    optim_solution: str = "binary",
    n_jobs: int = 1,
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Calculate optimal group_size for each metric on given set of parameters.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Given datframe with data for experiment.
    metrics : List
        Columns of ``dataframe`` for calculating sufficient group sizes.
    effect : float
        Expected effect in group B for the modeling.
    alpha : float
        First type error bound, 1 - alpha: correctness.
    beta : float
        Second type error, 1 - beta - power of criterion.
    groups_ratio : float, default: ``1.0``
        Ratio between groups A and B.
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed : int, optional
        A seed for the deterministic outcome of random processes.
    evals : int, default: ``50``
        Evals number for optimization.
    optim_soltuion : str, default: ``"binary"``
        Optimizer method.
    n_jobs : int, default: ``1``
        Amount of threads/workers for parallel.
    verbose : bool, default: ``False``
        Whether use logging or not.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    group_sizes : np.ndarray
        Optimal group size for each metric from metrics list
    """
    power: float = 1 - beta
    seed_sequence: np.ndarray = back_tools.create_seed_sequence(len(metrics), random_seed)
    iterator = zip(metrics, seed_sequence)
    group_sizes: List = []
    for metric, seed in iterator:
        optimal_group_size = optimize_group_size(
            dataframe=dataframe,
            metric=metric,
            effect=effect,
            alpha=alpha,
            power=power,
            groups_ratio=groups_ratio,
            bs_samples=bs_samples,
            criterion=criterion,
            injection_method=injection_method,
            random_seed=seed,
            evals=evals,
            solution=optim_solution,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs,
        )
        group_sizes.append(optimal_group_size)
    return np.array(group_sizes)


def get_group_sizes(
    dataframe: pd.DataFrame,
    metrics: List,
    effects: Iterable[float],
    alphas: Iterable[float],
    betas: Iterable[float],
    groups_ratio: float = 1.0,
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    evals: int = 50,
    optim_solution: str = "binary",
    n_jobs: int = 1,
    verbose: bool = False,
    **kwargs,
) -> Tuple[Tuple, List[np.ndarray]]:
    """
    Get optimal group sizes for the possible parameters sets of the experiment.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Given datframe with data for experiment.
    metrics : List
        Columns of ``dataframe`` for calculating sufficient group sizes.
    effects : Iterable[float]
        Expected effects in group B for the modeling.
    alphas : Iterable[float]
        First type error bounds, 1 - alpha: correctness.
    betas : Iterable[float]
        Second type errors, 1 - beta = power.
    groups_ratio : float, default: ``1.0``
        Ratio between groups A and B.
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed : int, optional
        A seed for the deterministic outcome of random processes.
    evals : int, default: ``50``
        Evals number for optimization.
    optim_soltuion : str, default: ``"binary"``
        Optimizer method.
    n_jobs : int, default: ``1``
        Amount of threads/workers for parallel.
    verbose : bool, default: ``False``
        Whether use logging or not.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    parameters, group_sizes_list : Tuple[Tuple, Tuple[np.ndarray]]
        Parameters - Cartesian product of all sample sizes, effects, alphas
        group_sizes_list - Array of lists of sizes for each metric from metrics list.
    """
    parameters: Tuple = tuple(product(effects, alphas, betas))
    seed_sequence: np.ndarray = back_tools.create_seed_sequence(len(parameters), random_seed)
    iterator = zip(parameters, seed_sequence)
    handled_params: Dict = back_tools.handle_nested_multiprocessing(
        n_jobs, criterion, calculate_group_size, desc="Group sizes calculation", total=len(parameters), **kwargs
    )
    with handled_params["progress_bar"]:
        with parallel_backend(n_jobs=handled_params["n_jobs"], backend="loky"):
            group_sizes_list = Parallel(verbose=verbose)(
                delayed(handled_params["parallel_func"])(
                    dataframe=dataframe,
                    metrics=metrics,
                    effect=params[0],
                    alpha=params[1],
                    beta=params[2],
                    groups_ratio=groups_ratio,
                    criterion=criterion,
                    bs_samples=bs_samples,
                    injection_method=injection_method,
                    n_jobs=handled_params["nested_n_jobs"],
                    verbose=verbose,
                    evals=evals,
                    optim_solution=optim_solution,
                    random_seed=seed,
                    **handled_params["kwargs"],
                )
                for params, seed in iterator
            )
    return parameters, group_sizes_list


def get_empirical_table_sample_size(
    dataframe: pd.DataFrame,
    metrics: List,
    effects: Iterable[float],
    alphas: Iterable[float],
    betas: Iterable[float],
    groups_ratio: float = 1.0,
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    evals: int = 50,
    optim_solution: str = "binary",
    n_jobs: int = 1,
    verbose: bool = False,
    **kwargs,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Create pandas table with optimal group sizes for given data and set of parameters.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Given datframe with data for experiment.
    metrics : List
        Columns of ``dataframe`` for calculating sufficient group sizes.
    effects : Iterable[float]
        Expected effects in group B.
    alphas : Iterable[float]
        First type error bounds, 1 - alpha: correctness.
    betas : Iterable[float]
        Second type errors, 1 - beta = power.
    groups_ratio : float, default: ``1.0``
        Ratio between groups A and B.
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed : int, optional
        A seed for the deterministic outcome of random processes.
    evals : int, default: ``50``
        Evals amount for optimization.
    optim_solution : str, default: ``"binary"``
        Optimizer method.
    n_jobs : int, default: ``1``
        Amount of threads/workers for parallel.
    verbose : bool, default: ``False``
        Whether use logging or not.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    report : Union[pd.DataFrame, Dict[str, pd.DataFrame]
        Tables with sizes for group A, group B, effects, errors and metrics names
        Effects as indices
        (alpha(1 type error), beta(2 type error)) for columns
        table for each metric: dict[metric name] = table with sizes.
    """
    parameters, group_sizes_list = get_group_sizes(
        dataframe=dataframe,
        metrics=metrics,
        effects=effects,
        alphas=alphas,
        betas=betas,
        groups_ratio=groups_ratio,
        bs_samples=bs_samples,
        criterion=criterion,
        injection_method=injection_method,
        random_seed=random_seed,
        evals=evals,
        optim_solution=optim_solution,
        n_jobs=n_jobs,
        verbose=verbose,
        **kwargs,
    )
    reports = {}
    for num, metric in enumerate(metrics):
        report = pd.DataFrame(list(parameters), columns=[EFFECT_COL_NAME, "alpha", "beta"]).join(
            pd.DataFrame(list(np.array(group_sizes_list)[:, num]), columns=[GROUP_SIZE_COL_NAME])
        )
        report[STAT_ERRORS_COL_NAME] = tuple(zip(report["alpha"], report["beta"]))
        report = report.pivot_table(
            index=EFFECT_COL_NAME, columns=STAT_ERRORS_COL_NAME, values=GROUP_SIZE_COL_NAME, sort=False
        )
        report.index = (np.round((report.index - 1) * 100, ROUND_DIGITS_PERCENT)).astype(str) + "%"
        if len(metrics) == 1:
            reports = report
        else:
            reports[metric] = report
    return reports


def optimize_mde(
    dataframe: pd.DataFrame,
    metric: str,
    group_size: int,
    alpha: float = 0.05,
    power: float = 0.8,
    groups_ratio: float = 1.0,
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    evals: int = 50,
    solution: str = "binary",
    n_jobs: int = 1,
    verbose: bool = False,
    **kwargs,
) -> float:
    """
    Find minimum detectable effect for each metric on given parameters set.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Given datframe with data for experiment.
    metric : str
        Column of ``dataframe`` to calculate criterion.
    group_size : int
        Size for group A/B.
    alpha : float
        First type error bound, 1 - alpha: correctness.
    power : float
        Desired power of criterion.
    groups_ratio : float, default: ``1.0``
        Ratio between groups A and B.
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed : int, optional
        A seed for the deterministic outcome of random processes.
    evals : int, default: ``50``
        Evals amount for optimization.
    solution : str, default: ``"binary"``
        Optimizer method.
    n_jobs : int, default: ``1``
        Amount of threads/workers for parallel.
    verbose : bool, default: ``False``
        Whether use logging or not.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    mde : float
        Minimal detectable effect for given power and group size.
    """
    solutions_names: List[str] = ["hyperopt", "binary"]
    power_calculation: Callable = bootstrap_over_statistical_population
    sample_size_a, sample_size_b = group_size, int(groups_ratio * group_size)

    def objective(params):
        effect = params["effect"]
        power_emp = power_calculation(
            dataframe=dataframe,
            metrics=[metric],
            sample_size_a=sample_size_a,
            sample_size_b=sample_size_b,
            effect=effect,
            alpha=alpha,
            bs_samples=bs_samples,
            criterion=criterion,
            injection_method=injection_method,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs,
        )
        if power_emp > power:
            val = effect
        else:
            # Some value more than right bound
            val = np.inf
        return val

    upper_bound_effect: float = 2 ** aid_pkg.helper_bin_searh_upper_bound_effect(
        power_calculation,
        power,
        dataframe=dataframe,
        metrics=[metric],
        alpha=alpha,
        sample_size_a=sample_size_a,
        sample_size_b=sample_size_b,
        bs_samples=bs_samples,
        criterion=criterion,
        injection_method=injection_method,
        random_seed=random_seed,
        n_jobs=n_jobs,
        verbose=verbose,
        **kwargs,
    )
    if solution == "hyperopt":
        space = {"effect": hp.uniform("effect", 1, upper_bound_effect)}
        best = fmin(objective, space, algo=tpe.suggest, max_evals=evals, verbose=False)
        optimal_effect: float = best["effect"]

    elif solution == "binary":
        bootstraped_samples: types.BootstrapedSamplesType = aid_pkg.__helper_generate_bootstrap_samples(
            dataframe=dataframe,
            metrics=[metric],
            total_size=sample_size_a + sample_size_b,
            bootstrap_size=bs_samples,
            random_seed=random_seed,
        )
        optimal_effect: int = aid_pkg.helper_binary_search_optimal_effect(
            aid_pkg.__helper_get_power_for_bootstraped,
            power,
            upper_bound_effect,
            bootstraped_samples,
            injection_method,
            bound_size=group_size,
            sample_size=group_size,
            alpha=alpha,
            groups_ratio=groups_ratio,
            criterion=criterion,
            random_seed=random_seed,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs,
        )
    else:
        raise ValueError(f'Choose soltuion from {", ".join(solutions_names)}')
    return optimal_effect


def calculate_empirical_mde(
    dataframe: pd.DataFrame,
    metrics: List,
    group_size: int,
    alpha: float,
    beta: float,
    groups_ratio: float = 1.0,
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    evals: int = 50,
    optim_solution: str = "binary",
    n_jobs: int = 1,
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Calculate empirical MDE for each metric on given set of parameters.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Given datframe with data for experiment.
    metric : str
        Column of ``dataframe`` to calculate criterion.
    group_size : int
        Size for group A/B.
    alpha : float
        First type error bound, 1 - alpha: correctness.
    beta : float
        Second type error, 1 - beta - power of criterion.
    groups_ratio : float, default: ``1.0``
        Ratio between groups A and B.
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed : int, optional
        A seed for the deterministic outcome of random processes.
    evals : int, default: ``50``
        Evals amount for optimization.
    optim_solution : str, default: ``"binary"``
        Optimizer method.
    n_jobs : int, default: ``1``
        Amount of threads/workers for parallel.
    verbose : bool, default: ``False``
        Whether use logging or not.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    mdes : np.ndarray
        List of minimal detectable effect for each metric from metrics list.
    """
    power: float = 1 - beta
    seed_sequence: np.ndarray = back_tools.create_seed_sequence(len(metrics), random_seed)
    iterator = zip(metrics, seed_sequence)
    mdes: List = []
    for metric, seed in iterator:
        mde: float = optimize_mde(
            dataframe=dataframe,
            metric=metric,
            group_size=group_size,
            alpha=alpha,
            power=power,
            groups_ratio=groups_ratio,
            bs_samples=bs_samples,
            criterion=criterion,
            injection_method=injection_method,
            random_seed=seed,
            evals=evals,
            solution=optim_solution,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs,
        )
        mdes.append(mde)
    return mdes


def get_empirical_mde(
    dataframe: pd.DataFrame,
    metrics: List,
    group_sizes: Iterable[int],
    alphas: Iterable[float],
    betas: Iterable[float],
    groups_ratio: float = 1.0,
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    evals: int = 50,
    optim_solution: str = "binary",
    n_jobs: int = 1,
    verbose: bool = False,
    **kwargs,
) -> Tuple[Tuple, List[np.ndarray]]:
    """
    Get empirical MDEs for the possible parameters sets of experiment.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Given datframe with data for experiment.
    metrics : str
        Column of ``dataframe`` to calculate minimal detectable effect.
    group_sizes : Iterable[int]
        Sizes for groups A/B.
    alphas : Iterable[float]
        First type error bounds, 1 - alpha: correctness
    betas : Iterable[float]
        Second type erros, 1 - beta = power.
    groups_ratio : float, default: ``1.0``
        Ratio between groups A and B.
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed : int, optional
        A seed for the deterministic outcome of random processes.
    evals : int, default: ``50``
        Evals amount for optimization.
    optim_solution : str, default: ``"binary"``
        Optimizer method.
    n_jobs : int, default: ``1``
        Amount of threads/workers for parallel.
    verbose : bool, default: ``False``
        Whether use logging or not.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    parameters, empirical_errors : Tuple[Tuple, np.ndarray]
        Parameters - Cartesian product of all sample sizes, effects, alphas
        mde_list - Array of lists of mde lists for each metric from metrics list.
    """
    parameters: Tuple = tuple(product(group_sizes, alphas, betas))
    seed_sequence: np.ndarray = back_tools.create_seed_sequence(len(parameters), random_seed)
    iterator = zip(parameters, seed_sequence)
    handled_params: Dict = back_tools.handle_nested_multiprocessing(
        n_jobs, criterion, calculate_empirical_mde, desc="MDE calculation", total=len(parameters), **kwargs
    )
    with handled_params["progress_bar"]:
        with parallel_backend(n_jobs=handled_params["n_jobs"], backend="loky"):
            mde_list = Parallel(verbose=verbose)(
                delayed(handled_params["parallel_func"])(
                    dataframe=dataframe,
                    metrics=metrics,
                    group_size=params[0],
                    alpha=params[1],
                    beta=params[2],
                    groups_ratio=groups_ratio,
                    criterion=criterion,
                    bs_samples=bs_samples,
                    injection_method=injection_method,
                    random_seed=seed,
                    evals=evals,
                    optim_solution=optim_solution,
                    n_jobs=handled_params["nested_n_jobs"],
                    verbose=verbose,
                    **handled_params["kwargs"],
                )
                for params, seed in iterator
            )
    return parameters, mde_list


def get_empirical_mde_table(
    dataframe: pd.DataFrame,
    metrics: List,
    group_sizes: Iterable[int],
    alphas: Iterable[float],
    betas: Iterable[float],
    groups_ratio: float = 1.0,
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    evals: int = 50,
    optim_solution: str = "binary",
    n_jobs: int = 1,
    as_numeric: bool = False,
    verbose: bool = False,
    **kwargs,
) -> Union[pd.DataFrame, Dict]:
    """
    Create pandas table with MDEs for given data and set of parameters.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Given datframe with data for experiment.
    metrics : List
        Columns of ``dataframe`` for calculating sufficient group sizes.
    group_sizes : Iterable[int]
        Sizes for groups A/B.
    alphas : Iterable[float]
        First type error bounds, 1 - alpha: correctness.
    betas : Iterable[float]
        Second type errors, 1 - beta = power.
    groups_ratio : float, default: ``1.0``
        Ratio between groups A and B.
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed : int, optional
        A seed for the deterministic outcome of random processes.
    evals : int, default: ``50``
        Evals amount for optimization.
    optim_solution : str, default: ``"binary"``
        Optimizer method.
    n_jobs : int, default: ``1``
        Amount of threads/workers for parallel.
    as_numeric : bool, default False
        Whether to return a number or a string with percentages.
    verbose : bool, default: ``False``
        Whether use logging or not.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    report: Union[pd.DataFrame, Dict[str, pd.DataFrame]
        Tables with sizes for group A, group B, effects, erros and metrics names
        Group sizes for indices
        (alpha(1 type error), beta(2 type error)) for columns
        table for each metric: dict[metric name] = table with mde(effects).
    """
    parameters, effects = get_empirical_mde(
        dataframe=dataframe,
        metrics=metrics,
        group_sizes=group_sizes,
        alphas=alphas,
        betas=betas,
        groups_ratio=groups_ratio,
        bs_samples=bs_samples,
        criterion=criterion,
        injection_method=injection_method,
        random_seed=random_seed,
        evals=evals,
        optim_solution=optim_solution,
        n_jobs=n_jobs,
        verbose=verbose,
        **kwargs,
    )
    reports = {}
    for num, metric in enumerate(metrics):
        report = pd.DataFrame(list(parameters), columns=[GROUP_SIZE_COL_NAME, "alpha", "beta"]).join(
            pd.DataFrame(list(np.array(effects)[:, num]), columns=[EFFECT_COL_NAME])
        )
        report[STAT_ERRORS_COL_NAME] = tuple(zip(report["alpha"], report["beta"]))
        report = report.pivot_table(
            index=GROUP_SIZE_COL_NAME, columns=STAT_ERRORS_COL_NAME, values=EFFECT_COL_NAME, sort=False
        )
        if as_numeric:
            report = report.applymap(lambda x: round(x, ROUND_DIGITS_TABLE))
        else:
            report = report.applymap(lambda x: str(round((x - 1) * 100, ROUND_DIGITS_PERCENT)) + "%")
        report = report[[(alpha, beta) for alpha in alphas for beta in betas]]
        if len(metrics) == 1:
            reports = report
        else:
            reports[metric] = report
    return reports
