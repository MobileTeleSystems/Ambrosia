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
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import ambrosia.tools._lib._tools_aide as aid_pkg
import ambrosia.tools.empirical_tools as emp_pkg
from ambrosia import types
from ambrosia.tools import back_tools

ROUND_DIGITS_TABLE: int = 3
ROUND_DIGITS_PERCENT: int = 1


def eval_error(
    sampled_metric_vals: np.ndarray,
    sample_size_a: int,
    alpha: float,
    mode: str = "ttest",
    stat: str = "mean",
    bootstrap_size: int = 1000,
    random_seed: Optional[int] = None,
    parallel: bool = False,
    use_tqdm: bool = False,
    verbose: bool = False,
) -> float:
    """
    Evaluate correctness/II type error of the experiment.

    Parameters
    ----------
    sampled_metric_vals : np.ndarray
        Samples of A/B groups: |group A values|group B values|.
    sample_size_a : int
        Size of  the group A in ``sampled_metric_vals``, i.e.
        first ``sample_size_a`` elements correspond to the group A.
    alpha : float
        First type error bound, 1 - alpha: correctness.
    mode : str, default: ``"ttest"``
        Statistical criterion, for example ``'ttest'``.
    stat : str, default: ``mean``
        Statistic to be calculated for sample groups during a bootstrap.
    bootstrap_size : int, default: ``1000``
        Number of bootstrap of A/B pairs.
    random_seed: int, optional
        A seed for the deterministic outcome of random processes.
    parallel : bool, default: ``False``
        Use parallel computations.
    use_tqdm : bool, default: ``False``
        Use tqdm progress bar.
    verbose : bool, default: ``False``
        Whether use logging.

    Returns
    -------
    error : float
        Second type error estimation or correctness, i.e.
        1 - P_{A=B} (criterion is completed) - correctness
        P_{A!=B} (criterion is completed) - second type error.
    """
    not_bootstrap_criteria: List[str] = ["ttest", "ttest_rel", "mw", "shapiro"]
    if mode in not_bootstrap_criteria:
        power: float = emp_pkg.stat_criterion_power(sampled_metric_vals, sample_size_a, criterion=mode, alpha=alpha)
    elif mode == "bootstrap":
        power: float = emp_pkg.make_bootstrap(
            sampled_metric_vals,
            sample_size_a,
            alpha,
            N=bootstrap_size,
            stat=stat,
            random_seed=random_seed,
            use_tqdm=use_tqdm,
            parallel=parallel,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Criterion {mode} is not found, choose from {not_bootstrap_criteria} or 'bootstrap'")
    return 1 - power


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
    calculate_alpha: bool = False,
    use_tqdm: bool = False,
    parallel: bool = False,
    verbose: bool = False,
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
    random_seed: int, optional
        A seed for the deterministic outcome of random processes.
    calculate_alpha : bool, default: ``False``
        Whether calculate correctness of criterion or not.
    use_tqdm : bool, default: ``False``
        Use tqdm progress bar.
    parallel : bool, default: ``False``
        Whether to use parallel computing.
    verbose : bool, default: ``False``
        Whether use logging.

    Returns
    -------
    result_list : List[Optional[float]]
        List of calculated correctness/power for bootstraped groups
        [corretness_0, power_0, correctness_1, power_1 ... corretness_k, power_k]
        If not calculate_alpha -> corretness_i = None.
    """
    seed_sequence: np.ndarray = emp_pkg.create_seed_sequence(len(metrics), random_seed)
    iterator = zip(metrics, seed_sequence)
    result_list = []
    for metric, seed in iterator:
        metric_vals = dataframe[metric].values
        sampled_metric_vals = np.random.default_rng(seed).choice(
            metric_vals, size=(sample_size_a + sample_size_b, bs_samples), replace=True
        )
        modified_metric_vals = emp_pkg.inject_effect(
            sampled_metric_vals, sample_size_a, effect, modeling_method=injection_method, random_seed=seed
        )
        if calculate_alpha:
            correctness = eval_error(
                sampled_metric_vals,
                sample_size_a,
                alpha,
                mode=criterion,
                random_seed=seed,
                use_tqdm=use_tqdm,
                parallel=parallel,
                verbose=verbose,
            )
        else:
            correctness = None
        power = 1 - eval_error(
            modified_metric_vals,
            sample_size_a,
            alpha,
            mode=criterion,
            random_seed=seed,
            use_tqdm=use_tqdm,
            parallel=parallel,
            verbose=verbose,
        )
        result_list.append(correctness)
        result_list.append(power)
    return result_list


def optimize_group_size(
    dataframe: pd.DataFrame,
    metric: str,
    effect: float,
    alpha: float,
    power: float,
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    calculate_alpha: bool = False,
    evals: int = 50,
    epsilon: float = 0.001,
    solution: str = "binary",
    use_tqdm: bool = False,
    parallel: bool = True,
    verbose: bool = False,
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
        Desired power of criterion
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed: int, optional
        A seed for the deterministic outcome of random processes.
    calculate_alpha : bool, default: ``False``
        Whether calculate correctness of criterion.
    evals : int, default: ``50``
        Evals number for optimization.
    epsilon : float, default: ``0.001``
        Precision for power estimation solution.
    soltuion : str, default: ``"binary"``
        Optimizer method.
    use_tqdm : bool, default: ``False``
        Use tqdm progress bar.
    parallel : bool, default: ``False``
        Whether to use parallel computing.
    verbose : bool, default: ``False``
        Whether use logging or not.

    Returns
    -------
    optimal_group_size : int
        Optimal size for groups A/B to reach sufficient power and confidence
        levels.
    """
    solutions_names: List[str] = ["hyperopt", "binary"]
    power_calculation: Callable = bootstrap_over_statistical_population

    def objective(params: Dict) -> float:
        group_size = int(params["group_size"])
        power_emp = power_calculation(
            dataframe=dataframe,
            metrics=[metric],
            sample_size_a=group_size,
            sample_size_b=group_size,
            effect=effect,
            alpha=alpha,
            bs_samples=bs_samples,
            criterion=criterion,
            injection_method=injection_method,
            use_tqdm=use_tqdm,
            parallel=parallel,
            verbose=verbose,
            calculate_alpha=calculate_alpha,
        )[1]
        delta = abs(power_emp - (power + epsilon))
        return delta

    upper_bound_degree: int = aid_pkg.helper_bin_search_upper_bound_size(
        power_calculation,
        power,
        ["sample_size_a", "sample_size_b"],
        dataframe=dataframe,
        metrics=[metric],
        effect=effect,
        alpha=alpha,
        bs_samples=bs_samples,
        criterion=criterion,
        injection_method=injection_method,
        random_seed=random_seed,
        use_tqdm=use_tqdm,
        parallel=parallel,
        verbose=verbose,
        calculate_alpha=calculate_alpha,
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
        bootstraped_samples: types.BootstrapedSamplesType = aid_pkg.__helper_generate_bootstrap_samples(
            dataframe=dataframe,
            metrics=[metric],
            total_size=upper_bound * 2,
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
            criterion=criterion,
            random_seed=random_seed,
            use_tqdm=use_tqdm,
            parallel=parallel,
            verbose=verbose,
        )
    else:
        raise ValueError(f'Choose soltuion from {", ".join(solutions_names)}')
    return optimal_group_size


def optimize_mde(
    dataframe: pd.DataFrame,
    metric: str,
    group_size: int,
    alpha: float = 0.05,
    power: float = 0.8,
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    calculate_alpha: bool = False,
    evals: int = 50,
    solution: str = "binary",
    use_tqdm: bool = False,
    parallel: bool = True,
    verbose: bool = False,
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
        Desired power of criterion
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed: int, optional
        A seed for the deterministic outcome of random processes.
    calculate_alpha : bool, default: ``False``
        Whether calculate correctness of criterion or not.
    evals : int, default: ``50``
        Evals amount for optimization.
    solution : str, default: ``"binary"``
        Optimizer method.
    use_tqdm : bool, default: ``False``
        Use tqdm progress bar.
    parallel : bool, default: ``False``
        Whether to use parallel computing.
    verbose : bool, default: ``False``
        Whether use logging or not.

    Returns
    -------
    mde : float
        Minimal detectable effect for given power and group size.
    """
    solutions_names: List[str] = ["hyperopt", "binary"]
    power_calculation: Callable = bootstrap_over_statistical_population

    def objective(params):
        effect = params["effect"]
        power_emp = bootstrap_over_statistical_population(
            dataframe=dataframe,
            metrics=[metric],
            sample_size_a=group_size,
            sample_size_b=group_size,
            effect=effect,
            alpha=alpha,
            bs_samples=bs_samples,
            criterion=criterion,
            injection_method=injection_method,
            calculate_alpha=calculate_alpha,
            use_tqdm=use_tqdm,
            parallel=parallel,
            verbose=verbose,
        )[1]
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
        sample_size_a=group_size,
        sample_size_b=group_size,
        bs_samples=bs_samples,
        criterion=criterion,
        injection_method=injection_method,
        random_seed=random_seed,
        calculate_alpha=calculate_alpha,
        use_tqdm=use_tqdm,
        parallel=parallel,
        verbose=verbose,
    )
    if solution == "hyperopt":
        space = {"effect": hp.uniform("effect", 1, upper_bound_effect)}
        best = fmin(objective, space, algo=tpe.suggest, max_evals=evals, verbose=False)
        optimal_effect: float = best["effect"]

    elif solution == "binary":
        bootstraped_samples: types.BootstrapedSamplesType = aid_pkg.__helper_generate_bootstrap_samples(
            dataframe=dataframe,
            metrics=[metric],
            total_size=group_size * 2,
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
            criterion=criterion,
            random_seed=random_seed,
            use_tqdm=use_tqdm,
            parallel=parallel,
            verbose=verbose,
        )
    else:
        raise ValueError(f'Choose soltuion from {", ".join(solutions_names)}')
    return optimal_effect


def calculate_group_size(
    dataframe: pd.DataFrame,
    metrics: List,
    effect: float,
    alpha: float,
    beta: float,
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    calculate_alpha: bool = False,
    evals: int = 50,
    optim_solution: str = "binary",
    use_tqdm: bool = False,
    parallel: bool = True,
    verbose: bool = False,
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
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed: int, optional
        A seed for the deterministic outcome of random processes.
    calculate_alpha : bool, default: ``False``
        Whether calculate correctness of criterion.
    evals : int, default: ``50``
        Evals number for optimization.
    optim_soltuion : str, default: ``"binary"``
        Optimizer method.
    use_tqdm : bool, default: ``False``
        Use tqdm progress bar.
    parallel : bool, default: ``False``
        Whether to use parallel computing.
    verbose : bool, default: ``False``
        Whether use logging or not.

    Returns
    -------
    group_sizes : np.ndarray
        Optimal group size for each metric from metrics list
    """
    power: float = 1 - beta
    seed_sequence: np.ndarray = emp_pkg.create_seed_sequence(len(metrics), random_seed)
    iterator = zip(metrics, seed_sequence)
    group_sizes: List = []
    for metric, seed in iterator:
        optimal_group_size = optimize_group_size(
            dataframe=dataframe,
            metric=metric,
            effect=effect,
            alpha=alpha,
            power=power,
            bs_samples=bs_samples,
            criterion=criterion,
            injection_method=injection_method,
            random_seed=seed,
            calculate_alpha=calculate_alpha,
            evals=evals,
            solution=optim_solution,
            use_tqdm=use_tqdm,
            parallel=parallel,
            verbose=verbose,
        )
        group_sizes.append(optimal_group_size)
    return np.array(group_sizes)


def calculate_empirical_mde(
    dataframe: pd.DataFrame,
    metrics: List,
    group_size: int,
    alpha: float,
    beta: float,
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    calculate_alpha: bool = False,
    evals: int = 50,
    optim_solution: str = "binary",
    use_tqdm: bool = False,
    parallel: bool = False,
    verbose: bool = False,
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
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed: int, optional
        A seed for the deterministic outcome of random processes.
    calculate_alpha : bool, default: ``False``
        Whether calculate correctness of criterion or not.
    evals : int, default: ``50``
        Evals amount for optimization.
    optim_solution : str, default: ``"binary"``
        Optimizer method.
    use_tqdm : bool, default: ``False``
        Use tqdm progress bar.
    parallel : bool, default: ``False``
        Whether to use parallel computing.
    verbose : bool, default: ``False``
        Whether use logging or not.

    Returns
    -------
    mdes : np.ndarray
        List of minimal detectable effect for each metric from metrics list.
    """
    power: float = 1 - beta
    seed_sequence: np.ndarray = emp_pkg.create_seed_sequence(len(metrics), random_seed)
    iterator = zip(metrics, seed_sequence)
    mdes: List = []
    for metric, seed in iterator:
        mde: float = optimize_mde(
            dataframe=dataframe,
            metric=metric,
            group_size=group_size,
            alpha=alpha,
            power=power,
            bs_samples=bs_samples,
            criterion=criterion,
            injection_method=injection_method,
            random_seed=seed,
            calculate_alpha=calculate_alpha,
            evals=evals,
            solution=optim_solution,
            use_tqdm=use_tqdm,
            parallel=parallel,
            verbose=verbose,
        )
        mdes.append(mde)
    return mdes


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
    use_tqdm: bool = False,
    parallel: bool = False,
    n_jobs: int = 8,
    verbose: bool = False,
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
    random_seed: int, optional
        A seed for the deterministic outcome of random processes.
    use_tqdm : bool, default: ``False``
        Use tqdm progress bar.
    parallel : bool, default: ``False``
        Whether to use parallel computing.
    n_jobs : int, default: ``8``
        Amount of threads/workers for parallel.
    verbose : bool, default: ``False``
        Whether use logging or not.

    Returns
    -------
    parameters, empirical_errors : Tuple[Tuple, np.ndarray]
        Parameters - Cartesian product of all sample sizes, effects, alphas
        empirical_errors - Array of lists [corretness_0, power_0,
        correctness_1, power_1 ... corretness_k, power_k].
    """
    parameters: Tuple = tuple(product(sample_sizes_a, sample_sizes_b, effects, alphas))
    seed_sequence: np.ndarray = emp_pkg.create_seed_sequence(len(parameters), random_seed)
    iterator = zip(parameters, seed_sequence)
    if parallel:
        with back_tools.tqdm_joblib(tqdm(desc="Empirical errors calculation", total=len(parameters))):
            empirical_errors = Parallel(n_jobs=n_jobs, verbose=verbose, backend="multiprocessing")(
                delayed(bootstrap_over_statistical_population)(
                    dataframe=dataframe,
                    metrics=metrics,
                    sample_size_a=params[0],
                    sample_size_b=params[1],
                    effect=params[2],
                    alpha=params[3],
                    bs_samples=bs_samples,
                    criterion=criterion,
                    injection_method=injection_method,
                    random_seed=seed,
                    calculate_alpha=True,
                    use_tqdm=use_tqdm,
                    parallel=parallel,
                    verbose=verbose,
                )
                for params, seed in iterator
            )
    else:
        empirical_errors: List[np.ndarray] = []
        if use_tqdm:
            iterator = tqdm(iterator)
        for params, seed in iterator:
            errors = bootstrap_over_statistical_population(
                dataframe=dataframe,
                metrics=metrics,
                sample_size_a=params[0],
                sample_size_b=params[1],
                effect=params[2],
                alpha=params[3],
                bs_samples=bs_samples,
                criterion=criterion,
                injection_method=injection_method,
                random_seed=seed,
                calculate_alpha=True,
                use_tqdm=use_tqdm,
                parallel=parallel,
                verbose=verbose,
            )
            empirical_errors.append(errors)
    return parameters, empirical_errors


def get_group_sizes(
    dataframe: pd.DataFrame,
    metrics: List,
    effects: Iterable[float],
    alphas: Iterable[float],
    betas: Iterable[float],
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    evals: int = 50,
    optim_solution: str = "binary",
    use_tqdm: bool = False,
    parallel: bool = False,
    n_jobs: int = 8,
    verbose: bool = False,
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
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed: int, optional
        A seed for the deterministic outcome of random processes.
    calculate_alpha : bool, default: ``False``
        Whether calculate correctness of criterion.
    evals : int, default: ``50``
        Evals number for optimization.
    optim_soltuion : str, default: ``"binary"``
        Optimizer method.
    use_tqdm : bool, default: ``False``
        Use tqdm progress bar.
    parallel : bool, default: ``False``
        Whether to use parallel computing.
    n_jobs : int, default: ``8``
        Amount of threads/workers for parallel
    verbose : bool, default: ``False``
        Whether use logging or not.

    Returns
    -------
    parameters, group_sizes_list : Tuple[Tuple, Tuple[np.ndarray]]
        Parameters - Cartesian product of all sample sizes, effects, alphas
        group_sizes_list - Array of lists of sizes for each metric from metrics list.
    """
    parameters: Tuple = tuple(product(effects, alphas, betas))
    seed_sequence: np.ndarray = emp_pkg.create_seed_sequence(len(parameters), random_seed)
    iterator = zip(parameters, seed_sequence)
    if parallel:
        with back_tools.tqdm_joblib(tqdm(desc="Group sizes calculation", total=len(parameters))):
            group_sizes_list = Parallel(n_jobs=n_jobs, verbose=verbose, backend="multiprocessing")(
                delayed(calculate_group_size)(
                    dataframe=dataframe,
                    metrics=metrics,
                    effect=params[0],
                    alpha=params[1],
                    beta=params[2],
                    criterion=criterion,
                    bs_samples=bs_samples,
                    injection_method=injection_method,
                    use_tqdm=use_tqdm,
                    parallel=parallel,
                    verbose=verbose,
                    evals=evals,
                    optim_solution=optim_solution,
                    random_seed=seed,
                )
                for params, seed in iterator
            )
    else:
        if use_tqdm:
            iterator = tqdm(iterator)
        group_sizes_list: List[np.ndarray] = []
        for params, seed in iterator:
            group_sizes = calculate_group_size(
                dataframe=dataframe,
                metrics=metrics,
                effect=params[0],
                alpha=params[1],
                beta=params[2],
                criterion=criterion,
                bs_samples=bs_samples,
                injection_method=injection_method,
                random_seed=seed,
                evals=evals,
                optim_solution=optim_solution,
                use_tqdm=use_tqdm,
                parallel=parallel,
                verbose=verbose,
            )
            group_sizes_list.append(group_sizes)
    return parameters, group_sizes_list


def get_empirical_mde(
    dataframe: pd.DataFrame,
    metrics: List,
    group_sizes: Iterable[int],
    alphas: Iterable[float],
    betas: Iterable[float],
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    evals: int = 50,
    optim_solution: str = "binary",
    use_tqdm: bool = False,
    parallel: bool = False,
    n_jobs: int = 8,
    verbose: bool = False,
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
        Second type erros, 1 - beta = power
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed: int, optional
        A seed for the deterministic outcome of random processes.
    calculate_alpha : bool, default: ``False``
        Whether calculate correctness of criterion or not.
    evals : int, default: ``50``
        Evals amount for optimization.
    optim_solution : str, default: ``"binary"``
        Optimizer method.
    use_tqdm : bool, default: ``False``
        Use tqdm progress bar.
    parallel : bool, default: ``False``
        Whether to use parallel computing.
    n_jobs : int, default: ``8``
        Amount of threads/workers for parallel.
    verbose : bool, default: ``False``
        Whether use logging or not.

    Returns
    -------
    parameters, empirical_errors : Tuple[Tuple, np.ndarray]
        Parameters - Cartesian product of all sample sizes, effects, alphas
        mde_list - Array of lists of mde lists for each metric from metrics list.
    """
    parameters: Tuple = tuple(product(group_sizes, alphas, betas))
    seed_sequence: np.ndarray = emp_pkg.create_seed_sequence(len(parameters), random_seed)
    iterator = zip(parameters, seed_sequence)
    if parallel:
        with back_tools.tqdm_joblib(tqdm(desc="MDE calculation", total=len(parameters))):
            mde_list = Parallel(n_jobs=n_jobs, verbose=verbose, backend="multiprocessing")(
                delayed(calculate_empirical_mde)(
                    dataframe=dataframe,
                    metrics=metrics,
                    group_size=params[0],
                    alpha=params[1],
                    beta=params[2],
                    criterion=criterion,
                    bs_samples=bs_samples,
                    injection_method=injection_method,
                    random_seed=seed,
                    evals=evals,
                    optim_solution=optim_solution,
                    use_tqdm=use_tqdm,
                    parallel=parallel,
                    verbose=verbose,
                )
                for params, seed in iterator
            )
    else:
        mde_list: List[np.ndarray] = []
        if use_tqdm:
            iterator = tqdm(iterator)
        for params, seed in iterator:
            mdes = calculate_empirical_mde(
                dataframe=dataframe,
                metrics=metrics,
                group_size=params[0],
                alpha=params[1],
                beta=params[2],
                criterion=criterion,
                bs_samples=bs_samples,
                injection_method=injection_method,
                random_seed=seed,
                evals=evals,
                optim_solution=optim_solution,
                use_tqdm=use_tqdm,
                parallel=parallel,
                verbose=verbose,
            )
            mde_list.append(mdes)
    return parameters, mde_list


def get_empirical_errors_table(
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
    use_tqdm: bool = False,
    parallel: bool = False,
    n_jobs: int = 8,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Create pandas table with type I and type II errors (power) for given data and set of parameters.

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
    random_seed: int, optional
        A seed for the deterministic outcome of random processes.
    use_tqdm : bool, default: ``False``
        Use tqdm progress bar.
    parallel : bool, default: ``False``
        Whether to use parallel computing.
    n_jobs : int, default: ``8``
        Amount of threads/workers for parallel.
    verbose : bool, default: ``False``
        Whether use logging or not.

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
        use_tqdm=use_tqdm,
        parallel=parallel,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    columns_params = ["size_A", "size_B", "effect", "alpha"]
    columns_errors = []
    for metric in metrics:
        columns_errors += [metric + "_correctness", metric + "_power"]
    report = pd.DataFrame(list(parameters), columns=columns_params).join(
        pd.DataFrame(list(emprical_errors), columns=columns_errors)
    )
    return report


def get_empirical_table_sample_size(
    dataframe: pd.DataFrame,
    metrics: List,
    effects: Iterable[float],
    alphas: Iterable[float],
    betas: Iterable[float],
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    evals: int = 50,
    optim_solution: str = "binary",
    use_tqdm: bool = False,
    parallel: bool = True,
    n_jobs: int = 8,
    verbose: bool = False,
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
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed: int, optional
        A seed for the deterministic outcome of random processes.
    evals : int, default: ``50``
        Evals amount for optimization.
    optim_solution : str, default: ``"binary"``
        Optimizer method.
    use_tqdm : bool, default: ``False``
        Use tqdm progress bar.
    parallel : bool, default: ``False``
        Whether to use parallel computing.
    n_jobs : int, default: ``8``
        Amount of threads/workers for parallel.
    verbose : bool, default: ``False``
        Whether use logging or not.

    Returns
    -------
    report : Union[pd.DataFrame, Dict[str, pd.DataFrame]
        Tables with sizes for group A, group B, effects, erros and metrics names
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
        bs_samples=bs_samples,
        criterion=criterion,
        injection_method=injection_method,
        random_seed=random_seed,
        evals=evals,
        optim_solution=optim_solution,
        use_tqdm=use_tqdm,
        parallel=parallel,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    reports = {}
    for num, metric in enumerate(metrics):
        report = pd.DataFrame(list(parameters), columns=["effect", "alpha", "beta"]).join(
            pd.DataFrame(list(np.array(group_sizes_list)[:, num]), columns=["sample_sizes"])
        )
        report["effect"] = (round((report["effect"] - 1) * 100, ROUND_DIGITS_PERCENT)).astype(str) + "%"
        report["errors"] = tuple(zip(report["alpha"], report["beta"]))
        report = report.pivot(index="effect", columns="errors", values="sample_sizes")
        report = report.sort_values(report.columns[0])
        if len(metrics) == 1:
            reports = report
        else:
            reports[metric] = report
    return reports


def get_empirical_mde_table(
    dataframe: pd.DataFrame,
    metrics: List,
    group_sizes: Iterable[int],
    alphas: Iterable[float],
    betas: Iterable[float],
    bs_samples: int = 1000,
    criterion: str = "ttest",
    injection_method: str = "constant",
    random_seed: Optional[int] = None,
    evals: int = 50,
    optim_solution: str = "binary",
    use_tqdm: bool = False,
    parallel: bool = True,
    n_jobs: int = 8,
    verbose: bool = False,
    as_numeric: bool = False,
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
    bs_samples : int, default: ``1000``
        Amount of bootstrap pairs A/B.
    criterion : str, default: ``"ttest"``
        Statistical criterion.
    injection_method : str, default: ``"constant"``
        Method to modify group B, for example
        constant: value = value * effect
        for more information inspect ``inject_effect`` function.
    random_seed: int, optional
        A seed for the deterministic outcome of random processes.
    evals : int, default: ``50``
        Evals amount for optimization.
    optim_solution : str, default: ``"binary"``
        Optimizer method.
    use_tqdm : bool, default: ``False``
        Use tqdm progress bar.
    parallel : bool, default: ``False``
        Whether to use parallel computing,
    n_jobs : int, default: ``8``
        Amount of threads/workers for parallel.
    verbose : bool, default: ``False``
        Whether use logging or not.
    as_numeric : bool, default False
        Whether to return a number or a string with percentages.

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
        bs_samples=bs_samples,
        criterion=criterion,
        injection_method=injection_method,
        random_seed=random_seed,
        evals=evals,
        optim_solution=optim_solution,
        use_tqdm=use_tqdm,
        parallel=parallel,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    reports = {}
    for num, metric in enumerate(metrics):
        report = pd.DataFrame(list(parameters), columns=["group_sizes", "alpha", "beta"]).join(
            pd.DataFrame(list(np.array(effects)[:, num]), columns=["effect"])
        )
        if as_numeric:
            report["effect"] = round(report["effect"], ROUND_DIGITS_TABLE)
        else:
            report["effect"] = (round((report["effect"] - 1) * 100, ROUND_DIGITS_PERCENT)).astype(str) + "%"
        report["errors"] = tuple(zip(report["alpha"], report["beta"]))
        report = report.pivot(index="group_sizes", columns="errors", values="effect")
        report = report.sort_index()
        if len(metrics) == 1:
            reports = report
        else:
            reports[metric] = report
    return reports
