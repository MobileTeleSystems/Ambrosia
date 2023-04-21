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


from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed, parallel_backend

import ambrosia.tools.pvalue_tools as pvalue_pkg
import ambrosia.tools.stat_criteria as criteria_pkg
from ambrosia import types
from ambrosia.tools.ab_abstract_component import ABStatCriterion
from ambrosia.tools.decorators import filter_kwargs

AVAILABLE_AB_CRITERIA: Dict[str, ABStatCriterion] = {
    "ttest": criteria_pkg.TtestIndCriterion,
    "ttest_rel": criteria_pkg.TtestRelCriterion,
    "mw": criteria_pkg.MannWhitneyCriterion,
    "wilcoxon": criteria_pkg.WilcoxonCriterion,
}


def inject_effect(
    sampled_metric_vals: np.ndarray,
    sample_size_a: int,
    effect: float,
    modeling_method: str = "constant",
    variation_factor: float = 10,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Inject effect to values of group B.

    Parameters
    ----------
    sampled_metric_vals : np.ndarray
        Samples of groups A/B: |group A values|group B values|.
    sample_size_a : int
        Size of group A in ``sampled_metric_vals``, i.e.
        first ``sample_size_a`` elements correspond to group A.
    effect : float
        Effect for modifying group B,
        for example value = value * effect.
    modeling_method : str, default: ``constant``
        Method to modify group B
        ``constant``: value = value * effect
        ``shift``: value = value + value_mean * effect
        ``normal``: Add gauss noise Normal(mean(A) * (1 - effect),
                    Variation(A) / variation_factor).
    variation_factor : float, default: ``10``
        Factor for variation in ``normal`` method.

    Returns
    -------
    modified_metric_vals : np.ndarray
        Modified metric in group B and same values in group A.
    """
    available_modeling_mehods: List[str] = ["constant", "shift", "normal"]
    rng = np.random.default_rng(random_seed)
    modified_metric_vals: np.ndarray = sampled_metric_vals.copy()
    mean: np.ndarray = modified_metric_vals[sample_size_a:, :].mean()
    bs_size: int = modified_metric_vals.shape[1]
    sample_size_b: int = len(modified_metric_vals) - sample_size_a

    if modeling_method == "constant":
        modified_metric_vals[sample_size_a:, :] = effect * modified_metric_vals[sample_size_a:, :]
    elif modeling_method == "shift":
        modified_metric_vals[sample_size_a:, :] = modified_metric_vals[sample_size_a:, :] + (effect - 1) * mean
    elif modeling_method == "normal":
        effect_delta = (effect - 1) * mean
        effect_std = modified_metric_vals[sample_size_a:, :].std(ddof=1) / variation_factor
        modified_metric_vals[sample_size_a:, :] = modified_metric_vals[sample_size_a:, :] + rng.normal(
            loc=effect_delta, scale=effect_std, size=(sample_size_b, bs_size)
        )
    else:
        raise ValueError(
            f"Effect modeling method {modeling_method} is not found, chose from {available_modeling_mehods}"
        )
    return modified_metric_vals


def estim_stat_criterion_power(
    sampled_metric_vals: np.ndarray,
    sample_size_a: int,
    criterion: types.CompoundCriterionType = "ttest",
    alpha: float = 0.05,
    **kwargs,
) -> float:
    """
    Estimate power of statistical criterion.

    Parameters
    ----------
    sampled_metric_vals : np.ndarray
        Sampled metrics for groups A/B
        |group A values|group B values|
    sample_size_a : int
        Size of group A
    criterion : Union[Callable[[np.ndarray, np.ndarray], CriterionResultType], str], default: ``"ttest"``
        Statistical criterion - function f: f(group A, group B) = (statistic, p_value)
        or name of criterion as string, for example 'ttest'
    alpha : float, default: ``0.05``
        First type error bound, 1 - alpha: correctness
    **kwargs : Dict
        Keyword arguments for statistical criterion.

    Returns
    -------
    power : float
        Empirical bootstraped estimation for power
    """
    a_group_metrics: np.ndarray = sampled_metric_vals[:sample_size_a]
    b_group_metrics: np.ndarray = sampled_metric_vals[sample_size_a:]

    if isinstance(criterion, str) & (criterion in AVAILABLE_AB_CRITERIA):
        criterion = AVAILABLE_AB_CRITERIA[criterion]
    elif not (hasattr(criterion, "calculate_pvalue") and callable(criterion.calculate_pvalue)):
        raise ValueError(
            f"Choose correct criterion name from {list(AVAILABLE_AB_CRITERIA)} or pass correct custom class"
        )
    power: float = np.mean(criterion().calculate_pvalue(a_group_metrics, b_group_metrics, **kwargs) <= alpha)
    return power


def get_bs_stat(sample: np.ndarray, stat: str = "mean", N: int = 1000, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Evaluate statistic (mean / median) using bootstrap method.

    Parameters
    ----------
    sample : np.ndarray
        Given sample array
    stat : str, default: ``"mean"``
        Name of statistic to be calculated
    N : int, default: ``1000``
       Bootstrap size
    random_seed : int, optional
        A seed for the deterministic outcome of random processes.

    Returns
    -------
    bs_stat : np.ndarray
        Statistic calculated via bootstrap
    """
    rng = np.random.default_rng(random_seed)
    permissible_string_statistics: List[str] = ["mean", "median"]
    bs_samples: np.ndarray = rng.choice(sample, replace=True, size=(len(sample), N))
    if stat == "mean":
        bs_stat: np.ndarray = np.mean(bs_samples, axis=0)
    elif stat == "median":
        bs_stat: np.ndarray = np.median(bs_samples, axis=0)
    else:
        raise ValueError(f'Statistic is not found, choose from {", ".join(permissible_string_statistics)}')
    return bs_stat


def get_bs_sample_stat(
    sample: np.ndarray,
    sample_size_a: int,
    alpha: float,
    N: int = 1000,
    stat: str = "mean",
    random_seed: Optional[int] = None,
    alternative: str = "two-sided",
) -> bool:
    """
    Evaluate if difference in groups is significant.

    If confidence interval contains 0 then effect is not statistically significant.
    Returns ``True`` if we have effect else ``False``.

    Parameters
    ----------
    sample : np.ndarray
        Sample with groups A and B
    sample_size_a : int
        Size of group A => sample = |group A| group B|
    alpha : float
        Bound for first type error
    N : int, default: ``1000``
       Bootstrap size
    stat : str, default: ``"mean"``
          Name of calculated statistic, for example 'mean'
    random_seed : int, optional
        A seed for the deterministic outcome of random processes.
    alternative : str, default: ``"two-sided"``
        Alternative hypothesis, can be ``"two-sided"``, ``"greater"``
        or ``"less"``.

    Returns
    -------
    overlap : bool
        True => H_0 is rejected <-> there is effect in groups
        False => H_0 is not rejected <-> there is no effect in groups
    """
    rng = np.random.SeedSequence(random_seed)
    seed_sequence: np.ndarray = rng.generate_state(2)
    bs_stat_a = get_bs_stat(sample[:sample_size_a], stat=stat, N=N, random_seed=seed_sequence[0])
    bs_stat_b = get_bs_stat(sample[sample_size_a:], stat=stat, N=N, random_seed=seed_sequence[1])
    bs_stat_diff: np.ndarray = bs_stat_b - bs_stat_a
    if alternative == "less":
        right_side = np.quantile(bs_stat_diff, 1 - alpha)
        overlap = right_side <= 0
    elif alternative == "greater":
        left_side = np.quantile(bs_stat_diff, alpha)
        overlap = left_side >= 0
    elif alternative == "two-sided":
        left_side, right_side = np.quantile(bs_stat_diff, [alpha / 2.0, 1 - alpha / 2.0])
        overlap = not left_side <= 0 <= right_side
    else:
        raise ValueError(f"Incorrect alternative value - {alternative}, choose from two-sided, less, greater")
    return overlap


def make_bootstrap(
    sampled_metric_vals: np.ndarray,
    sample_size_a: int,
    alpha: float = 0.05,
    N: int = 1000,
    stat: str = "mean",
    random_seed: Optional[int] = None,
    n_jobs: int = 1,
    backend: str = "loky",
    verbose: bool = False,
    **kwargs,
) -> float:
    """
    Evaluate share of cases when we find difference in groups using bootstrap.
    We can use parallel mode for faster evaluations in case of large data.

    Parameters
    ----------
    sampled_metric_vals : np.ndarray
         Metric values in both groups.
    sample_size_a : int
        Size of group A => |group A| group B|.
    alpha : float, default: ``0.05``
        Bound for the first type error.
    N : int, default: ``1000``
       Number of bootstraps.
    stat : str, default: ``"mean"``
        Statistics to be calculated.
    random_seed : int, optional
        A seed for the deterministic outcome of random processes.
    n_jobs : int, default: ``1``
        Amount of threads/workers for parallel.
    backend : str, default: ``"loky"``
        Type of backend for joblib parallel computation.
    verbose : bool, default: ``False``
        Whether to make logging.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    test_result : float
        Empirical estimation for power.
    """
    num_samples = sampled_metric_vals.shape[1]
    rng = np.random.SeedSequence(random_seed)
    seed_sequence: np.ndarray = rng.generate_state(num_samples)
    iterator = zip(range(num_samples), seed_sequence)
    with parallel_backend(backend=backend, n_jobs=n_jobs):
        results_parallel = Parallel(verbose=verbose)(
            delayed(get_bs_sample_stat)(
                sample=sampled_metric_vals[:, sample_num],
                sample_size_a=sample_size_a,
                alpha=alpha,
                N=N,
                stat=stat,
                random_seed=seed,
                **kwargs,
            )
            for sample_num, seed in iterator
        )
    return np.mean(results_parallel)


def eval_error(
    sampled_metric_vals: np.ndarray,
    sample_size_a: int,
    alpha: float,
    mode: str = "ttest",
    stat: str = "mean",
    bootstrap_size: int = 1000,
    random_seed: Optional[int] = None,
    n_jobs: int = 1,
    verbose: bool = False,
    **kwargs,
) -> float:
    """
    Evaluate I type error/power of the experiment.

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
    random_seed : int, optional
        A seed for the deterministic outcome of random processes.
    n_jobs : int, default: ``1``
        Amount of threads/workers for parallel.
    verbose : bool, default: ``False``
        Whether use logging.
    **kwargs : Dict
        Keyword arguments for statistical criterion.

    Returns
    -------
    error : float
        Second type error estimation or correctness, i.e.
        1 - P_{A=B} (criterion is completed) - correctness
        P_{A!=B} (criterion is completed) - second type error.
    """
    not_bootstrap_criteria: List[str] = ["ttest", "ttest_rel", "mw", "wilcoxon"]
    if mode in not_bootstrap_criteria:
        power: float = estim_stat_criterion_power(
            sampled_metric_vals, sample_size_a, criterion=mode, alpha=alpha, **kwargs
        )
    elif mode == "bootstrap":
        power: float = make_bootstrap(
            sampled_metric_vals,
            sample_size_a,
            alpha,
            N=bootstrap_size,
            stat=stat,
            random_seed=random_seed,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs,
        )
    else:
        raise ValueError(f"Criterion {mode} is not found, choose from {not_bootstrap_criteria} or 'bootstrap'")
    return power


class BootstrapStats:
    """
    Generation empirical distribution for statistic(group A values, group B values).

    Attributes
    ----------
    All attributes are private.

    Methods
    -------
    fit(group_a: Iterable[float], group_b: Iterable[float])
        Fits the empirical distribution using values from group_a and group_b

    confidence_interval(confidence_level: Union[float, Iterable[float]]=0.95)
        Build confidence interval using empirical distribution from fit method with
        given confidence level

    pvalue_criterion()
        Calculate pvalue using confidence interval as criterion

    """

    def __init__(self, bootstrap_size: int = 10000, metric: Union[str, Callable] = "mean", paired: bool = False):
        """
        Parameters
        ----------
        bootstrap_size: int
            Amount of bootstrap groups
        metric: str or callable
            Metric to be calculated - mean or fraction
        paired: bool, default False
            If True use paired sampling, could be usefull for paired groups
        """
        self.__bs_size = bootstrap_size
        self.__metric_distribution = np.nan
        if isinstance(metric, str):
            accepted_str_metrics: List[str] = ["mean", "fraction", "median"]
            if metric not in accepted_str_metrics:
                raise ValueError(f'Choose metric name from - {", ".join(accepted_str_metrics)}')
        self.__metric = metric
        self.__min_of_distribution = None
        self.__max_of_distribution = None
        if isinstance(paired, bool):
            self.__paired = paired
        else:
            raise ValueError("Parameter paired can only take boolean values")

    def __handle_str_metric(self, bootstrap_a: np.ndarray, bootstrap_b: np.ndarray) -> None:
        """
        Handle case if metric is string.
        """
        if self.__metric == "mean":
            self.__metric_distribution = np.mean(bootstrap_b, axis=1) - np.mean(bootstrap_a, axis=1)
        elif self.__metric == "fraction":
            self.__metric_distribution = np.mean(bootstrap_b, axis=1) / np.mean(bootstrap_a, axis=1) - 1
        elif self.__metric == "median":
            self.__metric_distribution = np.median(bootstrap_b, axis=1) - np.median(bootstrap_a, axis=1)

    def __handle_std_value(self) -> float:
        """
        Calculate value for criterion.
        """
        if isinstance(self.__metric, str):
            if self.__metric in ["mean", "fraction", "median"]:
                val = 0
        else:
            val = self.__metric(np.array([1]), np.array([1]))
        return val

    def __handle_sampling(
        self, group_a: Iterable[float], group_b: Iterable[float], random_seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_seed)
        if self.__paired:
            a_size, b_size = len(group_a), len(group_b)
            if a_size != b_size:
                err: str = f"Paired groups must have equal sizes, have - {len(group_a)} and {len(group_b)}"
                raise ValueError(err)
            idxs: np.ndarray = rng.choice(np.arange(a_size), size=(self.__bs_size, a_size))
            return group_a[idxs], group_b[idxs]
        return (
            rng.choice(group_a, size=(self.__bs_size, len(group_a))),
            rng.choice(group_b, size=(self.__bs_size, len(group_b))),
        )

    @filter_kwargs
    def fit(self, group_a: Iterable[float], group_b: Iterable[float], random_seed: Optional[int] = None) -> None:
        """
        Make bootstrap samples from given groups.
        Calculates and store empiric distribution for saved metric in __init__

        Parameters
        ----------
        group_a : Iterable[float]
            Values of A group
        group_b : Iterable[float]
            Values of B group
        random_seed : int, optional
            A seed for the deterministic outcome of random bootstrap processes.

        Returns
        -------
            Nothing
        """
        group_a = np.array(group_a)
        group_b = np.array(group_b)
        bootstraped_a_group, bootstraped_b_group = self.__handle_sampling(group_a, group_b, random_seed)
        if isinstance(self.__metric, str):
            self.__handle_str_metric(bootstraped_a_group, bootstraped_b_group)
        else:
            self.__metric_distribution = self.__metric(bootstraped_a_group, bootstraped_b_group)

    def min_of_distrbution(self) -> float:
        """
        Minimum empirical distribution
        """
        if self.__min_of_distribution is None:
            self.__min_of_distribution = np.min(self.__metric_distribution)
        return self.__min_of_distribution

    def max_of_distribution(self) -> float:
        """
        Maximum empirical distribution
        """
        if self.__max_of_distribution is None:
            self.__max_of_distribution = np.max(self.__metric_distribution)
        return self.__max_of_distribution

    @filter_kwargs
    def confidence_interval(
        self, confidence_level: Union[float, Iterable[float]] = 0.95, alternative: str = "two-sided"
    ) -> types.IntervalType:
        """
        Returns bootstraped confidence interval, based on fit method.

        Parameters
        ----------
        confidence_level: Union[float, Iterable[float]], default: ``0.95``
            Bounds for error, that is
            Pr (mean(metric) not in interval) <= alpha
        alternative : str, default: ``"two-sided"``
                Alternative for static criteria - two-sided, less, greater
                Less means, that mean in first group less, than mean in second group
        Returns
        -------
        interval : IntervalType
            Confidence interval for each error in alpha
        """
        if not hasattr(self, "_BootstrapStats__metric_distribution"):
            raise AttributeError("Use method fit to build empirical distribution on metric")

        alpha: Union[float, Iterable[float]] = 1 - confidence_level
        alpha = pvalue_pkg.corrected_alpha(alpha, alternative)
        if not isinstance(alpha, float):
            alpha = np.array(alpha)
        left_bounds: np.ndarray = np.quantile(self.__metric_distribution, q=alpha / 2)
        right_bounds: np.ndarray = np.quantile(self.__metric_distribution, q=1 - alpha / 2)

        return pvalue_pkg.choose_from_bounds(
            left_bounds,
            right_bounds,
            alternative,
            left_bound=self.min_of_distrbution(),
            right_bound=self.max_of_distribution(),
        )

    @filter_kwargs
    def pvalue_criterion(self, alternative: str = "two-sided") -> float:
        """
        Calculate pvalue for bootstrap criterion.

        Returns
        --------
        pvalue : float
            Corresponding pvalue
        alternative : str, defaulte: ``"two-sided"``
                Alternative for static criteria - two-sided, less, greater
                Less means, that mean in first group less, than mean in second group
        """
        criterion_value: float = self.__handle_std_value()
        return pvalue_pkg.calculate_pvalue_by_interval(
            BootstrapStats.confidence_interval, criterion_value, self=self, alternative=alternative, precision=10e-15
        )
