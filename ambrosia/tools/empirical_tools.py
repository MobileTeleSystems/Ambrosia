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

from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import ambrosia.tools.pvalue_tools as pvalue_pkg
import ambrosia.tools.stat_criteria as criteria_pkg
from ambrosia import types
from ambrosia.tools.ab_abstract_component import ABStatCriterion

AVAILABLE_AB_CRITERIA: Dict[str, ABStatCriterion] = {
    "ttest": criteria_pkg.TtestIndCriterion,
    "ttest_rel": criteria_pkg.TtestRelCriterion,
    "mw": criteria_pkg.MannWhitneyCriterion,
    "wilcoxon": criteria_pkg.WilcoxonCriterion,
}


def create_seed_sequence(length: int, entropy: Optional[Union[int, Iterable[int]]] = None) -> np.ndarray:
    """
    Create a seed sequence using ``numpy.random.SeedSequence`` class.

    Parameters
    ----------
    length : int
        Total length of a sequence.
    entropy : Union[int,Iterable[int]], optional
        The entropy for creating a ``SeedSequence``.
        Used to get a deterministic result.

    Returns
    -------
    seed_sequence : List
        The seed sequence of requested length.
    """
    rng = np.random.SeedSequence(entropy)
    seed_sequence: np.ndarray = rng.generate_state(length)
    return seed_sequence


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
        modified_metric_vals[sample_size_a:, :] += (effect - 1) * mean
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


def stat_criterion_power(
    sampled_metric_vals: np.ndarray,
    sample_size_a: int,
    criterion: types.CompoundCriterionType = "ttest",
    alpha: float = 0.05,
) -> float:
    """
    Power of statistic criterion.

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
    apha : float, default: ``0.05``
        First type error bound, 1 - alpha: correctness

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
    power: float = np.mean(criterion().calculate_pvalue(a_group_metrics, b_group_metrics) <= alpha)
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

    Returns
    -------
    bs_stat : np.ndarray
        Statistic calculated via booststrap
    """
    rng = np.random.default_rng(random_seed)
    permissible_string_statistics: List[str] = ["mean", "median"]
    bs_samples: np.ndaray = rng.choice(sample, replace=True, size=(len(sample), N))
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
) -> bool:
    """
    Evaluate if difference in groups is significant.
    If confidence interval contains 0 then effect is not significant else significant.
    Return True if we have effect else False.

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
    bs_stat_diff = bs_stat_b - bs_stat_a
    left_side, right_side = np.percentile(bs_stat_diff, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    overlap = not left_side <= 0 <= right_side
    return overlap


def make_bootstrap(
    sampled_metric_vals: np.ndarray,
    sample_size_a: int,
    alpha: float = 0.05,
    N: int = 1000,
    stat: str = "mean",
    random_seed: Optional[int] = None,
    use_tqdm: bool = False,
    parallel: bool = False,
    verbose: bool = False,
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
    use_tqdm : bool, default: ``False``
        Whether to use tqdm bar progress.
    parallel : bool, default: ``False``
        Whether to use parallel calculations.
    verbose : bool, default: ``False``
        Whether to make logging.

    Returns
    -------
    test_result : float
        Empirical estimation for power.
    """
    num_samples = sampled_metric_vals.shape[1]
    rng = np.random.SeedSequence(random_seed)
    seed_sequence: np.ndarray = rng.generate_state(num_samples)
    iterator = zip(range(num_samples), seed_sequence)
    if parallel:
        results_parallel = Parallel(n_jobs=64, verbose=verbose, backend="multiprocessing")(
            delayed(get_bs_sample_stat)(
                sample=sampled_metric_vals[:, sample_num],
                sample_size_a=sample_size_a,
                alpha=alpha,
                N=N,
                stat=stat,
                random_seed=seed,
            )
            for sample_num, seed in iterator
        )
        test_result = np.mean(results_parallel)
    else:
        over_group_statistic = []
        rng = range(num_samples)
        if use_tqdm:
            iterator = tqdm(iterator)
        for sample_num, seed in iterator:
            overlap = get_bs_sample_stat(
                sample=sampled_metric_vals[:, sample_num],
                sample_size_a=sample_size_a,
                alpha=alpha,
                N=N,
                stat=stat,
                random_seed=seed,
            )
            over_group_statistic.append(overlap)
        test_result = np.mean(over_group_statistic)
    return test_result


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

    def __init__(self, bootstrap_size: int = 100, metric: Union[str, Callable] = "mean"):
        self.__bs_size = bootstrap_size
        self.__metric_distribution = np.nan
        if isinstance(metric, str):
            accepted_str_metrics: List[str] = ["mean", "fraction"]
            if metric not in accepted_str_metrics:
                raise ValueError(f'Choose metric name from - {", ".join(accepted_str_metrics)}')
        self.__metric = metric
        self.__min_of_distribution = None
        self.__max_of_distribution = None

    def __handle_str_metric(self, bootstrap_a: np.ndarray, bootstrap_b: np.ndarray) -> None:
        """
        Handle case if metric is string.
        """
        if self.__metric == "mean":
            self.__metric_distribution = np.mean(bootstrap_b, axis=1) - np.mean(bootstrap_a, axis=1)
        elif self.__metric == "fraction":
            self.__metric_distribution = np.mean(bootstrap_b, axis=1) / np.mean(bootstrap_a, axis=1) - 1

    def __handle_std_value(self) -> float:
        """
        Calculate value for criterion.
        """
        if isinstance(self.__metric, str):
            if self.__metric in ["mean", "fraction"]:
                val = 0
        else:
            val = self.__metric(np.array([1]), np.array([1]))
        return val

    def fit(self, group_a: Iterable[float], group_b: Iterable[float]) -> None:
        """
        Make bootstrap samples from given groups.
        Calculates and store empiric distribution for saved metric in __init__

        Parameters
        ----------
        group_a : Iterable[float]
            Values of A group
        group_b : Iterable[float]
            Values of B group

        Returns
        -------
            Nothing
        """
        group_a = np.array(group_a)
        group_b = np.array(group_b)
        bootstraped_a_group: np.ndarray = np.random.choice(group_a, size=(self.__bs_size, len(group_a)))
        bootstraped_b_group: np.ndarray = np.random.choice(group_b, size=(self.__bs_size, len(group_b)))
        if isinstance(self.__metric, str):
            self.__handle_str_metric(bootstraped_a_group, bootstraped_b_group)
        else:
            self.__metric_distribution = self.__metric(
                np.mean(bootstraped_a_group, axis=1), np.mean(bootstraped_b_group, axis=1)
            )

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
        alternative: str, defaulte: ``"two-sided"``
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

    def pvalue_criterion(self, alternative: str = "two-sided") -> float:
        """
        Calculate pvalue for bootstrap criterion.

        Returns
        --------
        pvalue : float
            Corresponding pvalue
        alternative: str, defaulte: ``"two-sided"``
                Alternative for static criteria - two-sided, less, greater
                Less means, that mean in first group less, than mean in second group
        """
        criterion_value: float = self.__handle_std_value()
        return pvalue_pkg.calculate_pvalue_by_interval(
            BootstrapStats.confidence_interval, criterion_value, self=self, alternative=alternative
        )
