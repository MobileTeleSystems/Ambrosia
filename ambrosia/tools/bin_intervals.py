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

from abc import ABC, abstractmethod
from typing import Iterable, List

import numpy as np
import pandas as pd
import scipy.stats as sps

import ambrosia.tools._lib._bin_ci_aide as helper_dir
import ambrosia.tools.pvalue_tools as pvalue_pkg
from ambrosia import types

RELATIVE_ABSOLUTE_DELTA_ERROR = ValueError("Choose relative or absolute delta, not both")
ROUND_DIGITS: int = 3


class BinomTwoSampleCI(ABC):
    """
    Implementation of two-sample confidence interval.
    Supported the following types  interval_type:

    X ~ Bin(m, p1)
    Y ~ Bin(n, p2)

    Estimated Delta = p1 - p2
    For more information explore:
    http://stat.wharton.upenn.edu/~lbrown/Papers/2005c%20Confidence%20intervals%20for%
    20the%20two%20sample%20binomial%20distribution%20problem.pdf

    Methods
    -------

        Frequency:
        ----------

        Wald's CI - Using MLE estimation for proportions and
                asymptotic normality

        Yule's CI - Using estimation for Var(p1 - p2), in assumption
                (p1 = p2) and (m ~= n). Could be use for criterion
                H0: p1 = p2   vs   H1: p1 != p2
        Yule's modified CI - analogue for previous one for m != n
        Newcombe's CI - combination of CI for p1 and p2, building CI for difference
        Recentred CI - Using t quantiles

        Bayes:
        ------

        Jeffrey's CI - Using for prior distribution Beta(1/2, 1/2)
        Agresti's CI - Using for prior distribution Beta(1, 1)
        Conjugate Beta - Using for prior Beta(n_success, n_failure)

    """

    __PRECISION: float = 0.0001

    # This class just a container with all confidence intervals.
    # Thats why there is no instantiation.
    @abstractmethod
    def __init__(self):
        pass

    @staticmethod
    def __wald_ci(
        sample_a: int, sample_b: int, a_trials: int, b_trials: int, significance_level: float
    ) -> types.IntervalType:
        """
        Implementation of Wald confidence interval.
        """
        p_a_est = sample_a / a_trials
        p_b_est = sample_b / b_trials
        q_a_est = 1 - p_a_est
        q_b_est = 1 - p_b_est
        delta_estimation = p_b_est - p_a_est

        quantile = sps.norm.ppf(significance_level / 2)
        variation = p_a_est * q_a_est / a_trials + p_b_est * q_b_est / b_trials
        shift = -quantile * np.sqrt(variation)

        return delta_estimation - shift, delta_estimation + shift

    @staticmethod
    def __yule_ci(
        sample_a: int, sample_b: int, a_trials: int, b_trials: int, significance_level: float, modified: bool = False
    ) -> types.IntervalType:
        """
        Implementation of Yule's confidence intervals.
        """

        p_a_est = sample_a / a_trials
        p_b_est = sample_b / b_trials
        delta_estimation = p_b_est - p_a_est

        quantile = sps.norm.ppf(significance_level / 2)

        if modified:
            coef_a = b_trials / a_trials
            coef_b = a_trials / b_trials
            p_hat = (sample_a * coef_a + sample_b * coef_b) / (a_trials + b_trials)
            q_hat = 1 - p_hat
            variation = (1 / a_trials + 1 / b_trials) * p_hat * q_hat
        else:
            p_overline = (sample_a + sample_b) / (a_trials + b_trials)
            q_overline = 1 - p_overline
            variation = (1 / a_trials + 1 / b_trials) * p_overline * q_overline

        shift = -quantile * np.sqrt(variation)
        return delta_estimation - shift, delta_estimation + shift

    @staticmethod
    def __bayes_conjugate_beta(
        sample_a: int,
        sample_b: int,
        a_trials: int,
        b_trials: int,
        n_success: int,
        n_failure: int,
        significance_level: float,
    ) -> types.IntervalType:
        """
        Implementation of bayes confidence intervals with cojugate distribution
        Beta(n_success, n_falure).
        """

        p_tilde_a = (sample_a + n_success) / (n_success + n_failure + a_trials)
        p_tilde_b = (sample_b + n_success) / (n_success + n_failure + b_trials)

        q_tilde_a = 1 - p_tilde_a
        q_tilde_b = 1 - p_tilde_b
        delta_tilde = p_tilde_b - p_tilde_a

        quantile = sps.norm.ppf(significance_level / 2)
        variation = p_tilde_a * q_tilde_a / a_trials + p_tilde_b * q_tilde_b / b_trials
        shift = -quantile * np.sqrt(variation)
        return delta_tilde - shift, delta_tilde + shift

    @staticmethod
    def __square_eq_newcombe(p_est: float, m: int, quantile: float) -> types.IntervalType:  # pylint: disable=C0103
        """
        Helper function for newcombe_ci.
        """

        coef_a = 1 + quantile**2 / m
        coef_b = -(2 * p_est + quantile**2 / m)
        coef_c = p_est**2
        d = coef_b**2 - 4 * coef_a * coef_c  # pylint: disable=C0103
        left = (-coef_b - np.sqrt(d)) / (2 * coef_a)
        right = (-coef_b + np.sqrt(d)) / (2 * coef_a)

        return left, right

    @staticmethod
    def __newcombe_ci(
        sample_a: int,
        sample_b: int,
        a_trials: int,
        b_trials: int,
        significance_level: float = 0.05,
    ) -> types.IntervalType:
        """
        Implementation of Newcombe's confidence intervals.
        """
        p_a_est = sample_a / a_trials
        p_b_est = sample_b / b_trials
        delta_est = p_b_est - p_a_est

        quantile = sps.norm.ppf(significance_level / 2)

        # Calculate intevalls for A and B group
        left_bound_group_a, right_bound_group_a = BinomTwoSampleCI.__square_eq_newcombe(p_a_est, a_trials, quantile)
        left_bound_group_b, right_bound_group_b = BinomTwoSampleCI.__square_eq_newcombe(p_b_est, b_trials, quantile)

        variance_a_left = left_bound_group_a * (1 - left_bound_group_a) / a_trials
        variance_a_right = right_bound_group_a * (1 - right_bound_group_a) / a_trials
        variance_b_left = left_bound_group_b * (1 - left_bound_group_b) / b_trials
        variance_b_right = right_bound_group_b * (1 - right_bound_group_b) / b_trials
        variance_left = variance_a_left + variance_b_right
        variance_right = variance_b_left + variance_a_right
        left_bound = delta_est + quantile * np.sqrt(variance_left)
        right_bound = delta_est - quantile * np.sqrt(variance_right)

        return left_bound, right_bound

    @staticmethod
    def __recentered_ci(
        sample_a: int, sample_b: int, a_trials: int, b_trials: int, significance_level: float
    ) -> types.IntervalType:
        """
        Implementation of recentered confidence intervals.
        """
        p_a_est = sample_a / a_trials
        p_b_est = sample_b / b_trials
        delta_estimation = p_b_est - p_a_est
        p_est = (b_trials * p_a_est + a_trials * p_b_est) / (a_trials + b_trials)
        q_est = 1 - p_est
        quantile = -sps.t.ppf(significance_level / 2, df=a_trials + b_trials - 2)
        coef = 1 + quantile**2 / (a_trials + b_trials)
        center = delta_estimation / coef
        var_est = (1 / a_trials + 1 / b_trials) * p_est * q_est
        shift = quantile * np.sqrt(coef * var_est - delta_estimation**2 / (a_trials + b_trials)) / coef
        return center - shift, center + shift

    @staticmethod
    def calculate_pvalue(
        a_success: int,
        b_success: int,
        a_trials: int,
        b_trials: int,
        interval_type: str = "wald",
        alternative: str = "two-sided",
        n_success_conjugate: int = None,
        n_failure_conjugate: int = None,
    ) -> float:
        """
        Calculate pvalue for confidence interval.
        pvalue(x) = inf_a {a | x \\in S_a }

        Parameters
        ----------
        a_success : int
            Samples from Bin(a_trials, p_a)
        b_success : int
            Sample from Bin(b_trials, p_b)
        a_trials : int
            Parameter for trials amount for group A
        b_trials : int
            Parameter for trials amount for group B
        confidence_level : float
            Pr ( [p1 - p2] in CI ) -> confidence_level, n -> infty
        interval_type : str, default : ``"wald"``
            One from [wald, yule, yule_modif, newcombe, jeffrey, agresti, bayes_beta, recenter]
        alternative : str, default : ``"two-sided"``
            Alternative for static criteria - two-sided, less, greater
            Less means, that mean in first group less, than mean in second group
        n_success : int
            Arguments for conjugate distribution Beta(n_success, n_failure) if interval_type = "bayes_beta"
        n_failure : int
            Arguments for conjugate distribution Beta(n_success, n_failure) if interval_type = "bayes_beta"

        Comment
        -------
        You can pass numpy arrays with same shapes for a_success, b_success and e.t.c

        Returns
        -------
        pvalue : float
            P-value of the interval-induced criterion
        """

        return pvalue_pkg.calculate_pvalue_by_interval(
            BinomTwoSampleCI.confidence_interval,
            0,
            a_success=a_success,
            a_trials=a_trials,
            b_success=b_success,
            b_trials=b_trials,
            n_success_conjugate=n_success_conjugate,
            n_failure_conjugate=n_failure_conjugate,
            interval_type=interval_type,
            alternative=alternative,
        )

    @staticmethod
    def confidence_interval(
        a_success: int,
        b_success: int,
        a_trials: int,
        b_trials: int,
        confidence_level: float,
        interval_type: str = "wald",
        alternative: str = "two-sided",
        n_success_conjugate: int = None,
        n_failure_conjugate: int = None,
    ) -> types.IntervalType:
        """
        Main function building confidence interval.

        Parameters
        ----------
        a_success : int
            Samples from Bin(a_trials, p_a)
        b_success : int
            Sample from Bin(b_trials, p_b)
        a_trials : int
            Parameter for trials amount for group A
        b_trials : int
            Parameter for trials amount for group B
        confidence_level : float
            Pr ( [p1 - p2] in CI ) -> confidence_level, n -> infty
        interval_type : str, default : ``"wald"``
            One from [wald, yule, yule_modif, newcombe, jeffrey, agresti, bayes_beta, recenter]
        alternative : str, default : ``"two-sided"``
            Alternative for static criteria - two-sided, less, greater
            Less means, that mean in first group less, than mean in second group
        n_success : int
            Arguments for conjugate distribution Beta(n_success, n_failure) if interval_type = "bayes_beta"
        n_failure : int
            Arguments for conjugate distribution Beta(n_success, n_failure) if interval_type = "bayes_beta"

        Note
        ----
        You can pass numpy arrays with same shapes for a_success, b_success and e.t.c

        Returns
        -------
        interval : Tuple[float, float]
        """

        valid_types: List[str] = [
            "wald",
            "yule",
            "yule_modif",
            "newcombe",
            "jeffrey",
            "agresti",
            "bayes_beta",
            "recenter",
        ]
        pvalue_pkg.check_alternative(alternative)
        significance_level = pvalue_pkg.corrected_alpha(1 - confidence_level, alternative)
        if interval_type == "wald":
            left_ci, right_ci = BinomTwoSampleCI.__wald_ci(a_success, b_success, a_trials, b_trials, significance_level)
        elif interval_type == "yule":
            left_ci, right_ci = BinomTwoSampleCI.__yule_ci(
                a_success, b_success, a_trials, b_trials, significance_level, modified=False
            )
        elif interval_type == "yule_modif":
            left_ci, right_ci = BinomTwoSampleCI.__yule_ci(
                a_success, b_success, a_trials, b_trials, significance_level, modified=True
            )
        elif interval_type == "newcombe":
            left_ci, right_ci = BinomTwoSampleCI.__newcombe_ci(
                a_success, b_success, a_trials, b_trials, significance_level
            )
        elif interval_type == "recenter":
            left_ci, right_ci = BinomTwoSampleCI.__recentered_ci(
                a_success, b_success, a_trials, b_trials, significance_level
            )
        elif interval_type == "jeffrey":
            left_ci, right_ci = BinomTwoSampleCI.__bayes_conjugate_beta(
                a_success, b_success, a_trials, b_trials, 0.5, 0.5, significance_level
            )
        elif interval_type == "agresti":
            left_ci, right_ci = BinomTwoSampleCI.__bayes_conjugate_beta(
                a_success, b_success, a_trials, b_trials, 1, 1, significance_level
            )
        elif interval_type == "bayes_beta":
            error_beta_params: str = "Pass correct params n_success_conjugate, n_faliure_conjugate"
            args_correctness = (
                (n_success_conjugate is not None)
                and (n_failure_conjugate is not None)
                and (n_success_conjugate > 0)
                and (n_failure_conjugate > 0)
            )
            if not args_correctness:
                raise ValueError(error_beta_params)
            left_ci, right_ci = BinomTwoSampleCI.__bayes_conjugate_beta(
                a_success, b_success, a_trials, b_trials, n_success_conjugate, n_failure_conjugate, significance_level
            )
        else:
            algo_type_error: str = f'Choose one from accepted methods, from - {", ".join(valid_types)}'
            raise ValueError(algo_type_error)
        left_ci, right_ci = pvalue_pkg.choose_from_bounds(
            left_ci, right_ci, alternative, left_bound=np.array([-1]), right_bound=np.array([1])
        )
        return left_ci, right_ci


def get_table_power_on_size_and_conversions(
    interval_type: str = "wald",
    p_a_values: Iterable[float] = (0.5,),
    p_b_values: Iterable[float] = (0.4,),
    sample_sizes: Iterable[int] = (100,),
    amount: int = 10000,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Table with power / empirical 1 type error = 1 - coverage, for fixed size and conversions.

    Parameters
    ----------
    interval_type : str, default : ``"wald"``
        interval_type for confidence interval
    p_a_values : Iterable[float], default : ``(0.5,)``
        Conversions for A group
    p_b_values : Itrable[float], default : ``(0.5,)``
        Conversions for B group
    sample_sizes : Iterable[float], default : ``(100,)``
        Sizes for samples
    amount : int, default : ``10000``
        Amount of generated samples for one n(trials amount), to estimate power
    confidence_level : float, default : ``0.95``
        Such value x, that: Pr ( delta in I ) >= x

    Returns
    -------
    table : pd.DataFrame
        Required table with power
    """
    trials = np.array(sample_sizes)
    conversions_cond = np.all((np.array(p_a_values) >= 0) & (np.array(p_a_values) <= 1)) and np.all(
        (np.array(p_b_values) >= 0) & (np.array(p_b_values) <= 1)
    )
    if not conversions_cond:
        raise ValueError("Conversions must be from 0 to 1")
    powers_array: List[np.ndarray] = []
    for p_a in p_a_values:
        for p_b in p_b_values:
            sample_a = sps.binom.rvs(n=trials, p=p_a, size=(amount, len(sample_sizes)))
            sample_b = sps.binom.rvs(n=trials, p=p_b, size=(amount, len(sample_sizes)))
            binom_kwargs = {
                "interval_type": interval_type,
                "a_success": sample_a,
                "b_success": sample_b,
                "a_trials": trials,
                "b_trials": trials,
                "confidence_level": confidence_level,
            }
            conf_interval: types.ManyIntervalType = BinomTwoSampleCI.confidence_interval(**binom_kwargs)
            power: np.ndarray = helper_dir.__helper_calc_empirical_power(conf_interval)
            powers_array.append(power)
    power_matrix = np.vstack(powers_array)
    table = pd.DataFrame(
        power_matrix,
        index=pd.MultiIndex.from_tuples(
            [(round(p_a, 3), round(p_b, 3)) for p_a in p_a_values for p_b in p_b_values], names=[r"$p_a$", r"$p_b$"]
        ),
        columns=sample_sizes,
    )
    table.index.name = "conversions"
    table.columns.name = "sample sizes"
    return table


def get_table_power_on_size_and_delta(
    p_a: float,
    sample_sizes: Iterable[int],
    interval_type: str = "wald",
    delta_values: Iterable[float] = None,
    delta_relative_values: Iterable[float] = None,
    amount: int = 10000,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Table with power / empirical 1 type error = 1 - coverage for fixed size and effect.

    Parameters
    ----------
    interval_type : str
        interval_type for confidence interval
    p_a : Iterable[float]
        Conversion in A group
    sample_sizes : Iterable[float]
        Sizes for samples
    delta_values : Iterable[float]
        Absolute delta values: p_a - p_b = delta
    delta_relative_values : Iterable[float]
        Relative delta values: delta_relative * p_a = p_b
    amount : int, default : ``10000``
        Amount of generated samples for one n(trials amount), to estimate power
    confidence_level : float, default : ``0.95``
        Such value x, that: Pr ( delta in I ) >= x

    Returns
    -------
    table : pd.DataFrame
        Required table with power
    """
    trials = np.array(sample_sizes)
    if not (delta_values is None) ^ (delta_relative_values is None):
        raise RELATIVE_ABSOLUTE_DELTA_ERROR
    if delta_values is not None:
        p_b_values: np.ndarray = p_a - np.array(delta_values)
    else:
        p_b_values: np.ndarray = p_a * np.array(delta_relative_values)
    if not np.all((p_b_values >= 0) & (p_b_values <= 1)):
        raise ValueError(f"Probability of success in group B must be positive, not {p_b_values}")
    powers_array: List[np.ndarray] = []
    sample_a = sps.binom.rvs(n=trials, p=p_a, size=(amount, trials.shape[0]))

    for p_b in p_b_values:
        sample_b = sps.binom.rvs(n=trials, p=p_b, size=(amount, trials.shape[0]))
        binom_kwargs = {
            "interval_type": interval_type,
            "a_success": sample_a,
            "b_success": sample_b,
            "a_trials": trials,
            "b_trials": trials,
            "confidence_level": confidence_level,
        }
        conf_interval: types.ManyIntervalType = BinomTwoSampleCI.confidence_interval(**binom_kwargs)
        power: np.ndarray = helper_dir.__helper_calc_empirical_power(conf_interval)
        powers_array.append(power)
    power_matrix = np.vstack(powers_array)
    index = delta_values if delta_values is not None else delta_relative_values
    table = pd.DataFrame(power_matrix, index=index, columns=sample_sizes)
    table.index.name = r"$\Delta$-absolute" if delta_values is not None else r"$\delta$-relative"
    table.columns.name = "sample sizes"
    table_title: str = r"$1 - \beta$: power of criterion, " + (
        r"$p_a-p_b=\Delta$" if delta_values else r"$p_a\delta=p_b$"
    )
    table = table.style.set_caption(table_title)
    return table


def iterate_for_sample_size(
    interval_type: str,
    first_errors: Iterable[float],
    second_errors: Iterable[float],
    p_a: float,
    p_b_values: Iterable[float],
    grid_delta: Iterable[float],
    amount: int,
) -> pd.DataFrame:
    """
    Iterate over params for different sample size
    """
    values = [(round(a, ROUND_DIGITS), round(b, ROUND_DIGITS)) for a in first_errors for b in second_errors]
    table: pd.DataFrame = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(values, names=[r"$\alpha$", r"$\beta$"]),
        columns=grid_delta,
    )
    for alpha in first_errors:
        for second_error in second_errors:
            power = 1 - second_error
            for p_b, delta in zip(p_b_values, grid_delta):
                trials = helper_dir.__helper_bin_search_for_size(
                    interval_type=interval_type,
                    confidence_level=1 - alpha,
                    p_a=p_a,
                    p_b=p_b,
                    amount=amount,
                    power=power,
                )
                table.loc[(alpha, second_error), delta] = trials
    return table


def get_table_sample_size_on_effect(
    interval_type: str = "wald",
    first_errors: Iterable[float] = (0.05,),
    second_errors: Iterable[float] = (0.2,),
    p_a: float = 0.5,
    delta_values: Iterable[float] = None,
    delta_relative_values: Iterable[float] = None,
    amount: int = 10000,
) -> pd.DataFrame:
    """
    Table for sample sizes with given effect and errors.

    Parameters
    ----------
    interval_type : str, default : ``"wald"``
        interval_type for confidence interval
    first_errors : Iterable[float], default : ``(0.05,)``
        First type error values
    second_errors : Iterable[float], default : ``(0.2,)``
        Second type error values
    p_a : Iterable[float]
        Conversion in A group, default : ``0.5``
    delta_values : Iterable[float]
        Absolute delta values: p_a - p_b = delta
    delta_relative_values : Iterable[float]
        Relative delta values: delta_relative * p_a = p_b
    amount : int, default : ``1000``
        Amount of generated samples for one n(trials amount), to estimate power

    Returns
    -------
    table : pd.DataFrame
        Required table with sample sizes
    """

    errors_condition = np.all((np.array(first_errors) >= 0) & (np.array(first_errors) <= 1)) and np.all(
        (np.array(second_errors) >= 0) & (np.array(second_errors) <= 1)
    )
    if not errors_condition:
        raise ValueError("Errors must be from 0 to 1")

    if (delta_values is not None) and (delta_relative_values is not None):
        raise RELATIVE_ABSOLUTE_DELTA_ERROR

    # If delta type not set => set to absolute
    if delta_values is None and delta_relative_values is None:
        delta_values = [p_a / 2]

    if delta_values is not None:
        p_b_values: np.ndarray = p_a - np.array(delta_values)
    else:
        p_b_values: np.ndarray = p_a * np.array(delta_relative_values)
    if not np.all((p_b_values >= 0) & (p_b_values <= 1)):
        raise ValueError(f"Probability of success in group B must be positive, not {p_b_values}")

    grid_delta: np.ndarray = delta_values if delta_values is not None else delta_relative_values
    table = iterate_for_sample_size(interval_type, first_errors, second_errors, p_a, p_b_values, grid_delta, amount)

    table.columns.name = r"$\Delta$-absolute" if delta_values is not None else r"$\delta$-relative"
    return table


def iterate_for_delta(
    interval_type: str,
    first_errors: Iterable[float],
    second_errors: Iterable[float],
    sample_sizes: Iterable[int],
    p_a: float,
    amount: int,
    delta_type: str,
) -> pd.DataFrame:
    """
    Helps find effect for different params
    """
    values = [(round(a, ROUND_DIGITS), round(b, ROUND_DIGITS)) for a in first_errors for b in second_errors]
    table: pd.DataFrame = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(values, names=[r"$\alpha$", r"$\beta$"]),
        columns=sample_sizes,
    )
    for alpha in first_errors:
        for second_error in second_errors:
            power = 1 - second_error
            for trials in sample_sizes:
                delta = helper_dir.__helper_bin_search_for_delta(
                    interval_type=interval_type,
                    confidence_level=1 - alpha,
                    p_a=p_a,
                    trials=trials,
                    amount=amount,
                    power=power,
                )
                if delta is not None and delta_type == "relative":
                    delta = str(round(abs(delta) / p_a * 100, 2)) + "%"

                table.loc[(alpha, second_error), trials] = delta
    return table


def get_table_effect_on_sample_size(
    interval_type: str = "wald",
    first_errors: Iterable[float] = (0.05,),
    second_errors: Iterable[float] = (0.2,),
    sample_sizes: Iterable[int] = (100,),
    p_a: float = 0.5,
    amount: int = 10000,
    delta_type: str = "absolute",
) -> pd.DataFrame:
    """
    Table for effects with given sample sizes and erros.
    If there are no effects satisfy first and second errors value will be set to None

    Parameters
    ----------
    interval_type : str, default : ``"wald"``
        interval_type for confidence interval
    first_errors : Iterable[float], default : ``(0.05,)``
        First type error values
    second_errors : Iterable[float], default : ``(0.2,)``
        Second type error values
    p_a : Iterable[float], default : ``0.5``
        Conversion in A group
    amount : int, default : ``10000``
        Amount of generated samples for one n(trials amount), to estimate power
    sample_sizes : Iterable[int], default : ``(100,)``
        Sample sizes for A/B group
    delta_type : str, default : ``"absolute``
        absolute or relative, if relative give effect if percent: |delta| / p_a

    Returns
    -------
    table : pd.DataFrame
        Required table with effects
    """
    errors_condition = np.all((np.array(first_errors) >= 0) & (np.array(first_errors) <= 1)) and np.all(
        (np.array(second_errors) >= 0) & (np.array(second_errors) <= 1)
    )
    if not errors_condition:
        error_mesage_errors: str = "Errors must be from 0 to 1"
        raise ValueError(error_mesage_errors)

    delta_types: List[str] = ["absolute", "relative"]
    if delta_type not in delta_types:
        raise ValueError(f"Delta type must be absolute relative, not {delta_type}")

    table: pd.DataFrame = iterate_for_delta(
        interval_type,
        first_errors,
        second_errors,
        sample_sizes,
        p_a,
        amount,
        delta_type,
    )
    table.columns.name = "Sample size"
    return table
