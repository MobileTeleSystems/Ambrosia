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

from typing import Optional

import numpy as np
import scipy.stats as sps

from ambrosia import types


def __helper_calc_empirical_power(conf_interval: types.ManyIntervalType) -> np.ndarray:
    """
    Calculate esimation of power / first type error.

    Parameters
    ----------
    conf_interval : Tuple[np.ndarray]
        conf_interval[0] - left boundaries, conf_interval[1] - right
        shapes - (amount_of_samples, size_of_parameter_grid)

    Returns
    -------
    power : np.ndarray
        power / first type error for each parameter in parameter grid
    """
    left, right = conf_interval
    mask: np.ndarray = (left <= 0) & (0 <= right)
    power: np.ndarray = 1 - mask.mean(axis=0)
    return power


def __helper_bin_search_for_size(
    interval_type: str, confidence_level: float, p_a: float, p_b: float, amount: int, power: float
) -> int:
    """
    Make binary search for size to gain given power.

    Parameters
    ----------
    interval_type : str
        Type of confidence interval
    confidence_level : float
        1 - first type error value
    p_a : Iterable[float]
        Conversion in A group
    p_b : Iterable[float]
        Conversion in B group
    amount : int
        Amount of generated samples for one n(trials amount), to estimate power
    power : float
        Desired level of power

    Returns
    -------
    sample_size : int
        Such sample size, that for arguments confidence interval give
        satisfying power
    """

    def power_helper(trials: int) -> float:
        import ambrosia.tools.bin_intervals as bi

        sample_a = sps.binom.rvs(n=trials, p=p_a, size=amount)
        sample_b = sps.binom.rvs(n=trials, p=p_b, size=amount)
        binom_kwargs = {
            "interval_type": interval_type,
            "a_success": sample_a,
            "b_success": sample_b,
            "a_trials": trials,
            "b_trials": trials,
            "confidence_level": confidence_level,
        }
        conf_interval: types.IntervalType = bi.BinomTwoSampleCI.confidence_interval(**binom_kwargs)
        return __helper_calc_empirical_power(conf_interval)

    # Find upper bound for size
    k: int = 10
    current_power: float = 0
    while current_power < power:
        current_power = power_helper(trials=2**k)
        k += 1
    right: int = 2**k
    left: int = 1

    while right - left > 0:
        middle: int = (left + right) // 2
        current_power = power_helper(middle)
        if current_power < power:
            left = middle
        else:
            right = middle
    return right


def __helper_bin_search_for_delta(
    interval_type: str,
    confidence_level: float,
    p_a: float,
    trials: int,
    amount: int,
    power: float,
    epsilon: float = 0.0001,
) -> Optional[float]:
    """
    Make binary search for delta to gain given power for
    get_table_effect_on_sample_size function

    Parameters
    ----------
    interval_type : str
        Type of confidence interval
    confidence_level : float
        1 - first type error value
    p_a : Iterable[float]
        Conversion in A group
    trials: int
        Number of trials in groups
    amount : int
        Amount of generated samples for one n(trials amount), to estimate power
    power : float
        Desired level of power
    epsilon: float, default : ``0.001``
        Precision for binary search solution

    Returns
    -------
    delta : Optional[float]
        |delta - delta_optimal| < epsilon
        None if there are no satisfying deltas
    """

    def power_helper(delta: float) -> float:
        import ambrosia.tools.bin_intervals as bi

        sample_a = sps.binom.rvs(n=trials, p=p_a, size=amount)
        p_b: float = p_a - delta
        sample_b = sps.binom.rvs(n=trials, p=p_b, size=amount)
        binom_kwargs = {
            "interval_type": interval_type,
            "a_success": sample_a,
            "b_success": sample_b,
            "a_trials": trials,
            "b_trials": trials,
            "confidence_level": confidence_level,
        }
        conf_interval: types.IntervalType = bi.BinomTwoSampleCI.confidence_interval(**binom_kwargs)
        return __helper_calc_empirical_power(conf_interval)

    current_delta: float = epsilon
    current_power: float = 0
    mult_coef: float = -1 if p_a <= 0.5 else 1
    while current_power < power:
        current_power = power_helper(delta=mult_coef * current_delta)
        current_delta *= 2
        p_b: float = p_a - mult_coef * current_delta
        if (p_b < 0 or p_b > 1) and (current_delta != p_a):
            current_delta = p_a
            continue
        if p_b < 0 or p_b > 1:
            return None

    right: float = current_delta
    left: float = 0

    while right - left > epsilon:
        middle: float = (left + right) / 2
        current_power = power_helper(mult_coef * middle)
        if current_power < power:
            left = middle
        else:
            right = middle
    return mult_coef * right
