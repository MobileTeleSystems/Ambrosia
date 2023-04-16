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

from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

import ambrosia.tools.empirical_tools as emp_pkg
from ambrosia import types

EPSILON: float = 0.001


def __helper_generate_bootstrap_samples(
    dataframe: pd.DataFrame, metrics: List[str], total_size: int, bootstrap_size: int, random_seed: Optional[int] = None
) -> types.BootstrapedSamplesType:
    """
    Make dictionary {metric : samples}.

    Samples - groups A/B (total_size, bootstrap_size)
    """
    rng = np.random.default_rng(random_seed)
    sampled_metrics: types.BootstrapedSamplesType = {}
    for metric in metrics:
        sampled_metric_vals = rng.choice(dataframe[metric].values, size=(total_size, bootstrap_size))
        sampled_metrics[metric] = sampled_metric_vals
    return sampled_metrics


def __helper_inject_effect(
    sampled_metrics: types.BootstrapedSamplesType,
    sample_size_a: int,
    effect: float,
    injection_method: str = "constant",
    variation_factor: float = 10,
    random_seed: Optional[int] = None,
) -> types.BootstrapedSamplesType:
    """
    Help to inject effect after sampling groups A/B.
    """
    modified_samples_values: types.BootstrapedSamplesType = {}
    for metric, sampled_metric in sampled_metrics.items():
        modified_samples_values[metric] = emp_pkg.inject_effect(
            sampled_metric,
            sample_size_a=sample_size_a,
            effect=effect,
            modeling_method=injection_method,
            variation_factor=variation_factor,
            random_seed=random_seed,
        )
    return modified_samples_values


def __helper_get_power_for_bootstraped(
    modified_samples: types.BootstrapedSamplesType,
    sample_size: int,
    bound_size: int,
    alpha: float,
    groups_ratio: float = 1.0,
    criterion: str = "ttest",
    random_seed: Optional[int] = None,
    n_jobs: int = 1,
    verbose: bool = False,
    **kwargs,
) -> List[float]:
    """
    Calculate power for bootstraped samples.
    """
    result_power = []
    for _, values in modified_samples.items():
        sampled_metric_vals = np.vstack(
            [values[:sample_size], values[bound_size : bound_size + int(groups_ratio * sample_size)]]
        )
        power = emp_pkg.eval_error(
            sampled_metric_vals,
            sample_size_a=sample_size,
            alpha=alpha,
            mode=criterion,
            random_seed=random_seed,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs,
        )
        result_power.append(power)
    return result_power


def estimate_power(power_function: Callable, **kwargs_power) -> float:
    """
    Helps calc power with cases with cases with multioutput.
    """
    power_estimation = power_function(**kwargs_power)
    if isinstance(power_estimation, Sequence):
        power_estimation = power_estimation[0]
    return power_estimation


def helper_bin_search_upper_bound_size(
    power_function: Callable,
    power_level: float,
    groups_sizes_names: List[str],
    groups_ratio: float = 1.0,
    **kwargs_power,
) -> int:
    """
    Binary search for upper bound group size.
    """
    upper_bound_degree: int = 4
    power_estimation: float = 0
    while power_estimation < power_level:
        for gr_name in groups_sizes_names:
            kwargs_power[gr_name] = 2**upper_bound_degree
        kwargs_power[groups_sizes_names[-1]] = int(groups_ratio * kwargs_power[groups_sizes_names[-1]])
        power_estimation: float = estimate_power(power_function, **kwargs_power)
        upper_bound_degree += 1
    return upper_bound_degree


def helper_bin_searh_upper_bound_effect(power_function: Callable, power_level: float, **kwargs_power) -> int:
    """
    Binary search for upper bound effect.
    """
    upper_bound_degree: float = 1
    power_estimation: float = 0
    while power_estimation < power_level:
        kwargs_power["effect"] = 2**upper_bound_degree
        power_estimation = estimate_power(power_function, **kwargs_power)
        upper_bound_degree += 1
    return upper_bound_degree


def helper_binary_search_optimal_effect(
    power_function: Callable,
    power_level: float,
    upper_bound_effect: float,
    bootstraped_samples: np.ndarray,
    injection_method: str,
    epsilon: float = EPSILON,
    **kwargs_power,
) -> float:
    """
    Binary search for optimal effect for power and size.
    """
    left: float = 1
    right: float = upper_bound_effect
    while right - left > epsilon:
        middle: float = (left + right) / 2
        modified_samples = __helper_inject_effect(
            bootstraped_samples,
            sample_size_a=kwargs_power["sample_size"],
            effect=middle,
            injection_method=injection_method,
            random_seed=kwargs_power["random_seed"],
        )
        power_estimation: float = power_function(**kwargs_power, modified_samples=modified_samples)[0]
        if power_estimation >= power_level:
            right = middle
        else:
            left = middle
    return right


def helper_binary_search_effect_with_injection(
    power_function: Callable,
    power_level: float,
    upper_bound_effect: float,
    effect_injection_name: str,
    epsilon: float = EPSILON,
    **kwargs_power,
) -> float:
    """
    Binary search for effect using function incapsulating injection.
    """
    left: float = 1
    right: float = upper_bound_effect
    while right - left > epsilon:
        middle: float = (left + right) / 2
        kwargs_power[effect_injection_name] = middle
        power_estimation: float = estimate_power(power_function, **kwargs_power)
        if power_estimation >= power_level:
            right = middle
        else:
            left = middle
    return right


def helper_binary_search_optimal_size(
    power_function: Callable, power_level: float, upper_bound_size: int, groups_sizes_names: List[str], **kwargs_power
) -> int:
    """
    Binary search for optimal groups size.
    """
    left: int = 1
    right: int = upper_bound_size

    while right - left > 1:
        middle: int = (left + right) // 2
        for name in groups_sizes_names:
            kwargs_power[name] = middle
        power_estimation: float = estimate_power(power_function, **kwargs_power)
        if power_estimation >= power_level:
            right = middle
        else:
            left = middle
    return right
