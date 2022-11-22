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


from typing import List, Tuple

import numpy as np
import scipy.stats as sps

import ambrosia.tools.pvalue_tools as pvalue_pkg
import ambrosia.tools.theoretical_tools as theory_pkg
from ambrosia import types
from ambrosia.tools.ab_abstract_component import ABStatCriterion, StatCriterion


def get_results_dict(alpha: float, pvalue: float, effect: float, conf_int: Tuple[float, float]):
    """
    Returns dictionary with given values
    """
    return {"first_type_error": alpha, "pvalue": pvalue, "effect": effect, "confidence_interval": conf_int}


def get_calc_effect_ttest(group_a: np.ndarray, group_b: np.ndarray, effect_type: str = "absolute"):
    """
    Calculation effect for ttest
    """
    if effect_type == "absolute":
        return np.mean(group_b, axis=0) - np.mean(group_a, axis=0)
    elif effect_type == "relative":
        return np.mean(group_b, axis=0) / np.mean(group_a, axis=0) - 1
    else:
        raise ValueError(ABStatCriterion._send_type_error_msg())  # pylint: disable=W0212


class TtestIndCriterion(ABStatCriterion):
    """
    Unit for independent T-test.
    """

    implemented_effect_types: List = ["absolute", "relative"]

    def calculate_pvalue(self, group_a: np.ndarray, group_b: np.ndarray, effect_type: str = "absolute", **kwargs):
        if effect_type == "absolute":
            return sps.ttest_ind(a=group_a, b=group_b, equal_var=False, **kwargs).pvalue
        elif effect_type == "relative":
            _, pvalue = theory_pkg.apply_delta_method(group_a, group_b, "fraction", **kwargs)
            return pvalue
        else:
            raise ValueError(self._send_type_error_msg())

    def calculate_effect(self, group_a: np.ndarray, group_b: np.ndarray, effect_type: str = "absolute"):
        return get_calc_effect_ttest(group_a, group_b, effect_type)

    def _build_intervals_absolute(
        self,
        center: float,
        group_a: np.ndarray,
        group_b: np.ndarray,
        alpha: types.StatErrorType = np.array([0.05]),
        alternative: str = "two-sided",
    ):
        """
        Helps handle different alternatives and dimension for student distribution
        """
        alpha_corrected: float = pvalue_pkg.corrected_alpha(alpha, alternative)
        quantiles, std_error = theory_pkg.get_ttest_info(group_a, group_b, alpha_corrected)
        left_ci: np.ndarray = center - quantiles * std_error
        right_ci: np.ndarray = center + quantiles * std_error
        left_ci, right_ci = pvalue_pkg.choose_from_bounds(left_ci, right_ci, alternative)
        conf_intervals = list(zip(left_ci, right_ci))
        return conf_intervals

    def calculate_conf_interval(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        alpha: types.StatErrorType = np.array([0.05]),
        effect_type: str = "absolute",
        **kwargs,
    ):
        if isinstance(alpha, float):
            alpha = np.array([alpha])
        if effect_type == "absolute":
            difference_estimation: float = group_b.mean() - group_a.mean()
            conf_intervals = self._build_intervals_absolute(difference_estimation, group_a, group_b, alpha, **kwargs)
        elif effect_type == "relative":
            conf_intervals, _ = theory_pkg.apply_delta_method(group_a, group_b, "fraction", alpha, **kwargs)
        else:
            raise ValueError(self._send_type_error_msg())
        return conf_intervals

    def get_results(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        alpha: types.StatErrorType = 0.05,
        effect_type: str = "absolute",
        **kwargs,
    ) -> types.StatCriterionResult:
        """
        Override this method from Base class for handle cases with
        """
        if effect_type == "relative":
            conf_int, pvalue = theory_pkg.apply_delta_method(group_a, group_b, "fraction", alpha, **kwargs)
            effect: float = self.calculate_effect(group_a, group_b, effect_type)
            return get_results_dict(alpha, pvalue, effect, conf_int)
        return super().get_results(group_a, group_b, alpha, effect_type, **kwargs)


class TtestRelCriterion(ABStatCriterion):
    """
    Unit for relative paired T-test.
    """

    implemented_effect_types: List = ["absolute", "relative"]

    def calculate_pvalue(self, group_a: np.ndarray, group_b: np.ndarray, effect_type: str = "absolute", **kwargs):
        if effect_type == "absolute":
            return sps.ttest_rel(a=group_a, b=group_b, **kwargs).pvalue
        elif effect_type == "relative":
            _, pvalue = theory_pkg.apply_delta_method(group_a, group_b, "fraction", dependent=True, **kwargs)
            return pvalue
        else:
            raise ValueError(self._send_type_error_msg())

    def calculate_effect(self, group_a: np.ndarray, group_b: np.ndarray, effect_type: str = "absolute"):
        return get_calc_effect_ttest(group_a, group_b, effect_type)

    def _build_intervals_absolute(
        self,
        center: float,
        group_a: np.ndarray,
        group_b: np.ndarray,
        alpha: types.StatErrorType = np.array([0.05]),
        alternative: str = "two-sided",
    ):
        """
        Helps handle different alternatives and build confidence interval
        for related sampels
        """
        alpha_corrected: float = pvalue_pkg.corrected_alpha(alpha, alternative)
        std_error = np.sqrt(np.var(group_b - group_a, ddof=1) / len(group_a))
        quantiles = sps.t.ppf(1 - alpha_corrected / 2, df=len(group_a) - 1)
        left_ci: float = center - quantiles * std_error
        right_ci: float = center + quantiles * std_error
        left_ci, right_ci = pvalue_pkg.choose_from_bounds(left_ci, right_ci, alternative)
        conf_intervals = list(zip(left_ci, right_ci))
        return conf_intervals

    def calculate_conf_interval(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        alpha: types.StatErrorType = np.array([0.05]),
        effect_type: str = "absolute",
        **kwargs,
    ):
        if isinstance(alpha, float):
            alpha = np.array([alpha])
        if effect_type == "absolute":
            difference_estimation: float = np.mean(group_b - group_a)
            conf_intervals = self._build_intervals_absolute(difference_estimation, group_a, group_b, alpha, **kwargs)
        elif effect_type == "relative":
            conf_intervals, _ = theory_pkg.apply_delta_method(
                group_a, group_b, "fraction", alpha, dependent=True, **kwargs
            )
        else:
            raise ValueError(self._send_type_error_msg())
        return conf_intervals

    def get_results(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        alpha: types.StatErrorType = 0.05,
        effect_type: str = "absolute",
        **kwargs,
    ) -> types.StatCriterionResult:
        """
        Override this method from Base class for handle cases with
        """
        if effect_type == "relative":
            conf_int, pvalue = theory_pkg.apply_delta_method(
                group_a, group_b, "fraction", alpha, dependent=True, **kwargs
            )
            effect: float = self.calculate_effect(group_a, group_b, "relative")
            return get_results_dict(alpha, pvalue, effect, conf_int)
        return super().get_results(group_a, group_b, alpha, effect_type, **kwargs)


class MannWhitneyCriterion(ABStatCriterion):
    """
    Unit for Mann-Whitney U test.
    """

    implemented_effect_types: List = ["absolute"]

    def calculate_pvalue(self, group_a: np.ndarray, group_b: np.ndarray, effect_type: str = "absolute", **kwargs):
        if effect_type == "absolute":
            return sps.mannwhitneyu(x=group_a, y=group_b, **kwargs).pvalue
        else:
            raise ValueError(self._send_type_error_msg())

    def calculate_effect(self, group_a: np.ndarray, group_b: np.ndarray, effect_type: str = "absolute"):
        if effect_type == "absolute":
            return np.median(group_b, axis=0) - np.median(group_a, axis=0)
        else:
            raise ValueError(self._send_type_error_msg())

    def calculate_conf_interval(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        alpha: types.StatErrorType = np.array([0.05]),
        effect_type: str = "absolute",
        **kwargs,
    ):
        if isinstance(alpha, float):
            alpha = np.array([alpha])
        if effect_type == "absolute":
            return [(None, None)] * len(alpha)
        else:
            raise ValueError(self._send_type_error_msg())


class WilcoxonCriterion(ABStatCriterion):
    """
    Unit for Wilcoxon paired test.
    """

    implemented_effect_types: List = ["absolute"]

    def calculate_pvalue(self, group_a: np.ndarray, group_b: np.ndarray, effect_type: str = "absolute", **kwargs):
        if effect_type == "absolute":
            return sps.wilcoxon(x=group_a, y=group_b, **kwargs).pvalue
        else:
            raise ValueError(self._send_type_error_msg())

    def calculate_effect(self, group_a: np.ndarray, group_b: np.ndarray, effect_type: str = "absolute"):
        if effect_type == "absolute":
            return np.median(group_b, axis=0) - np.median(group_a, axis=0)
        else:
            raise ValueError(self._send_type_error_msg())

    def calculate_conf_interval(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        alpha: types.StatErrorType = np.array([0.05]),
        effect_type: str = "absolute",
        **kwargs,
    ):
        if effect_type == "absolute":
            return [(None, None)] * len(alpha)
        else:
            raise ValueError(self._send_type_error_msg())


class ShapiroCriterion(StatCriterion):
    """
    Unit for Shapiro-Wilk test.
    """
