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
from ambrosia.spark_tools.theory import get_stats_from_table
from ambrosia.tools.ab_abstract_component import ABStatCriterion
from ambrosia.tools.configs import Effects
from ambrosia.tools.import_tools import spark_installed

if spark_installed():
    import pyspark.sql.functions as F
    from pyspark.sql.functions import col, row_number
    from pyspark.sql.window import Window


class ABSparkCriterion(ABStatCriterion):
    """
    Abstract class for Spark criteria
    """

    def _init_cache(self) -> None:
        self.parameters_are_cached: bool = False
        self.data_stats = dict()

    def __init__(self, cache_parameters: bool = True) -> None:
        self.cache_parameters = cache_parameters
        self._init_cache()

    def _delete_cached_data_parameters(self) -> None:
        self.parameters_are_cached = False

    def _calc_and_cache_data_parameters(self, *args, **kwargs) -> None:
        """
        Uses for recalc parameters for cache
        """
        pass

    def _recalc_cache(self, *args, **kwargs) -> None:
        if not self.parameters_are_cached:
            self._calc_and_cache_data_parameters(*args, **kwargs)

    def _check_clear_cache(self) -> None:
        if not self.cache_parameters:
            self._delete_cached_data_parameters()

    def _check_effect(self, effect_type: str) -> None:
        Effects.raise_if_value_incorrect_enum(effect_type)

    def get_results(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        column: str,
        alpha: types.StatErrorType = 0.05,
        effect_type: str = "absolute",
        **kwargs,
    ) -> types.StatCriterionResult:
        return {
            "first_type_error": alpha,
            "pvalue": self.calculate_pvalue(group_a, group_b, column=column, effect_type=effect_type, **kwargs),
            "effect": self.calculate_effect(group_a, group_b, column=column, effect_type=effect_type),
            "confidence_interval": self.calculate_conf_interval(
                group_a, group_b, column=column, alpha=alpha, effect_type=effect_type, **kwargs
            ),
        }


class TtestIndCriterionSpark(ABSparkCriterion):
    """
    Unit for pyspark independent T-test.
    """

    __implemented_effect_types: List = ["absolute", "relative"]
    __type_error_msg: str = f"Choose effect type from {__implemented_effect_types}"
    __data_parameters = ["mean_group_a", "mean_group_b", "std_group_a", "std_group_b", "nobs_group_a", "nobs_group_b"]

    def __calc_and_cache_data_parameters(
        self, group_a: types.SparkDataFrame, group_b: types.SparkDataFrame, column: types.ColumnNameType
    ):
        self.data_stats["mean_group_a"], self.data_stats["std_group_a"] = get_stats_from_table(group_a, column)
        self.data_stats["mean_group_b"], self.data_stats["std_group_b"] = get_stats_from_table(group_b, column)
        self.data_stats["nobs_group_a"] = group_a.count()
        self.data_stats["nobs_group_b"] = group_b.count()
        self.parameters_are_cached = True

    def _apply_delta_method(
        self, alpha: types.StatErrorType = (0.05,), **kwargs
    ) -> Tuple[types.ManyIntervalType, float]:
        if not self.parameters_are_cached:
            raise RuntimeError("Incorrect usage, firstly calculate parameters")
        # Transforms std for delta method
        s1 = theory_pkg.unbiased_to_sufficient(self.data_stats["std_group_a"], self.data_stats["nobs_group_a"])
        s2 = theory_pkg.unbiased_to_sufficient(self.data_stats["std_group_b"], self.data_stats["nobs_group_b"])
        return theory_pkg.apply_delta_method_by_stats(
            size=(self.data_stats["nobs_group_a"] + self.data_stats["nobs_group_a"]) // 2,
            mean_group_a=self.data_stats["mean_group_a"],
            mean_group_b=self.data_stats["mean_group_b"],
            var_group_a=s1**2,
            var_group_b=s2**2,
            alpha=np.array(alpha),
            **kwargs,
        )

    def calculate_pvalue(
        self,
        group_a: types.SparkDataFrame,
        group_b: types.SparkDataFrame,
        column: types.ColumnNameType,
        effect_type: str = "absolute",
        **kwargs,
    ):
        if effect_type not in TtestIndCriterionSpark.__implemented_effect_types:
            raise ValueError(TtestIndCriterionSpark.__type_error_msg)
        if not self.parameters_are_cached:
            self.__calc_and_cache_data_parameters(group_a, group_b, column)
        if effect_type == "absolute":
            p_value = sps.ttest_ind_from_stats(
                self.data_stats["mean_group_b"],
                self.data_stats["std_group_b"],
                self.data_stats["nobs_group_b"],
                self.data_stats["mean_group_a"],
                self.data_stats["std_group_a"],
                self.data_stats["nobs_group_a"],
                **kwargs,
            ).pvalue
        elif effect_type == "relative":
            p_value = self._apply_delta_method(**kwargs)[1]
        if not self.cache_parameters:
            self._delete_cached_data_parameters()
        return p_value

    def calculate_effect(
        self,
        group_a: types.SparkDataFrame,
        group_b: types.SparkDataFrame,
        column: types.ColumnNameType,
        effect_type: str = "absolute",
    ):
        if self.parameters_are_cached is not True:
            self.__calc_and_cache_data_parameters(group_a, group_b, column)
        if effect_type == "absolute":
            effect = self.data_stats["mean_group_b"] - self.data_stats["mean_group_a"]
        elif effect_type == "relative":
            effect = (self.data_stats["mean_group_b"] - self.data_stats["mean_group_a"]) / self.data_stats[
                "mean_group_a"
            ]
        else:
            raise ValueError(TtestIndCriterionSpark.__type_error_msg)
        return effect

    def calculate_conf_interval(
        self,
        group_a: types.SparkDataFrame,
        group_b: types.SparkDataFrame,
        column: types.ColumnNameType,
        alpha: types.StatErrorType = (0.05,),
        effect_type: str = "absolute",
        **kwargs,
    ):
        if self.parameters_are_cached is not True:
            self.__calc_and_cache_data_parameters(group_a, group_b, column)
        if effect_type == "absolute":
            alpha_corrected: float = pvalue_pkg.corrected_alpha(alpha, kwargs["alternative"])
            quantiles, sd = theory_pkg.get_ttest_info_from_stats(
                var_a=self.data_stats["std_group_a"] ** 2,
                var_b=self.data_stats["std_group_b"] ** 2,
                n_obs_a=self.data_stats["nobs_group_a"],
                n_obs_b=self.data_stats["nobs_group_b"],
                alpha=alpha_corrected,
            )
            mean = self.data_stats["mean_group_b"] - self.data_stats["mean_group_a"]
            left_ci: np.ndarray = mean - quantiles * sd
            right_ci: np.ndarray = mean + quantiles * sd
            return self._make_ci(left_ci, right_ci, kwargs["alternative"])
        elif effect_type == "relative":
            conf_interval = self._apply_delta_method(alpha, **kwargs)[0]
            return conf_interval
        else:
            raise ValueError(TtestIndCriterionSpark.__type_error_msg)


class TtestRelativeCriterionSpark(ABSparkCriterion):
    """
    Relative ttest for spark
    """

    __add_index_name: str = "__ambrosia_ind"
    __diff: str = "__ambrosia_rel_diff"
    __ord_col: str = "__ambrosia_ord"

    @staticmethod
    def _rename_col(column: str, group: str) -> str:
        return f"__{column}_{group}"

    def _calc_and_cache_data_parameters(
        self, group_a: types.SparkDataFrame, group_b: types.SparkDataFrame, column: types.ColumnNameType
    ) -> None:
        a_ = (
            group_a.withColumn(self.__ord_col, F.lit(1))
            .withColumn(self.__add_index_name, row_number().over(Window().orderBy(self.__ord_col)))
            .withColumnRenamed(column, self._rename_col(column, "a"))
        )
        b_ = (
            group_b.withColumn(self.__ord_col, F.lit(1))
            .withColumn(self.__add_index_name, row_number().over(Window().orderBy(self.__ord_col)))
            .withColumnRenamed(column, self._rename_col(column, "b"))
        )

        n_a_obs: int = group_a.count()
        n_b_obs: int = group_b.count()

        if n_a_obs != n_b_obs:
            raise ValueError("Size of group A and B must be equal")

        both = a_.join(b_, self.__add_index_name, "inner").withColumn(
            self.__diff, col(self._rename_col(column, "b")) - col(self._rename_col(column, "a"))
        )
        self.data_stats["mean"], self.data_stats["std"] = get_stats_from_table(both, self.__diff)
        self.data_stats["n_obs"] = n_a_obs
        self.parameters_are_cached = True

    def calculate_pvalue(
        self,
        group_a: types.SparkDataFrame,
        group_b: types.SparkDataFrame,
        column: types.ColumnNameType,
        effect_type: str = Effects.abs.value,
        **kwargs,
    ):
        self._recalc_cache(group_a, group_b, column)
        if effect_type == Effects.abs.value:
            p_value = theory_pkg.ttest_1samp_from_stats(
                mean=self.data_stats["mean"], std=self.data_stats["std"], n_obs=self.data_stats["n_obs"], **kwargs
            )
        elif effect_type == Effects.rel.value:
            raise NotImplementedError("Will be implemented later")
        self._check_clear_cache()
        return p_value

    def calculate_conf_interval(
        self,
        group_a: types.SparkDataFrame,
        group_b: types.SparkDataFrame,
        alpha: types.StatErrorType,
        effect_type: str,
        **kwargs,
    ) -> List[Tuple]:
        raise NotImplementedError("Will be implemented later")

    def calculate_effect(
        self, group_a: types.SparkDataFrame, group_b: types.SparkDataFrame, column: str, effect_type: str
    ) -> float:
        self._recalc_cache(group_a, group_b, column)
        if effect_type == Effects.abs.value:
            effect: float = self.data_stats["mean"]
        else:
            raise NotImplementedError("Will be implemented later")
        return effect
