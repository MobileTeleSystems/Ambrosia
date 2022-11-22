from typing import List, Tuple

import pyspark.sql.functions as F
import scipy.stats as sps

import ambrosia.tools.theoretical_tools as theory_pkg
from ambrosia import types
from ambrosia.tools.ab_abstract_component import ABStatCriterion


class TtestIndCriterionSpark(ABStatCriterion):
    """
    Unit for pyspark independent T-test.
    """

    __implemented_effect_types: List = ["absolute", "relative"]
    __type_error_msg: str = f"Choose effect type from {__implemented_effect_types}"
    __data_parameters = ["mean_group_a", "mean_group_b", "std_group_a", "std_group_b", "nobs_group_a", "nobs_group_b"]

    def __init__(self, cache_parameters: bool = False):
        self.cache_parameters = cache_parameters
        self.parameters_are_cached: bool = False
        self.data_stats = {}
        for param in TtestIndCriterionSpark.__data_parameters:
            self.data_stats[param] = None

    def __calc_and_cache_data_parameters(
        self, group_a: types.SparkDataFrame, group_b: types.SparkDataFrame, column: types.ColumnNameType
    ):
        self.data_stats["mean_group_a"], self.data_stats["std_group_a"] = self.get_stats_from_table(group_a, column)
        self.data_stats["mean_group_b"], self.data_stats["std_group_b"] = self.get_stats_from_table(group_b, column)
        self.data_stats["nobs_group_a"] = group_a.count()
        self.data_stats["nobs_group_b"] = group_b.count()
        self.parameters_are_cached = True

    def __delete_cached_data_parameters(self):
        for stat in self.data_stats:
            self.data_stats[stat] = None

    def get_stats_from_table(
        self, dataframe: types.SparkDataFrame, column: types.ColumnNameType
    ) -> Tuple[float, float]:
        """
        Get table for designing samples size for experiment.
        """
        stats = dataframe.select(F.mean(F.col(column)).alias("mean"), F.stddev(F.col(column)).alias("std")).collect()
        mean = stats[0]["mean"]
        std = stats[0]["std"]
        return mean, std

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
        if self.parameters_are_cached is not True:
            self.__calc_and_cache_data_parameters(group_a, group_b, column)
        if effect_type == "absolute":
            p_value = sps.ttest_ind_from_stats(
                self.data_stats["mean_group_a"],
                self.data_stats["std_group_a"],
                self.data_stats["nobs_group_a"],
                self.data_stats["mean_group_b"],
                self.data_stats["std_group_b"],
                self.data_stats["nobs_group_b"],
                **kwargs,
            ).pvalue
        elif effect_type == "relative":
            p_value = theory_pkg.apply_delta_method_by_stats(**kwargs)
        if self.cache_parameters is not True:
            self.__delete_cached_data_parameters()
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
        alpha: types.StatErrorType = 0.05,
        effect_type: str = "absolute",
        **kwargs,
    ):
        if self.parameters_are_cached is not True:
            self.__calc_and_cache_data_parameters(group_a, group_b, column)
        if effect_type == "absolute":
            return None, None
        elif effect_type == "relative":
            conf_interval, _ = theory_pkg.apply_delta_method(group_a, group_b, "fraction", alpha, **kwargs)
            return conf_interval
        else:
            raise ValueError(TtestIndCriterionSpark.__type_error_msg)
