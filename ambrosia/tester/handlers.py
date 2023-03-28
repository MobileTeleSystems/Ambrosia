import enum
import typing as tp

import numpy as np
import pandas as pd

import ambrosia.spark_tools.stat_criteria as spark_crit_pkg
import ambrosia.tools.stat_criteria as criteria_pkg
from ambrosia import types
from ambrosia.tools.ab_abstract_component import StatCriterion, choose_on_table
from ambrosia.tools.import_tools import spark_installed

# Avoid errors with not installed spark

if spark_installed():
    import pyspark.sql.functions as spark_funcs


def filter_spark_and_make_groups(
    dataframe: types.SparkDataFrame,
    df_mapping: types.GroupsInfoType,
    column_groups: types.ColumnNameType,
    group_labels: types.GroupLabelsType,
    id_column: types.ColumnNameType,
) -> types.TwoSamplesType:
    if dataframe is None:
        return None
    if df_mapping is not None:
        raise NotImplementedError("For spark tables df_mapping can't be used. Use column_groups instead")
    group_labels = dataframe.select(column_groups).distinct().collect()
    group_labels = sorted([label[column_groups] for label in group_labels])
    experiment_results: types.ExperimentResults = {
        label: dataframe.where(spark_funcs.col(column_groups) == label) for label in group_labels
    }
    return experiment_results


class PandasCriteria(enum.Enum):
    ttest: StatCriterion = criteria_pkg.TtestIndCriterion
    ttest_rel: StatCriterion = criteria_pkg.TtestRelCriterion
    mw: StatCriterion = criteria_pkg.MannWhitneyCriterion
    wilcoxon: StatCriterion = criteria_pkg.WilcoxonCriterion


class SparkCriteria(enum.Enum):
    ttest: StatCriterion = spark_crit_pkg.TtestIndCriterionSpark
    ttest_rel: StatCriterion = None  # spark_crit_pkg.TtestRelativeCriterionSpark it's in development now
    mw: StatCriterion = None
    wilcoxon: StatCriterion = None


class TheoreticalTesterHandler:
    def __init__(
        self, group_a, group_b, column: str, alpha: np.ndarray, effect_type: str, criterion: StatCriterion, **kwargs
    ):
        self.group_a = group_a
        self.group_b = group_b
        self.column = column
        self.alpha = alpha
        self.effect_type = effect_type
        self.criterion = criterion
        self.kwargs = kwargs

    def _correct_criterion(self, criterion: tp.Any) -> bool:
        return isinstance(criterion, StatCriterion)

    def _raise_correct_criterion(self, criterion: tp.Any) -> None:
        if not self._correct_criterion(criterion):
            raise TypeError("Criterion must be inherited from StatCriterion")

    def get_criterion(self, criterion: str, data_example: types.SparkOrPandas):
        if not isinstance(criterion, str):
            return criterion
        CriteriaEnum = choose_on_table([PandasCriteria, SparkCriteria], data_example)
        criterion = CriteriaEnum[criterion].value
        if criterion is None:
            raise NotImplementedError("This criterion will be implemented later")
        return criterion()

    def _set_kwargs(self):
        if isinstance(self.group_a, pd.DataFrame):
            self.group_a = self.group_a[self.column].values
            self.group_b = self.group_b[self.column].values
        elif isinstance(self.group_a, types.SparkDataFrame):
            self.kwargs["column"] = self.column
        self.kwargs["alpha"] = self.alpha
        self.kwargs["effect_type"] = self.effect_type
        self.kwargs["group_a"] = self.group_a
        self.kwargs["group_b"] = self.group_b

    def solve(self) -> types._SubResultType:
        criterion: tp.Union[str, StatCriterion] = self.criterion if self.criterion is not None else "ttest"
        criterion = self.get_criterion(criterion, self.group_a)
        self._raise_correct_criterion(criterion)
        self._set_kwargs()
        return criterion.get_results(**self.kwargs)
