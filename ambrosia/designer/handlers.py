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

"""
Handlers for different dataframe types.

Module containes functions and classes that help to deal
with data of different type during the experiment design problem.

These objects are used in `Designer` core class.
"""
import warnings
from typing import List

import pandas as pd
import pyspark.sql.functions as spark_functions

from ambrosia import types
import ambrosia.tools.tools as empiric_pkg
import ambrosia.tools.bin_intervals as bin_pkg
import ambrosia.spark_tools.theory as theory_spark
import ambrosia.spark_tools.empiric as empiric_spark
import ambrosia.tools.theoretical_tools as theory_pkg
from ambrosia.tools.ab_abstract_component import SimpleDesigner
from ambrosia.tools.theoretical_binary import TheoreticalBinary


DATA: str = "dataframe"
AVAILABLE: List[str] = ["pandas", "spark"]
AVAILABLE_TABLES_ERROR = TypeError(f'Type of table must be one of {", ".join(AVAILABLE)}')
THEORY_METHOD: str = "theory_method"


class TheoryHandler(SimpleDesigner):
    """
    Unit for theory design.
    """

    def size_design(self, **kwargs) -> pd.DataFrame:
        return self._handle_cases(theory_pkg.design_groups_size, theory_spark.design_groups_size, **kwargs)

    def effect_design(self, **kwargs) -> pd.DataFrame:
        return self._handle_cases(theory_pkg.design_effect, theory_spark.design_effect, **kwargs)

    def power_design(self, **kwargs) -> pd.DataFrame:
        return self._handle_cases(theory_pkg.design_power, theory_spark.design_power, **kwargs)


class EmpiricHandler(SimpleDesigner):
    """
    Unit for empiric design.
    """

    def size_design(self, **kwargs) -> pd.DataFrame:
        return self._handle_cases(empiric_pkg.get_empirical_table_sample_size, empiric_spark.get_table_size, **kwargs)

    def effect_design(self, **kwargs) -> pd.DataFrame:
        return self._handle_cases(empiric_pkg.get_empirical_mde_table, empiric_spark.get_table_effect, **kwargs)

    def power_design(self, **kwargs) -> pd.DataFrame:
        if isinstance(kwargs[DATA], types.SparkDataFrame):
            kwargs["group_sizes"] = kwargs["sample_sizes_a"]
            del kwargs["sample_sizes_a"]
            del kwargs["sample_sizes_b"]
        return self._handle_cases(empiric_pkg.get_empirical_errors_table, empiric_spark.get_table_power, **kwargs)


def calc_prob_control_class(table: types.PassedDataType, metric: types.MetricNameType) -> float:
    """
    Calculate conversion on binary metric for pandas or Spark dataframe.

    Parameters
    ----------
    table : SparkDataFrame or pd.DataFrame
        Table with binary metric.
    metric : MetricNameType
        Table Column name that containes binary metric of interest.

    Returns
    -------
    p_a : float
        Conversion in control group.
    """
    warning_message_values: str = "Metric values are not binary, choose empiric or theory method!"
    if isinstance(table, pd.DataFrame):
        if not set(table[metric].unique()).issubset({0, 1}):
            warnings.warn(warning_message_values)
        p_a = table[metric].mean()
    else:
        if not set(table.select(metric).distinct().toPandas()[metric]).issubset({0, 1}):
            warnings.warn(warning_message_values)
        p_a = table.select(spark_functions.mean(metric)).collect()[0][0]
    return p_a


class BinaryDesignHandler(SimpleDesigner):
    '''
    Handle method used in binary metrics design
    We can use theoretical transformations from tools.theoretical_binary
    or confidence_interval from tools.bin_intervals

    query -> Designer ---(got binary)--> HandlerWhatToDesign -> BinaryDesignHandler (Bin CI / theory)

    '''

    def _get_solver_theory(self, **kwargs):
        solver = bin_pkg
        if THEORY_METHOD in kwargs:
            method = kwargs[THEORY_METHOD]
            del kwargs[THEORY_METHOD]
            solver = TheoreticalBinary.METHODS[method]()
        return solver, kwargs

    def size_design(self, **kwargs) -> pd.DataFrame:
        solver, kwargs = self._get_solver_theory(**kwargs)
        return solver.get_table_sample_size_on_effect(**kwargs)

    def effect_design(self, **kwargs) -> pd.DataFrame:
        solver, kwargs = self._get_solver_theory(**kwargs)
        return solver.get_table_effect_on_sample_size(**kwargs)

    def power_design(self, **kwargs) -> pd.DataFrame:
        solver, kwargs = self._get_solver_theory(**kwargs)
        return solver.get_table_power_on_size_and_delta(**kwargs)
