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

from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import pyspark

# PySpark types

SparkSession = pyspark.sql.session.SparkSession
SparkDataFrame = pyspark.sql.dataframe.DataFrame
SparkColumn = pyspark.sql.column.Column
SparkOrPandas = Union[SparkDataFrame, pd.DataFrame]

# Global types

# Type for passed data
PassedDataType = Union[pd.DataFrame, SparkDataFrame, str]

# Type for selected set of arguments names and values
_UsageArgumentsType = Dict

# Type of the unified sets of arguments prepared for the  further selection
_PrepareArgumentsType = Dict

# Type for columns name
ColumnNameType = str

# Type for columns
ColumnNamesType = Union[ColumnNameType, Iterable[ColumnNameType]]

# Type for metrics name
MetricNameType = ColumnNameType

# Type for metrics names
MetricNamesType = Union[MetricNameType, Iterable[MetricNameType]]

# Type for statistical errors
StatErrorType = Union[Iterable[float], float]

# Type for Statistical Criterion results
StatCriterionResult = Dict[str, Any]


# Tools types

# Type for bootstraped samples A/B
BootstrapedSamplesType = Dict[str, np.ndarray]

# Criterion result (statistic, pvalue)
CriterionResultType = Tuple[np.ndarray, np.ndarray]

# Type for one coinfedence interval
SingleIntervalType = Tuple[float, float]

# Coinfedence intervall type
ManyIntervalType = Tuple[np.ndarray, np.ndarray]

# Type for coinfedence interval - 1 or many
IntervalType = Union[SingleIntervalType, ManyIntervalType]

# Type for criterion
CompoundCriterionType = Union[Callable[[np.ndarray, np.ndarray], CriterionResultType], str]


# Designer types

# Type for size(s) of groups in A/B experiment
SampleSizeType = Union[Iterable[int], int]

# Type for effect(s) in A/B experiment
EffectType = Union[Iterable[float], float]

# Type for the result of designing the experiment via Designer
DesignerResult = Union[pd.DataFrame, Dict[MetricNameType, pd.DataFrame]]


# Splitter types

# Type for indices set of A or B group
IndicesType = Iterable[int]

# Type for the result of splitting data in A/B groups via Splitter
SplitterResult = Union[pd.DataFrame, SparkDataFrame]


# Tester types

# Type for data containg the information about group belonging of experimental objects
GroupsInfoType = pd.DataFrame

# Type for A and B groups data
TwoSamplesType = Tuple[pd.DataFrame, pd.DataFrame]

# Type for pair of labels belongs to A and B groups
GroupLabelsType = Tuple[Any, Any]

# Type for the values of metrics of experimental groups
GroupType = np.ndarray

# Type for experiment measurements packed in dictionary
ExperimentResults = Dict[str, PassedDataType]

# Type for experiment measurement action result for one metric
_SubResultType = Dict[str, Any]

# Type for the result of effect measurement in the A/B experiment via Tester
TesterResult = Union[pd.DataFrame, List[Dict[str, Any]]]


# AggregatePreprocessor types

# Type for set of methods/functions for data aggregation
MethodType = Union[str, Callable[[pd.Series], Any]]
