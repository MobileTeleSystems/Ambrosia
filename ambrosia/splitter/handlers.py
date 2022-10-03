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

Module containes functions that help to deal
with data of different type during the groups split task.

Mainly these functions are used in `Splitter` core class.
"""
from typing import List, Optional

import pandas as pd
import pyspark.sql.functions as spark_funcs

import ambrosia.spark_tools.split_tools as split_spark
import ambrosia.tools.split_tools as split_pandas
from ambrosia import types

AVAILABLE: List[str] = ["pandas", "spark"]
GROUPS_COLUMN: str = "group"
DATA: str = "dataframe"
THREADS: str = "threads"


def add_data_pandas(dataframe: pd.DataFrame, splitted_dataframe: pd.DataFrame, group_label: str) -> pd.DataFrame:
    """
    Add data to splitted dataframe for pandas tables.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Source full dataframe.
    splitted_dataframe : pd.DataFrame
        Part of dataframe, which was splitted into two groups.
    group_label : str
        Group label which will be set.

    Returns
    -------
    customized_table : pd.DataFrame
        Table, where rest of table is added.
    """
    additional_table: pd.DataFrame = dataframe.iloc[dataframe.index.delete(splitted_dataframe.index)].copy()
    additional_table[GROUPS_COLUMN] = group_label
    customized_table = pd.concat([splitted_dataframe, additional_table])
    return customized_table


def add_data_spark(
    dataframe: types.SparkDataFrame,
    splitted_dataframe: types.SparkDataFrame,
    group_label: str,
    id_column: types.ColumnNamesType,
) -> types.SparkDataFrame:
    """
    Add data to splitted dataframe for spark tables.

    Parameters
    ----------
    dataframe : SparkDataFrame
        Source full dataframe.
    splitted_dataframe: SparkDataFrame
        Part of dataframe, which was splitted into two groups.
    group_label : str
        Group label which will be set.
    id_column : ColumnNamesType
        Column with id's.

    Returns
    -------
    customized_table : SparkDataFrame
        Table, where rest of table added.
    """
    new_data = (
        dataframe.join(splitted_dataframe.select(id_column, GROUPS_COLUMN), on=id_column, how="left")
        .where(spark_funcs.col(GROUPS_COLUMN).isNull())
        .withColumn(GROUPS_COLUMN, spark_funcs.lit(group_label))
    )
    return splitted_dataframe.union(new_data)


def add_data_to_splitted(
    dataframe: types.PassedDataType,
    splitted_dataframe: types.SplitterResult,
    group_label: str,
    id_column: Optional[types.ColumnNamesType] = None,
) -> types.SplitterResult:
    """
    Add data to splitted groups.

    Parameters
    ----------
    dataframe : PassedDataType
        Source full dataframe.
    splitted_dataframe :  SplitterResult
        Part of dataframe, which was splitted into two groups.
    group_label : str
        Group label which will be set.
    id_columns : ColumnNamesType, optional
        Columns with id's for spark tables.

    Returns
    -------
    customized_table : SplitterResult
        Table, where rest of table added.
    """
    if isinstance(dataframe, pd.DataFrame):
        return add_data_pandas(dataframe, splitted_dataframe, group_label)
    elif isinstance(dataframe, types.SparkDataFrame):
        return add_data_spark(dataframe, splitted_dataframe, group_label, id_column)
    else:
        raise TypeError(f'Type of table must be one of {", ".join(AVAILABLE)}')


def handle_full_split(
    dataframe: types.PassedDataType,
    splitted_dataframe: types.SplitterResult,
    split_factor: float,
    id_column: Optional[types.ColumnNamesType] = None,
) -> types.SplitterResult:
    """
    Finish split dataframe according to split_factor.
    """
    if split_factor < 0.5:
        group_label = "B"
    else:
        group_label = "A"
    return add_data_to_splitted(dataframe, splitted_dataframe, group_label, id_column)


def data_shape(dataframe: types.PassedDataType) -> int:
    """
    Calculate table length size in cases of different tables.
    """
    if isinstance(dataframe, pd.DataFrame):
        return dataframe.shape[0]
    elif isinstance(dataframe, types.SparkDataFrame):
        return dataframe.count()
    else:
        raise TypeError(f'Type of table must be one of {", ".join(AVAILABLE)}')


def split_data_handler(**kwargs) -> types.SplitterResult:
    """
    Call split function according to table type.
    """
    if isinstance(kwargs[DATA], pd.DataFrame):
        return split_pandas.get_split(**kwargs)
    elif isinstance(kwargs[DATA], types.SparkDataFrame):
        return split_spark.get_split(**kwargs)
    else:
        raise TypeError(f'Type of table must be one of {", ".join(AVAILABLE)}')
