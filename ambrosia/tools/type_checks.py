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

import functools
from typing import Iterable, List

import pandas as pd

from ambrosia import types

split_methods_list: List[str] = ["hash", "metric", "simple", "dim_decrease"]
metric_methods_list: List[str] = ["equal_size", "fast", "cluster"]
norm_list = ["l2", "l1"]


def check_type_decorator(type_checker=lambda set_value: set_value):
    """
    Decorator for setter method.
    If set_value is None => self.field = None
    Else using type_checker function set result of type_checker(set_value)
    """

    def inner_decorator(method):
        @functools.wraps(method)
        def wrapper(self, set_value):
            if set_value is None:
                method(self, None)
            else:
                method(self, type_checker(set_value))

        return wrapper

    return inner_decorator


def none_check_decorator(function):
    """
    Decorator for type checkers.
    If argument is None return None
    else return result of checker
    """

    @functools.wraps(function)
    def wrapper(argument):
        if argument is None:
            return None
        else:
            return function(argument)

    return wrapper


@none_check_decorator
def check_type_dataframe(dataframe: types.PassedDataType) -> types.PassedDataType:
    if isinstance(dataframe, str):
        if dataframe.endswith(".csv"):
            return pd.read_csv(dataframe)
        else:
            raise ValueError("``dataframe`` string value must be a link to .csv file")
    elif isinstance(dataframe, pd.DataFrame):
        return dataframe
    elif isinstance(dataframe, types.SparkDataFrame):
        return dataframe
    else:
        raise TypeError("``dataframe`` variable must be a pd.DataFrame object or a link to .csv file")


@none_check_decorator
def check_type_id_column(id_column: types.ColumnNameType) -> types.ColumnNameType:
    if isinstance(id_column, str):
        return id_column
    else:
        raise TypeError("id_column variable must be string")


@none_check_decorator
def check_type_id_columns(id_columns: types.ColumnNamesType) -> types.ColumnNamesType:
    if isinstance(id_columns, str):
        return [id_columns]
    elif isinstance(id_columns, Iterable):
        return id_columns
    else:
        raise TypeError("id_columns variable must be a string or an Iterable object")


@none_check_decorator
def check_type_group_size(groups_size: int) -> int:
    if isinstance(groups_size, int):
        return groups_size
    elif isinstance(groups_size, float):
        return int(groups_size)
    else:
        raise TypeError("groups_size variable must be int or float")


@none_check_decorator
def check_type_test_group_ids(test_group_ids: types.IndicesType) -> types.IndicesType:
    if isinstance(test_group_ids, Iterable):
        return test_group_ids
    else:
        raise TypeError("test_group_ids variable must be Iterable")


@none_check_decorator
def check_type_fit_columns(fit_columns: types.ColumnNamesType) -> types.ColumnNamesType:
    if isinstance(fit_columns, str):
        return [fit_columns]
    elif isinstance(fit_columns, Iterable):
        return fit_columns
    else:
        raise TypeError("fit_columns variable must be a string or an Iterable object")


@none_check_decorator
def check_type_strat_columns(strat_columns: types.ColumnNamesType) -> types.ColumnNamesType:
    if isinstance(strat_columns, str):
        return [strat_columns]
    elif isinstance(strat_columns, Iterable):
        return strat_columns
    else:
        raise TypeError("strat_columns variable must be a string or an Iterable object")


@none_check_decorator
def check_type_salt(salt: str) -> str:
    if isinstance(salt, str):
        return salt
    else:
        raise TypeError("salt variable must be a string")


@none_check_decorator
def check_split_method_value(split_method: str) -> str:
    if isinstance(split_method, str):
        if split_method in split_methods_list:
            return split_method
        else:
            raise ValueError(f'Choose correct split method, from {", ".join(split_methods_list)}')
    else:
        raise TypeError(f'method variable must be a string and from{", ".join(split_methods_list)}')


@none_check_decorator
def check_metric_method_value(method_metric: str) -> None:
    if isinstance(method_metric, str):
        if method_metric in metric_methods_list:
            return method_metric
        else:
            raise ValueError(f'Choose correct method_metric, from {", ".join(metric_methods_list)}')
    else:
        raise TypeError(f'method_metric variable must be a string and from{", ".join(metric_methods_list)}')


@none_check_decorator
def check_norm_value(norm: str) -> None:
    if isinstance(norm, str):
        if norm in norm_list:
            return norm
        else:
            raise ValueError(f'Choose correct norm, from {", ".join(norm_list)}')
    else:
        raise TypeError(f'norm variable must be a string and from{", ".join(norm_list)}')
