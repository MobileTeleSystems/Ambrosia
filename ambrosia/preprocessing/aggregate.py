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
Module contains class for data aggregation during a preprocessing task.
"""
import copy
from typing import Any, Dict, Optional

import pandas as pd

from ambrosia import types


class AggregatePreprocessor:
    """
    Preprocessing class for data aggregation.

    Can group data by multiple columns and aggregate it using methods
    for real and categorial features.

    Parameters
    ----------
    categorial_method : types.MethodType, default: ``"mode"``
        Aggregation method for categorial variables that
        will become as a default behavior.
    real_method : types.MethodType, default: ``"sum"``
        Aggregation method for real variables that
        will become as a default behavior.

    Attributes
    ----------
    categorial_method : types.MethodType
        Default aggregation method for categorial variables.
    real_method : types.MethodType
        Default aggregation method for real variables.
    groupby_columns : types.ColumnNamesType
        Columns which were used for groupping in the last aggregation.
        Get value after run() / transform() method call.
    agg_params : Dict
        Dictionary with aggregation rules which was used in the last aggregation.
        Get value after run() / transform() method call.
    """

    @staticmethod
    def __mode_calculation(values: pd.Series) -> Any:
        """
        Mode function for aggregation.
        """
        return values.value_counts().index[0]

    @staticmethod
    def __simple_agg(values: pd.Series) -> Any:
        """
        Simple aggregation, just picks the first element.
        """
        return values.iloc[0]

    @staticmethod
    def __transform_agg_param(aggregation_method: types.MethodType) -> types.MethodType:
        """
        Invoke an aggregation callable function by given string alias.
        """
        if aggregation_method == "mode":
            return AggregatePreprocessor.__mode_calculation
        if aggregation_method == "simple":
            return AggregatePreprocessor.__simple_agg
        return aggregation_method

    @staticmethod
    def __transform_params(dataframe: pd.DataFrame, aggregation_params: Dict) -> Dict:
        """
        Iteratively apply transformations specified by aggragation parameters.
        """
        agg_params = copy.deepcopy(aggregation_params)
        for column, method in agg_params.items():
            if column not in dataframe.columns:
                raise ValueError(f"{column} not in columns!")
            agg_params[column] = AggregatePreprocessor.__transform_agg_param(method)
        return agg_params

    def __init__(self, categorial_method: types.MethodType = "mode", real_method: types.MethodType = "sum"):
        self.categorial_method = self.__transform_agg_param(categorial_method)
        self.real_method = self.__transform_agg_param(real_method)
        self.agg_params = None
        self.groupby_columns = None

    def __real_case_step(
        self,
        dataframe: pd.DataFrame,
        agg_params: Optional[Dict] = None,
        real_cols: Optional[types.ColumnNamesType] = None,
    ) -> None:
        """
        A private method containing aggregation parameters filling logic
        for real metrics.
        """
        if isinstance(real_cols, str):
            real_cols = [real_cols]
        for real_feature in real_cols:
            if real_feature not in dataframe.columns:
                raise ValueError(f"{real_feature} is not in columns!")
            agg_params[real_feature] = self.real_method

    def __categorial_case_step(
        self,
        dataframe: pd.DataFrame,
        agg_params: Optional[Dict] = None,
        categorial_cols: Optional[types.ColumnNamesType] = None,
    ) -> None:
        """
        A private method containing aggregation parameters filling logic
        for categorial metrics.
        """
        if isinstance(categorial_cols, str):
            categorial_cols = [categorial_cols]
        for categorial_feature in categorial_cols:
            if categorial_feature not in dataframe.columns:
                raise ValueError(f"{categorial_feature} is not in columns")
            agg_params[categorial_feature] = self.categorial_method

    def __empty_args_step(
        self,
        dataframe: pd.DataFrame,
        agg_params: Optional[Dict] = None,
        real_cols: Optional[types.ColumnNamesType] = None,
        categorial_cols: Optional[types.ColumnNamesType] = None,
    ) -> None:
        """
        A private method containing aggregation parameters filling logic
        if no aggregation parameters passed.
        """
        if real_cols is not None:
            self.__real_case_step(dataframe, agg_params, real_cols)
        if categorial_cols is not None:
            self.__categorial_case_step(dataframe, agg_params, categorial_cols)

    def run(
        self,
        dataframe: pd.DataFrame,
        groupby_columns: types.ColumnNamesType,
        agg_params: Optional[Dict] = None,
        real_cols: Optional[types.ColumnNamesType] = None,
        categorial_cols: Optional[types.ColumnNamesType] = None,
    ) -> pd.DataFrame:
        """
        The main method of data aggregation transformation.

        Aggregation is performed using aggregation dictionary with
        defined aggregation conditions for each columns of interest,
        or columns lists with default class aggregation behavior.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Table to aggregate.
        groupby_columns : types.ColumnNamesType
            Columns for GROUP BY.
        agg_params : Dict, optional
            Dictionary with aggregation parameters.
        real_cols : types.ColumnNamesType, optional
            Columns with real metrics.
            Overriden by ``agg_params`` parameter and could be passed if
            expected default aggregation behavior.
        categorial_cols : types.ColumnNamesType, optional
            Columns with categorial metrics
            Overriden by ``agg_params`` parameter and could be passed if
            expected default aggregation behavior.

        Returns
        -------
        agg_table : pd.DataFrame
            Aggregated table.
        """
        if agg_params is None and real_cols is None and categorial_cols is None:
            raise ValueError("Set agg_params or pass real_cols and categorial_cols")
        if agg_params is None:
            agg_params = {}
            self.__empty_args_step(dataframe, agg_params, real_cols, categorial_cols)
        else:
            agg_params = AggregatePreprocessor.__transform_params(dataframe, agg_params)
        self.groupby_columns = groupby_columns
        self.agg_params = agg_params
        return dataframe.groupby(groupby_columns, as_index=False).agg(agg_params)

    def transform(
        self, dataframe: pd.DataFrame, groupby_columns: types.ColumnNamesType, agg_params: Dict
    ) -> pd.DataFrame:
        """
        Apply table transformation by its aggregation with specified
        parameters.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Table to aggregate.
        groupby_columns : types.ColumnNamesType
            Columns for GROUP BY.
        agg_params : Dict
            Dictionary with aggregation parameters.

        Returns
        -------
        agg_table : pd.DataFrame
            Aggregated table.
        """
        agg_params = AggregatePreprocessor.__transform_params(dataframe, agg_params)
        self.groupby_columns = groupby_columns
        self.agg_params = agg_params
        return dataframe.groupby(groupby_columns, as_index=False).agg(agg_params)

    def get_params_dict(self) -> Dict:
        """
        Returns dictionary with parameters of the last run() or transform() call.
        """
        if self.agg_params is None:
            raise AttributeError("Firstly use run or transform method")
        if self.groupby_columns is None:
            raise AttributeError("Firstly use run or transform method")
        return {"aggregation_params": self.agg_params, "groupby_columns": self.groupby_columns}
