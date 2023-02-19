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
from typing import Any, Dict, Optional, Union

import pandas as pd

from ambrosia import types
from ambrosia.tools.ab_abstract_component import AbstractFittableTransformer
from ambrosia.tools.back_tools import wrap_cols


class AggregatePreprocessor(AbstractFittableTransformer):
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
        Gets value after fitting the class instance.
    agg_params : Dict
        Dictionary with aggregation rules which was used in the last
        aggregation.
        Gets value after fitting the class instance.
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
                raise ValueError(f"{column} does not exist in the dataframe!")
            agg_params[column] = AggregatePreprocessor.__transform_agg_param(method)
        return agg_params

    def __init__(self, categorial_method: types.MethodType = "mode", real_method: types.MethodType = "sum"):
        self.categorial_method = categorial_method
        self.real_method = real_method
        self.agg_params = None
        self.groupby_columns = None
        super().__init__()

    def __real_case_step(
        self,
        agg_params: Optional[Dict] = None,
        real_cols: Optional[types.ColumnNamesType] = None,
    ) -> None:
        """
        A private method containing aggregation parameters filling logic
        for real metrics.
        """
        real_cols = wrap_cols(real_cols)
        for real_feature in real_cols:
            agg_params[real_feature] = self.real_method

    def __categorial_case_step(
        self,
        agg_params: Optional[Dict] = None,
        categorial_cols: Optional[types.ColumnNamesType] = None,
    ) -> None:
        """
        A private method containing aggregation parameters filling logic
        for categorial metrics.
        """
        categorial_cols = wrap_cols(categorial_cols)
        for categorial_feature in categorial_cols:
            agg_params[categorial_feature] = self.categorial_method

    def __empty_args_step(
        self,
        agg_params: Optional[Dict] = None,
        real_cols: Optional[types.ColumnNamesType] = None,
        categorial_cols: Optional[types.ColumnNamesType] = None,
    ) -> None:
        """
        A private method containing aggregation parameters filling logic
        if no aggregation parameters passed.
        """
        if real_cols is not None:
            self.__real_case_step(agg_params, real_cols)
        if categorial_cols is not None:
            self.__categorial_case_step(agg_params, categorial_cols)

    def get_params_dict(self) -> Dict:
        """
        Returns dictionary with parameters of the last run() or transform() call.
        """
        self._check_fitted()
        return {"aggregation_params": self.agg_params, "groupby_columns": self.groupby_columns}

    def load_params_dict(self, params: Dict) -> None:
        """
        Load prefitted parameters form a dictionary.

        Parameters
        ----------
        params : Dict
            Dictionary with prefitted params.
        """
        if "groupby_columns" in params:
            self.groupby_columns = params["groupby_columns"]
        else:
            raise TypeError(f"params argument must contain: {'column_names'}")
        if "aggregation_params" in params:
            self.agg_params = params["aggregation_params"]
        else:
            raise TypeError(f"params argument must contain: {'aggregation_params'}")
        self.fitted = True

    def fit(
        self,
        dataframe: pd.DataFrame,
        groupby_columns: types.ColumnNamesType,
        agg_params: Optional[Dict] = None,
        real_cols: Optional[types.ColumnNamesType] = None,
        categorial_cols: Optional[types.ColumnNamesType] = None,
    ) -> pd.DataFrame:
        """
        Fit preprocessor with parameters of aggregation.

        Aggregation will be performed using passed dictionary with
        defined aggregation conditions for each columns of interest,
        or lists of columns with default class aggregation behavior.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Table with selected columns.
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
        self : object
            Instance object.
        """
        if agg_params is None and real_cols is None and categorial_cols is None:
            raise ValueError("Set agg_params or pass real_cols and categorial_cols")
        if agg_params is None:
            agg_params = {}
            self.__empty_args_step(agg_params, real_cols, categorial_cols)
        self._check_cols(dataframe, agg_params.keys())
        self.groupby_columns = groupby_columns
        self.agg_params = copy.deepcopy(agg_params)
        self.fitted = True
        return self

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply table transformation by its aggregation with prefitted
        parameters.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Table to aggregate.

        Returns
        -------
        agg_table : pd.DataFrame
            Aggregated table.
        """
        self._check_fitted()
        self._check_cols(dataframe, self.agg_params.keys())
        agg_params = AggregatePreprocessor.__transform_params(dataframe, self.agg_params)
        return dataframe.groupby(self.groupby_columns, as_index=False).agg(agg_params)

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
        groupby_columns: types.ColumnNamesType,
        agg_params: Optional[Dict] = None,
        real_cols: Optional[types.ColumnNamesType] = None,
        categorial_cols: Optional[types.ColumnNamesType] = None,
    ) -> pd.DataFrame:
        """
        Fit preprocessor parameters using given dataframe and aggregate it.

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
        self.fit(dataframe, groupby_columns, agg_params, real_cols, categorial_cols)
        return self.transform(dataframe)
