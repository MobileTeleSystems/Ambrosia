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
Module contains `Preprocessor` class that combines all data preprocessing
methods in one single chain pipeline. The resulting pipeline allows one to
consistently apply the desired transformations to the data, including outliers
removal, data aggregation and target metric transformations for the variance
reduction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ambrosia import types
from ambrosia.preprocessing.aggregate import AggregatePreprocessor
from ambrosia.preprocessing.cuped import Cuped, MultiCuped
from ambrosia.preprocessing.robust import RobustPreprocessor


class Preprocessor:
    """
    Preprocessor class, implementation is based on the chain pattern.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Table with data used for further transformations.
    verbose : bool, default: ``True``
        If ``True`` will print in sys.stdout the information
        about the variance reduction.

    Attributes
    ----------
    dataframe : pd.DataFrame
        Table with data for transformations.
    transformers: List of transformations
        List of transformation that have been called before.
    verbose : bool
        Verbose info flag.

    Examples
    --------
    >>> transformer = Preprocessor(dataframe)
    >>> transformer.aggregate(aggregate_params)
    >>>            .robust(robust_params)
    >>>            .cuped(cuped_params)
    >>>            .data()

    Methods
    -------
    data(copy=True)
        Returns a copy or a link for the stored dataframe.
    cuped(target, by, name, load_path)
        Make cuped transformation for stored dataframe.
    aggregate(groupby_columns, categorial_method, real_method, agg_params, real_cols, categorial_cols)
        Aggreagate data by columns.
    robust(column_name, alpha=0.05)
        Make a robust transformation.
    transformations()
        Returns a list of transformations.
    """

    def __len__(self) -> int:
        return len(self.dataframe)

    def __init__(self, dataframe: pd.DataFrame, verbose: bool = True) -> None:
        self.dataframe = dataframe.copy()
        self.transformers = []
        self.verbose = verbose

    def data(self, copy: bool = True):
        """
        Return the inner dataframe.

        Use after all transformations to get transformed data.

        Parameters
        ----------
        copy : bool, default: ``True``
            If true returns copy, otherwise link
        """
        return self.dataframe if copy else self.dataframe.copy()

    def cuped(
        self,
        target: types.ColumnNameType,
        by: types.ColumnNameType,
        name: Optional[str] = None,
        load_path: Optional[Path] = None,
    ) -> Preprocessor:
        """
        Make CUPED transformation for the chosen column.

        Parameters
        ----------
        target : ColumnNameType
            Column from the dataframe, for which CUPED transformation will be
            applied.
        by : ColumnNameType
            Covariance column in the dataframe.
        name : str
            Name for the new transformed target column, if is not defined
            it will be generated automatically.
        load_path : Path, optional
            Path to a json file with parameters.
        """
        transformer = Cuped(self.dataframe, target, verbose=self.verbose)
        if load_path is None:
            transformer.fit_transform(by, inplace=True, name=name)
        else:
            transformer.load_params(load_path)
            transformer.transform(by, inplace=True, name=name)
        self.transformers.append(transformer)
        return self

    def multicuped(
        self,
        target: types.ColumnNameType,
        by: types.ColumnNamesType,
        name: Optional[str] = None,
        load_path: Optional[Path] = None,
    ) -> Preprocessor:
        """
        Make Multi CUPED transformation for the chosen column.

        Parameters
        ----------
        target : ColumnNameType
            Column from the dataframe, for which CUPED transformation will be
            applied.
        by : ColumnNameType
            Covariance columns in the dataframe.
        name : str
            Name for the new transformed target column, if is not defined
            it will be generated automatically.
        load_path : Path, optional
            Path to a json file with parameters.
        """
        transformer = MultiCuped(self.dataframe, target, verbose=self.verbose)
        if load_path is None:
            transformer.fit_transform(by, inplace=True, name=name)
        else:
            transformer.load_params(load_path)
            transformer.transform(by, inplace=True, name=name)
        self.transformers.append(transformer)
        return self

    def aggregate(
        self,
        groupby_columns: types.ColumnNamesType,
        categorial_method: types.MethodType = "mode",
        real_method: types.MethodType = "sum",
        agg_params: Optional[Dict] = None,
        real_cols: Optional[types.ColumnNamesType] = None,
        categorial_cols: Optional[types.ColumnNamesType] = None,
    ) -> Preprocessor:
        """
        Make an aggregation of the data frame.

        Parameters
        ----------
        groupby_columns : List of columns
            Columns for GROUP BY.
        categorial_method : String or callable
            Aggregation method  that will be applied for all selected
            categorial variables.
        real_method : types.MethodType, default: ``"sum"``
            Aggregation method  that will be applied for all selected
            real variables.
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
        """
        transformer = AggregatePreprocessor(categorial_method, real_method)
        self.dataframe = transformer.run(self.dataframe, groupby_columns, agg_params, real_cols, categorial_cols)
        self.transformers.append(transformer)
        return self

    def robust(self, column_names: types.ColumnNamesType, alpha: float = 0.05) -> Preprocessor:
        """
        Make a robust transformation.

        Remove objects from the dataframe which are in the head and tail alpha
        parts of chosen metrics distributions.

        Parameters
        ----------
        column_names : ColumnNamesType
            One or number of columns in the dataframe.
        alpha : float, default: ``0.05``
            The percentage of removed data from head and tail.
        """
        transformer = RobustPreprocessor(self.dataframe, verbose=self.verbose)
        transformer.run(column_names, alpha, inplace=True)
        self.transformers.append(transformer)
        return self

    def transformations(self) -> List:
        """
        List of all transformations which were called.
        """
        return self.transformers
