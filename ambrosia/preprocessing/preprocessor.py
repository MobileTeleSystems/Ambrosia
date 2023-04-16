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

import inspect
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ambrosia import types
from ambrosia.preprocessing.aggregate import AggregatePreprocessor
from ambrosia.preprocessing.cuped import Cuped, MultiCuped
from ambrosia.preprocessing.robust import IQRPreprocessor, RobustPreprocessor
from ambrosia.preprocessing.transformers import BoxCoxTransformer, LogTransformer


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
    transformers : List of transformations
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
    aggregate(groupby_columns, categorial_method, real_method, agg_params,
              real_cols, categorial_cols)
        Aggreagate data by columns.
    robust(column_names, alpha=0.05)
        Make a robust preprocessing of data.
    iqr(column_names, alpha=0.05)
        Make an IQR preprocessing of data.
    boxcox(column_names, alpha=0.05)
        Make a Box-Cox transformation.
    log(column_names, alpha=0.05)
        Make a log transformation.
    cuped(target, by, name, load_path)
        Make CUPED transformation for the stored dataframe.
    multicuped(target, by, name, load_path)
        Make Multi CUPED transformation for the stored dataframe.
    transformations()
        Returns a list of transformations.
    store_transformations(store_path)
        Store transformations in a json file.
    load_transformations(load_path)
        Load transformations from a json file.
    apply_transformations()
        Apply transformations for the stored dataframe.
    transform_from_config(load_path)
        Transform inner data frame using pre-saved config file.
    """

    def __len__(self) -> int:
        return len(self.dataframe)

    def __init__(self, dataframe: pd.DataFrame, verbose: bool = True) -> None:
        self.dataframe = dataframe.copy()
        self.transformers = []
        self.verbose = verbose

    def data(self, copy: bool = True):
        """
        Return the inner data frame.

        Use after all transformations to get transformed data.

        Parameters
        ----------
        copy : bool, default: ``True``
            If true returns copy, otherwise link

        Returns
        -------
        dataframe : pd.DataFrame
            Table with the modified data after the sequential preprocessing.
        """
        return self.dataframe.copy() if copy else self.dataframe

    def aggregate(
        self,
        groupby_columns: Optional[types.ColumnNamesType] = None,
        categorial_method: types.MethodType = "mode",
        real_method: types.MethodType = "sum",
        agg_params: Optional[Dict] = None,
        real_cols: Optional[types.ColumnNamesType] = None,
        categorial_cols: Optional[types.ColumnNamesType] = None,
        load_path: Optional[Path] = None,
    ) -> Preprocessor:
        """
        Make an aggregation of the dataframe.

        Parameters
        ----------
        groupby_columns : List of columns, optional
            Columns for GROUP BY.
        categorial_method : types.MethodType, default: ``"mode"``
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

        Returns
        -------
        self : Preprocessor
            Instance object
        """
        transformer = AggregatePreprocessor(categorial_method, real_method)
        if load_path is None:
            self.dataframe = transformer.fit_transform(
                self.dataframe, groupby_columns, agg_params, real_cols, categorial_cols
            )
        else:
            transformer.load_params(load_path)
            self.dataframe = transformer.transform(self.dataframe)
        self.transformers.append(transformer)
        return self

    def robust(
        self,
        column_names: Optional[types.ColumnNamesType] = None,
        alpha: Union[float, np.ndarray] = 0.05,
        tail: str = "both",
        load_path: Optional[Path] = None,
    ) -> Preprocessor:
        """
        Make a robust preprocessing of the selected columns to remove outliers.

        Removes objects from the dataframe which are in the head, end or
        both tail parts of the selected metrics distributions.

        Parameters
        ----------
        column_names : ColumnNamesType
            One or number of columns in the dataframe.
        alpha : Union[float, np.ndarray], default: ``0.05``
            The percentage of removed data from head and tail.
        tail : str, default: ``"both"``
            Part of distribution to be removed.
            Can be ``"left"``, ``"right"`` or ``"both"``.
        load_path : Path, optional
            Path to json file with parameters.

        Returns
        -------
        self : Preprocessor
            Instance object
        """
        transformer = RobustPreprocessor(verbose=self.verbose)
        if load_path is None:
            transformer.fit_transform(self.dataframe, column_names, alpha, tail, inplace=True)
        else:
            transformer.load_params(load_path)
            transformer.transform(self.dataframe, inplace=True)
        self.transformers.append(transformer)
        return self

    def iqr(
        self,
        column_names: Optional[types.ColumnNamesType] = None,
        load_path: Optional[Path] = None,
    ) -> Preprocessor:
        """
        Make an IQR preprocessing of the selected columns to remove outliers.

        Removes objects from the dataframe which are behind boxplot maximum
        and minimum of the selected metrics distributions.

        Parameters
        ----------
        column_names : ColumnNamesType, optional
            One or number of columns in the dataframe.
        load_path : Path, optional
            Path to json file with parameters.

        Returns
        -------
        self : Preprocessor
            Instance object
        """
        transformer = IQRPreprocessor(verbose=self.verbose)
        if load_path is None:
            transformer.fit_transform(self.dataframe, column_names, inplace=True)
        else:
            transformer.load_params(load_path)
            transformer.transform(self.dataframe, inplace=True)
        self.transformers.append(transformer)
        return self

    def boxcox(
        self,
        column_names: Optional[types.ColumnNamesType] = None,
        load_path: Optional[Path] = None,
    ) -> Preprocessor:
        """
        Make a Box-Cox transformation on the selected columns.

        Optimal transformation parameters are selected automatically.

        Parameters
        ----------
        column_names : ColumnNamesType, optional
            One or number of columns in the dataframe.
        load_path : Path, optional
            Path to json file with parameters.

        Returns
        -------
        self : Preprocessor
            Instance object
        """
        transformer = BoxCoxTransformer()
        if load_path is None:
            transformer.fit_transform(self.dataframe, column_names, inplace=True)
        else:
            transformer.load_params(load_path)
            transformer.transform(self.dataframe, inplace=True)
        self.transformers.append(transformer)
        return self

    def log(
        self,
        column_names: Optional[types.ColumnNamesType] = None,
        load_path: Optional[Path] = None,
    ) -> Preprocessor:
        """
        Make a logarithmic transformation on the selected columns.

        Parameters
        ----------
        column_names : ColumnNamesType, optional
            One or number of columns in the dataframe.
        load_path : Path, optional
            Path to json file with parameters.

        Returns
        -------
        self : Preprocessor
            Instance object
        """
        transformer = LogTransformer()
        if load_path is None:
            transformer.fit_transform(self.dataframe, column_names, inplace=True)
        else:
            transformer.load_params(load_path)
            transformer.transform(self.dataframe, inplace=True)
        self.transformers.append(transformer)
        return self

    def cuped(
        self,
        target: Optional[types.ColumnNameType] = None,
        by: Optional[types.ColumnNameType] = None,
        transformed_name: Optional[types.ColumnNameType] = None,
        load_path: Optional[Path] = None,
    ) -> Preprocessor:
        """
        Make CUPED transformation on the selected column.

        Parameters
        ----------
        target : ColumnNameType
            Column from the dataframe, for which CUPED transformation will be
            applied.
        by : ColumnNameType
            Covariance column in the dataframe.
        transformed_name : types.ColumnNameType, optional
            Name for the new transformed target column, if is not defined
            it will be generated automatically.
        load_path : Path, optional
            Path to json file with parameters.

        Returns
        -------
        self : Preprocessor
            Instance object
        """
        transformer = Cuped(verbose=self.verbose)
        if load_path is None:
            transformer.fit_transform(self.dataframe, target, by, transformed_name, inplace=True)
        else:
            transformer.load_params(load_path)
            transformer.transform(self.dataframe, inplace=True)
        self.transformers.append(transformer)
        return self

    def multicuped(
        self,
        target: Optional[types.ColumnNameType] = None,
        by: Optional[types.ColumnNamesType] = None,
        transformed_name: Optional[types.ColumnNameType] = None,
        load_path: Optional[Path] = None,
    ) -> Preprocessor:
        """
        Make Multi CUPED transformation on the selected column.

        Parameters
        ----------
        target : ColumnNameType
            Column from the dataframe, for which CUPED transformation will be
            applied.
        by : ColumnNameType
            Covariance columns in the dataframe.
        transformed_name : types.ColumnNameType, optional
            Name for the new transformed target column, if is not defined
            it will be generated automatically.
        load_path : Path, optional
            Path to json file with parameters.

        Returns
        -------
        self : Preprocessor
            Instance object
        """
        transformer = MultiCuped(verbose=self.verbose)
        if load_path is None:
            transformer.fit_transform(self.dataframe, target, by, transformed_name, inplace=True)
        else:
            transformer.load_params(load_path)
            transformer.transform(self.dataframe, inplace=True)
        self.transformers.append(transformer)
        return self

    def transformations(self) -> List:
        """
        List of all transformations which were called.

        Returns
        -------
        transformers : List[object]
            List of executed transformations
        """
        return self.transformers

    def store_transformations(self, store_path: Path) -> None:
        """
        Store transformations with parameters in the json file.

        Parameters
        ----------
        store_path : Path
            Path to a json file where transformations will be stored
        """
        if len(self.transformers) == 0:
            raise ValueError("No transformations have been made yet.")
        transformations_counter = {}
        transformations_config = {}
        for transformer in self.transformers:
            alias = transformer.__class__.__name__
            if alias in transformations_counter:
                transformations_counter[alias] += 1
            else:
                transformations_counter[alias] = 1
            alias += "_" + str(transformations_counter[alias])
            transformations_config[alias] = transformer.get_params_dict()

        with open(store_path, "w+") as file:
            json.dump(transformations_config, file)

    def load_transformations(self, load_path: Path) -> None:
        """
        Load pre-saved transformations from the json file.

        Parameters
        ----------
        load_path : Path
            Path to a json file where transformations are stored
        """
        with open(load_path, "r+") as file:
            params = json.load(file)
        for key, value in params.items():
            class_alias = "".join(filter(str.isalpha, key))
            transformer = getattr(sys.modules[__name__], class_alias)
            kwargs = {}
            if "verbose" in inspect.signature(transformer).parameters:
                kwargs["verbose"] = self.verbose
            transformer = transformer(**kwargs)
            transformer.load_params_dict(value)
            self.transformers.append(transformer)

    def apply_transformations(self) -> pd.DataFrame:
        """
        Apply all transformations to the inner data frame.

        Returns
        -------
        dataframe : pd.DataFrame
            Transformed inner data frame
        """
        for transformer in self.transformers:
            if isinstance(transformer, AggregatePreprocessor):
                self.dataframe = transformer.transform(self.dataframe)
            else:
                transformer.transform(self.dataframe, inplace=True)
        return self.data()

    def transform_from_config(self, load_path: Path) -> pd.DataFrame:
        """
        Run transformations from the config file on the internal data frame.

        Parameters
        ----------
        load_path : Path
            Path to a json file where transformations are stored.

        Returns
        -------
        dataframe : pd.DataFrame
            Transformed inner data frame
        """
        self.load_transformations(load_path)
        return self.apply_transformations()
