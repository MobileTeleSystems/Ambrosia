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
Module contains CUPED-based data transformation methods for the experiment
acceleration.
"""
import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from ambrosia import types
from ambrosia.tools.ab_abstract_component import AbstractVarianceReduction


class Cuped(AbstractVarianceReduction):
    """
    Class for data CUPED transformation.

    https://towardsdatascience.com/how-to-double-a-b-testing-speed-with-cuped-f80460825a90
    Y_hat = Y - theta * X
    theta := cov(X, Y) / Var(Y)
    It is important, that the mean covariance metric did not change over time!!!

    Parameters
    ----------
    dataframe : pd.DataFrame
        Table with data for CUPED transformation.
    target_column : ColumnNameType
        Column from the dataframe, for which CUPED transformation will be
        applied.
    verbose : bool, default: ``True``
        If ``True`` will print in sys.stdout the information
        about the variance reduction.

    Attributes
    ----------
    dataframe : pd.DataFrame
        Table with data for CUPED transformation.
    target_column : ColumnNameType
        Column from the dataframe, for which CUPED transformation will be
        applied.
    verbose : bool
        Verbose info flag.
    theta : float
        Linear coefficient for CUPED transformation.
    bias : float
        Bias value for mean equality.
    fitted : bool
        Flag if class was fitted.

    Examples
    --------
    Suppose we have the dataframe with users info which contains two columns:
    a "target" columns and a column with metric "income". Let us can assume,
    that over time, the average of the "income" values do not change. Then, we
    can use CUPED transformation based on "income" data to reduce "target"
    column variation.

    >>> cuped_transformer = Cuped(dataframe, 'target', verbose=True)
    >>> cuped_transformer.fit_transform(
    >>>     covariate_column='income',
    >>>     inplace=True,
    >>>     name='cuped_target'
    >>> )

    Now in the dataframe a new column "cuped_target" appeared, we can use it
    to design our experiment and estimate variance reduction. For further CUPED
    usagein the future experiment, let us store the parameters:

    >>> cuped_transformer.store_params('cuped_transform_params.json')

    Now we conduct an experiment and want to transform our data to reduce its
    variation:

    >>> cuped_transformation = Cuped(exp_results, 'target')
    >>> cuped_transformation.load_params('cuped_transform_params.json')
    >>> cuped_transformation.transform(
    >>>     covariate_column='income',
    >>>     inplace=True,
    >>>     name='cuped_transformed'
    >>> )

    Methods
    -------
    get_params_dict()
        Returns dictionary with params if fit() method has been previously
        called.
    load_params_dict(params)
        Load params from a dictionary.
    store_params(store_path)
        Store params to json file if fit() method has been previously called.
    load_params(load_path)
        Load params from a json file.
    fit(covariate_column)
        Fit model using a specific covariate column.
    transform(covariate_column, inplace, name)
        Transform target column after a class instance fitting.
    fit_transform(covariate_column, inplace, name)
        Combination of fit() and transform() methods.
    """

    THETA_NAME: str = "theta"
    BIAS_NAME: str = "bias"

    def __init__(self, dataframe: pd.DataFrame, target_column: types.ColumnNameType, verbose: bool = True) -> None:
        super().__init__(dataframe, target_column, verbose)
        self.theta = None
        self.bias = None
        self.cov: pd.DataFrame = dataframe.cov()

    def __str__(self) -> str:
        return f"СUPED for {self.target_column}"

    def get_params_dict(self) -> Dict:
        """
        Returns a dictionary with params.
        """
        self._check_fitted()
        return {
            Cuped.THETA_NAME: self.theta,
            Cuped.BIAS_NAME: self.bias,
        }

    def load_params_dict(self, params: Dict) -> None:
        """
        Parameters
        ----------
        params : Dict
            Dictionary with params.
        """
        if Cuped.THETA_NAME in params:
            self.theta = params[Cuped.THETA_NAME]
        else:
            raise TypeError(f"params argument must contain: {Cuped.THETA_NAME}")
        if Cuped.BIAS_NAME in params:
            self.bias = params[Cuped.BIAS_NAME]
        else:
            raise TypeError(f"params argument must contain: {Cuped.BIAS_NAME}")
        self.fitted = True

    def store_params(self, store_path: Path) -> None:
        """
        Parameters
        ----------
        store_path : Path
            Path where parameters will be stored in a json format.
        """
        with open(store_path, "w+") as file:
            json.dump(self.get_params_dict(), file)

    def load_params(self, load_path: Path) -> None:
        """
        Parameters
        ----------
        load_path : Path
            Path to a json file with parameters.
        """
        with open(load_path, "r+") as file:
            params = json.load(file)
            self.load_params_dict(params)

    def fit(self, covariate_column: types.ColumnNameType) -> None:
        """
        Fit to calculate CUPED parameters for target column using given
        covariate column.

        Parameters
        ----------
        covariate_column : ColumnNameType
            Column which will be used as the covariate in CUPED transformation.
        """
        variance_covariate: float = self.cov.loc[covariate_column, covariate_column]
        self.theta = self.cov.loc[self.target_column, covariate_column] / (super().EPSILON + variance_covariate)
        self.bias = np.mean(self.df[covariate_column])
        self.fitted = True

    def __call__(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        y_hat: np.ndarray = y - self.theta * (X - self.bias)
        return y_hat

    def transform(  # pylint: disable=W0237
        self,
        covariate_column: types.ColumnNameType,
        inplace: bool = False,
        name: Optional[str] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Make CUPED transformation for target column.

        Could be performed inplace or not.

        Parameters
        ----------
        covariate_column : ColumnNameType
            Column which will be used as the covariate.
        inplace : bool, default: ``False``
            If is ``True``, then method returns ``None`` and
            sets a new column for the original dataframe.
            Otherwise return copied dataframe with a new column.
        name : str
            Name for the new transformed target column, if is not defined
            it will be generated automatically.
        """
        self._check_columns([covariate_column])
        old_variance: float = self.cov.loc[self.target_column, self.target_column]
        new_target: np.ndarray = self(self.df[self.target_column], self.df[covariate_column])
        new_variance: float = np.var(new_target)
        if self.verbose:
            self._verbose(old_variance, new_variance)
        return self._return_result(new_target, inplace, name)

    def fit_transform(
        self,
        covariate_column: types.ColumnNameType,
        inplace: bool = False,
        name: Optional[str] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Combination of fit() and transform() methods.

        Parameters
        ----------
        covariate_column : ColumnNameType
            Column which will be used as the covariate.
        inplace : bool, default: ``False``
            If is ``True``, then method returns ``None`` and
            sets a new column for the original dataframe.
            Otherwise return copied dataframe with a new column.
        name : str
            Name for the new transformed target column, if is not defined
            it will be generated automatically.
        """
        self.fit(covariate_column)
        return self.transform(covariate_column, inplace, name)


class MultiCuped(AbstractVarianceReduction):
    """
    Class for data Multi CUPED transformation.

    Y_hat = Y - X theta (Matrix multiplication)
    theta := argmin Var (Y - X theta)
    It is important, that the mean covariance metric do not change over time!!!


    Parameters
    ----------
    dataframe : pd.DataFrame
        Table with data for Multi CUPED transformation.
    target_column : ColumnNameType
        Column from the dataframe, for which CUPED transformation will be
        applied.
    verbose : bool, default: ``True``
        If ``True`` will print in sys.stdout the information
        about the variance reduction.

    Attributes
    ----------
    dataframe : pd.DataFrame
        Table with data for Multi CUPED transformation.
    target_column : ColumnNameType
        Column from the dataframe, for which CUPED transformation will be
        applied.
    verbose : bool
        Verbose info flag.
    theta : float
        Linear coefficient for Multi CUPED transformation.
    bias : float
        Bias value for mean equality.
    fitted : bool
        Flag if class was fitted.

    Examples
    --------
    We have dataframe with users info with column 'target' and
    columns 'income' and 'age'. We can assume, that over time,
    the average of this covariate values does not change. Then, we can use
    multi cuped transformation to reduce variation.

    Suppose we have the dataframe with users info which contains two columns:
    a "target" columns and columns "income" and "age". Let us can assume,
    that over time, the average of the "income" and "age" values do not change.
    Then, we can use Multi CUPED transformation based on "income" and "age"
    data in order to reduce "target" column variation.

    >>> cuped_transformer = MultiCuped(dataframe, 'target', verbose=True)
    >>> cuped_transformer.fit_transform(
    >>>     ['income', 'age'],
    >>>     inplace=True,
    >>>     name='cuped_target'
    >>> )

    Now in the dataframe a new column "cuped_target" appeared, we can use it
    to design our experiment and estimate variance reduction. For further
    Multi CUPED usage in the future experiment, let us store the parameters:

    >>> cuped_transformer.store_params('cuped_transform_params.json')

    Now we conduct an experiment and want to transform our data to reduce its
    variation:

    >>> cuped_transformation = MultiCuped(exp_results, 'target')
    >>> cuped_transformation.load_params('cuped_transform_params.json')
    >>> cuped_transformation.transform(
    >>>     ['income', 'age'],
    >>>     inplace=True,
    >>>     name='cuped_transformed'
    >>> )

    Methods
    -------
    get_params_dict()
        Returns dictionary with params if fit() method has been previously
        called.
    load_params_dict(params)
        Load params from a dictionary.
    store_params(store_path)
        Store params to json file if fit() method has been previously called.
    load_params(load_path)
        Load params from a json file.
    fit(covariate_columns)
        Fit model using covariate columns.
    transform(covariate_columns, inplace, name)
        Transform target column after a class instance fitting.
    fit_transform(covariate_columns, inplace, name)
        Combination of fit() and transform() methods.
    """

    THETA_NAME: str = "theta"
    BIAS_NAME: str = "bias"

    def __str__(self) -> str:
        return f"Multi СUPED for {self.target_column}"

    def __init__(self, dataframe: pd.DataFrame, target_column: types.ColumnNamesType, verbose: bool = True) -> None:
        super().__init__(dataframe, target_column, verbose)
        self.theta = None
        self.bias = None
        self.cov: pd.DataFrame = dataframe.cov()

    def store_params(self, store_path: Path) -> None:
        """
        Parameters
        ----------
        store_path : Path
            Path where parameters will be stored in a json format.
        """
        self._check_fitted()
        with open(store_path, "w+") as file:
            params: Dict[str, float] = {}
            for j in range(self.theta.shape[0]):
                params["theta_" + str(j)] = self.theta[j][0]
            params["bias"] = self.bias
            json.dump(params, file)

    def get_params_dict(self) -> Dict:
        """
        Parameters
        ----------
        params : Dict
            Dictionary with params.
        """
        self._check_fitted()
        return {
            Cuped.THETA_NAME: self.theta,
            Cuped.BIAS_NAME: self.bias,
        }

    def load_params_dict(self, params: Dict) -> None:
        """
        Parameters
        ----------
        params : Dict
            Dictionary with params.
        """
        if Cuped.THETA_NAME in params:
            self.theta = params[Cuped.THETA_NAME]
        else:
            raise TypeError(f"params argument must contain: {Cuped.THETA_NAME}")
        if Cuped.BIAS_NAME in params:
            self.bias = params[Cuped.BIAS_NAME]
        else:
            raise TypeError(f"params argument must contain: {Cuped.BIAS_NAME}")
        self.fitted = True

    def load_params(self, load_path: Path) -> None:
        """
        Parameters
        ----------
        load_path : Path
            Path to a json file with parameters.
        """
        with open(load_path, "r+") as file:
            params = json.load(file)
            dimension: int = len(params) - 1
            self.theta: np.ndarray = np.zeros((dimension, 1))
            for j in range(dimension):
                self.theta[j][0] = params["theta_" + str(j)]
            self.bias = params["bias"]
            self.fitted = True

    def fit(self, covariate_columns: types.ColumnNamesType) -> None:
        """
        Fit to calculate Multi CUPED parameters for target column using given
        covariate columns.

        Parameters
        ----------
        covariate_columns : ColumnNamesType
            Columns which will be used as the covariates in Multi CUPED
            transformation.
        """
        self._check_columns(covariate_columns)
        matrix: np.ndarray = self.cov.loc[covariate_columns, covariate_columns]
        amount_features: int = len(covariate_columns)
        covariance_target: np.ndarray = self.cov.loc[covariate_columns, self.target_column].values.reshape(
            amount_features, -1
        )
        self.theta = np.linalg.inv(matrix) @ covariance_target
        self.bias: np.ndarray = (self.df[covariate_columns].values @ self.theta).reshape(-1).mean()
        self.fitted = True

    def __call__(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        y_hat: np.ndarray = y - (X @ self.theta).reshape(-1) + self.bias
        return y_hat

    def transform(
        self, covariate_columns: types.ColumnNamesType, inplace: bool = False, name: Optional[str] = None
    ) -> Union[pd.DataFrame, None]:
        """
        Make Multi CUPED transformation for target column.

        Could be performed inplace or not.

        Parameters
        ----------
        covariate_columns : ColumnNamesType
            Columns which will be used as the covariates.
        inplace : bool, default: ``False``
            If is ``True``, then method returns ``None`` and
            sets a new column for the original dataframe.
            Otherwise return copied dataframe with a new column.
        name : str
            Name for the new transformed target column, if is not defined
            it will be generated automatically.
        """
        self._check_columns(covariate_columns)
        old_variance: float = self.cov.loc[self.target_column, self.target_column]
        self._check_fitted()
        new_target: np.ndarray = self(self.df[self.target_column].values, self.df[covariate_columns].values)
        new_variance: float = np.var(new_target)
        if self.verbose:
            self._verbose(old_variance, new_variance)
        return self._return_result(new_target, inplace, name)

    def fit_transform(
        self,
        covariate_columns: types.ColumnNamesType,
        inplace: bool = False,
        name: Optional[str] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Combination of fit() and transform() methods.

        Parameters
        ----------
        covariate_columns : ColumnNamesType
            Columns which will be used as the covariates.
        inplace : bool, default: ``False``
            If is ``True``, then method returns ``None`` and
            sets a new column for the original dataframe.
            Otherwise return copied dataframe with a new column.
        name : str
            Name for the new transformed target column, if is not defined
            it will be generated automatically.
        """
        self.fit(covariate_columns)
        return self.transform(covariate_columns, inplace, name)
