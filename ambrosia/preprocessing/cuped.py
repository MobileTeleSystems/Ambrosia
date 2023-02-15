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
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ambrosia import types
from ambrosia.tools.ab_abstract_component import AbstractVarianceReducer
from ambrosia.tools.back_tools import wrap_cols


class Cuped(AbstractVarianceReducer):
    """
    Class for data CUPED transformation.

    https://towardsdatascience.com/how-to-double-a-b-testing-speed-with-cuped-f80460825a90
    Y_hat = Y - theta * X
    theta := cov(X, Y) / Var(Y)
    It is important, that the mean covariance metric did not change over time!!!

    Parameters
    ----------
    verbose : bool, default: ``True``
        If ``True`` will print in sys.stdout the information
        about the variance reduction.

    Attributes
    ----------
    params : Dict
        Parameters of instance that will be updated after calling fit() method.
        Include:
        - target column name
        - covariate column name
        - name of column after the transformation
        - linear coefficient for CUPED transformation.
        - bias value for mean equality
    verbose : bool
        Verbose info flag.
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
    >>>     dataframe=dataframe
    >>>     target_column='target'
    >>>     covariate_column='income',
    >>>     transformed_name='cuped_target'
    >>>     inplace=True,
    >>> )

    Now in the dataframe a new column "cuped_target" appeared, we can use it
    to design our experiment and estimate variance reduction. For further CUPED
    usage in the future experiment, let us store the parameters:

    >>> cuped_transformer.store_params('cuped_transform_params.json')

    Now we conduct an experiment and want to transform our data to reduce its
    variation:

    >>> cuped_transformation = Cuped()
    >>> cuped_transformation.load_params('cuped_transform_params.json')
    >>> cuped_transformation.transform(
    >>>     dataframe=exp_results,
    >>>     inplace=True,
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
    non_serializable_params: List = [THETA_NAME, BIAS_NAME]

    def __init__(self, verbose: bool = True) -> None:
        super().__init__(verbose)
        self.params["covariate_column"] = None
        self.params[Cuped.THETA_NAME] = None
        self.params[Cuped.BIAS_NAME] = None

    def __str__(self) -> str:
        return f"СUPED for {self.params['target_column']}"

    def __call__(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        y_hat: np.ndarray = y - self.params[Cuped.THETA_NAME] * (X - self.params[Cuped.BIAS_NAME])
        return y_hat

    def get_params_dict(self) -> Dict:
        """
        Returns a dictionary with params.

        Returns
        -------
        params : Dict
            Dictionary with fitted params.
        """
        self._check_fitted()
        return {
            key: (value if key not in Cuped.non_serializable_params else value.tolist())
            for key, value in self.params.items()
        }

    def load_params_dict(self, params: Dict) -> None:
        """
        Load model parameters from the dictionary.

        Parameters
        ----------
        params : Dict
            Dictionary with params.
        """
        for parameter in self.params:
            if parameter in params:
                if parameter in Cuped.non_serializable_params:
                    self.params[parameter] = np.array(params[parameter])
                else:
                    self.params[parameter] = params[parameter]
            else:
                raise TypeError(f"params argument must contain: {parameter}")
        self.fitted = True

    def fit(
        self,
        dataframe: pd.DataFrame,
        target_column: types.ColumnNameType,
        covariate_column: types.ColumnNameType,
        transformed_name: Optional[types.ColumnNameType] = None,
    ) -> None:
        """
        Fit to calculate CUPED parameters for target column using given
        covariate column and data.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Table with data for the calculation of CUPED parameters.
        target_column : ColumnNameType
            Column from the dataframe, for which CUPED transformation will be
            applied.
        covariate_column : ColumnNameType
            Column which will be used as the covariate in CUPED transformation.
        transformed_name : ColumnNamesType, optional
            Name for the new transformed target column, if is not defined
            it will be generated automatically.
        """
        self._check_cols(dataframe, [target_column, covariate_column])
        covariance: pd.DataFrame = dataframe[[target_column, covariate_column]].cov()
        covariate_variance: float = covariance.loc[covariate_column, covariate_column]

        self.params[Cuped.THETA_NAME] = covariance.loc[target_column, covariate_column] / (
            super().EPSILON + covariate_variance
        )
        self.params[Cuped.BIAS_NAME] = np.mean(dataframe[covariate_column])
        self.params["target_column"] = target_column
        self.params["covariate_column"] = covariate_column
        self.params["transformed_name"] = transformed_name
        self.fitted = True

    def transform(
        self,
        dataframe: pd.DataFrame,
        inplace: bool = False,
    ) -> Union[pd.DataFrame, None]:
        """
        Make CUPED transformation for the target column.

        Could be performed inplace or not.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Table with data for CUPED transformation.
        inplace : bool, default: ``False``
            If is ``True``, then method returns ``None`` and
            sets a new column for the original dataframe.
            Otherwise return copied dataframe with a new column.
        """
        self._check_cols(dataframe, [self.params["target_column"], self.params["covariate_column"]])
        new_target: np.ndarray = self(
            dataframe[self.params["target_column"]], dataframe[self.params["covariate_column"]]
        )
        if self.verbose:
            old_variance: float = np.var(dataframe[self.params["target_column"]])
            new_variance: float = np.var(new_target)
            self._verbose(old_variance, new_variance)
        return self._return_result(dataframe, new_target, inplace)

    def fit_transform(
        self,
        dataframe,
        target_column,
        covariate_column: types.ColumnNameType,
        transformed_name: Optional[types.ColumnNameType] = None,
        inplace: bool = False,
    ) -> Union[pd.DataFrame, None]:
        """
        Combination of fit() and transform() methods.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Table with data for fitting and applying CUPED transformation.
        target_column : ColumnNameType
            Column from the dataframe, for which CUPED transformation will be
            applied.
        covariate_column : ColumnNameType
            Column which will be used as the covariate.
        transformed_name : ColumnNamesType, optional
            Name for the new transformed target column, if is not defined
            it will be generated automatically.
        inplace : bool, default: ``False``
            If is ``True``, then method returns ``None`` and
            sets a new column for the original dataframe.
            Otherwise return copied dataframe with a new column.
        """
        self.fit(dataframe, target_column, covariate_column, transformed_name)
        return self.transform(dataframe, inplace)


class MultiCuped(AbstractVarianceReducer):
    """
    Class for data Multi CUPED transformation.

    Y_hat = Y - X theta (Matrix multiplication)
    theta := argmin Var (Y - X theta)
    It is important, that the mean covariance metric do not change over time!!!


    Parameters
    ----------
    verbose : bool, default: ``True``
        If ``True`` will print in sys.stdout the information
        about the variance reduction.

    Attributes
    ----------
    params : Dict
        Parameters of instance that will be updated after calling fit() method.
        Include:
        - target column name
        - covariate columns names
        - name of column after the transformation
        - linear coefficients for Multi CUPED transformation.
        - bias value for mean equality
    verbose : bool
        Verbose info flag.
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

    >>> cuped_transformer = MultiCuped(verbose=True)
    >>> cuped_transformer.fit_transform(
    >>>     dataframe=dataframe
    >>>     target_column='target'
    >>>     ['income', 'age'],
    >>>     transformed_name='cuped_target'
    >>>     inplace=True,
    >>> )

    Now in the dataframe a new column "cuped_target" appeared, we can use it
    to design our experiment and estimate variance reduction. For further
    Multi CUPED usage in the future experiment, let us store the parameters:

    >>> cuped_transformer.store_params('cuped_transform_params.json')

    Now we conduct an experiment and want to transform our data to reduce its
    variation:

    >>> cuped_transformation = MultiCuped()
    >>> cuped_transformation.load_params('cuped_transform_params.json')
    >>> cuped_transformation.transform(
    >>>     exp_results,
    >>>     inplace=True,
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
        Fit model using covariate columns.
    transform(covariate_column, inplace, name)
        Transform target column after a class instance fitting.
    fit_transform(covariate_column, inplace, name)
        Combination of fit() and transform() methods.
    """

    THETA_NAME: str = "theta"
    BIAS_NAME: str = "bias"
    non_serializable_params: List = [THETA_NAME, BIAS_NAME]

    def __init__(self, verbose: bool = True) -> None:
        super().__init__(verbose)
        self.params["covariate_columns"] = None
        self.params[MultiCuped.THETA_NAME] = None
        self.params[MultiCuped.BIAS_NAME] = None

    def __str__(self) -> str:
        return f"Multi СUPED for {self.params['target_column']}"

    def __call__(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        y_hat: np.ndarray = y - (X @ self.params[MultiCuped.THETA_NAME]).reshape(-1) + self.params[MultiCuped.BIAS_NAME]
        return y_hat

    def get_params_dict(self) -> Dict:
        """
        Returns a dictionary with params.

        Returns
        -------
        params : Dict
            Dictionary with fitted params.
        """
        self._check_fitted()
        return {
            key: (value if key not in MultiCuped.non_serializable_params else value.tolist())
            for key, value in self.params.items()
        }

    def load_params_dict(self, params: Dict) -> None:
        """
        Load model parameters from the dictionary.

        Parameters
        ----------
        params : Dict
            Dictionary with params.
        """
        for parameter in self.params:
            if parameter in params:
                if parameter in MultiCuped.non_serializable_params:
                    self.params[parameter] = np.array(params[parameter])
                else:
                    self.params[parameter] = params[parameter]
            else:
                raise TypeError(f"params argument must contain: {parameter}")
        self.fitted = True

    def fit(
        self,
        dataframe: pd.DataFrame,
        target_column: types.ColumnNameType,
        covariate_columns: types.ColumnNamesType,
        transformed_name: Optional[types.ColumnNameType] = None,
    ) -> None:
        """
        Fit to calculate Multi CUPED parameters for target column using selected
        covariate columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Table with data for the calculation of CUPED parameters.
        target_column : ColumnNameType
            Column from the dataframe, for which CUPED transformation will be
            applied.
        covariate_columns : ColumnNamesType
            Columns which will be used as the covariates in Multi CUPED
            transformation.
        transformed_name : ColumnNamesType, optional
            Name for the new transformed target column, if is not defined
            it will be generated automatically.
        """
        covariate_columns = wrap_cols(covariate_columns)
        cols_concat: List = [target_column] + covariate_columns
        self._check_cols(dataframe, cols_concat)
        covariance: np.ndarray = dataframe[cols_concat].cov()
        matrix: np.ndarray = covariance.loc[covariate_columns, covariate_columns]
        num_features: int = len(covariate_columns)
        covariance_target: np.ndarray = covariance.loc[covariate_columns, target_column].values.reshape(
            num_features, -1
        )

        self.params[MultiCuped.THETA_NAME] = np.linalg.inv(matrix) @ covariance_target
        self.params[MultiCuped.BIAS_NAME]: np.ndarray = (
            (dataframe[covariate_columns].values @ self.params[MultiCuped.THETA_NAME]).reshape(-1).mean()
        )
        self.params["target_column"] = target_column
        self.params["covariate_columns"] = covariate_columns
        self.params["transformed_name"] = transformed_name
        self.fitted = True

    def transform(
        self,
        dataframe: pd.DataFrame,
        inplace: bool = False,
    ) -> Union[pd.DataFrame, None]:
        """
        Make Multi CUPED transformation for the target column.

        Could be performed inplace or not.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Table with data for Multi CUPED transformation.
        inplace : bool, default: ``False``
            If is ``True``, then method returns ``None`` and
            sets a new column for the original dataframe.
            Otherwise return copied dataframe with a new column.
        """
        self._check_cols(dataframe, [self.params["target_column"]] + self.params["covariate_columns"])
        self._check_fitted()
        new_target: np.ndarray = self(
            dataframe[self.params["target_column"]].values, dataframe[self.params["covariate_columns"]].values
        )
        if self.verbose:
            old_variance: float = np.var(dataframe[self.params["target_column"]])
            new_variance: float = np.var(new_target)
            self._verbose(old_variance, new_variance)
        return self._return_result(dataframe, new_target, inplace)

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
        target_column: types.ColumnNameType,
        covariate_columns: types.ColumnNamesType,
        transformed_name: Optional[types.ColumnNameType] = None,
        inplace: bool = False,
    ) -> Union[pd.DataFrame, None]:
        """
        Combination of fit() and transform() methods.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Table with data for fitting and applying Multi CUPED transformation.
        target_column : ColumnNameType
            Column from the dataframe, for which CUPED transformation will be
            applied.
        covariate_column : ColumnNameType
            Column which will be used as the covariate.
        transformed_name : ColumnNamesType, optional
            Name for the new transformed target column, if is not defined
            it will be generated automatically.
        inplace : bool, default: ``False``
            If is ``True``, then method returns ``None`` and
            sets a new column for the original dataframe.
            Otherwise return copied dataframe with a new column.
        """
        self.fit(dataframe, target_column, covariate_columns, transformed_name)
        return self.transform(dataframe, inplace)
