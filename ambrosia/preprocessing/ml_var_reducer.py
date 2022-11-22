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
Module contains ML-based data transformation methods for the experiment
acceleration.
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from ambrosia import types
from ambrosia.tools import log
from ambrosia.tools.ab_abstract_component import AbstractVarianceReduction


class MLVarianceReducer(AbstractVarianceReduction):
    """
    Machine Learning approach for variance reduction.

    Building a model M, we can make a transformation:
    Y_hat = Y - M(X) + MEAN(M(X))

    It is important, that that the mean of M(X) do not change over time!!!
    You can choose models from Gradient boosting or Ridge regression or your
    own model class, for example ``sklearn.ensemble.RandomForest``, and pass
    models params to constructor function for a model assembly.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Table with data for transformation.
    target_column : ColumnNameType
        Column from the dataframe, for which transformation will be
        applied.
    model : str or model type, default: ``"boosting"``
        Model which will be used for the transformations.
    model_params : Dict, optional
        Dictionary with parameters which will be used in constructor
        for a model assembly.
    scores : Dict[str, Callable], optional
        Scores which will be used.
    verbose : bool, default: ``True``
        If ``True`` will print in sys.stdout the information
        about the reduction in variance.

    Attributes
    ----------
    dataframe : pd.DataFrame
        Table with data for transformation.
    target_column : ColumnNameType
        Column from the dataframe, for which transformation will be
        applied.
    model : model type
        Model which will be used for the transformations.
    bias : float
        Additional bias equals mean(M(X)).
    scores : Dict[str, Callable]
        Scores which will be used.
    verbose : bool
        Verbose info flag.

    Examples
    --------
    We have data table with column 'target' and columns 'feature_1',
    'feature_2', 'feature_3'. Let us assume, that means of all these metrics
    don't change over the time, it can be age for example. We want to reduce
    variance using the predictions some of ML model, then we can use this class:

    >>> transformer = MLVarianceReducer(dataframe, 'target') # By default CatBoost model will be choosen
    >>> transformer.fit_transform([feature columns], inplace=True, name='new_target')
    >>> transformer.store_params('path_ml_params.json')

    Now to transform the experimental data we use the following commands:

    >>> transformer = MLVarianceReducer(exp_data, 'target')
    >>> transformer.load_params('path_ml_params.json')
    >>> transformer.transform([feature columns], inplace=True, name='new_target')

    Methods
    -------
    store_params(store_path)
        Store params to json file if fit() method has been previously called.
    load_params(load_path)
        Load params from a json file.
    fit(covariate_columns)
        Fit model using a specific covariate columns.
    transform(covariate_columns, inplace, name)
        Transform target column after a model fitting.
    fit_transform(covariate_column, inplace, name)
        Combination of fit() and transform() methods.
    """

    def __set_scorer(self, scores: Optional[Dict[str, Callable]]):
        """
        Support method for scorer setting.
        """
        if scores is not None:
            self.score = scores
        else:
            self.score = {"MSE": mean_squared_error}

    def store_params(self, store_path: Path) -> None:
        """
        Store params of model as a json file, available only for CatBoost
        model.

        You can reach model using instance.model and store it by yourself.

        Parameters
        ----------
         store_path : Path
            Path where models parameters will be stored in a json format.
        """
        self.fitted = True
        if isinstance(self.model, CatBoostRegressor):
            self.model.save_model(store_path, format="json")
            with open(store_path, "r+") as file:
                data = json.load(file)
                data.update({"train_bias": self.bias})
            with open(store_path, "w+") as file:
                json.dump(data, file)
        else:
            raise ValueError("Model cant be stored to json file, you can reach model by instance.model")

    def load_params(self, load_path: Path) -> None:
        """
        Load models params from a json file, works only for CatBoost model.

        Parameters
        ----------
        load_path: Path
            Path to a json file with model parameters.
        """
        self.fitted = True
        if not isinstance(self.model, CatBoostRegressor):
            raise ValueError("Model cant be load from the file, set it via instance.model = ...")
        self.model.load_model(load_path, format="json")
        with open(load_path, "r+") as file:
            data = json.load(file)
            self.bias = data["train_bias"]

    def __create_model(self) -> None:
        if not isinstance(self.model, str):
            self.model = self.model(**self.model_params)
        if self.model == "linear":
            self.model = Ridge(**self.model_params)
        if self.model == "boosting":
            if "verbose" not in self.model_params:
                self.model_params["verbose"] = False
            self.model = CatBoostRegressor(**self.model_params)

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: types.ColumnNameType,
        model: Union[str, Any] = "boosting",
        model_params: Optional[Dict] = None,
        scores: Optional[Dict[str, Callable]] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(dataframe, target_column, verbose)
        self.model = model
        self.bias = None
        if model_params is None:
            self.model_params = {}
        else:
            self.model_params = model_params
        self.__create_model()
        self.__set_scorer(scores)

    def __str__(self) -> str:
        return f"ML approach reduce for {self.target_column}"

    def __call__(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Transform target values using its predictions based on covariates.

        Class must be fitted.
        """
        self._check_fitted()
        y_hat = y - self.model.predict(X) + self.bias
        return y_hat

    def _verbose_score(self, prediction: np.ndarray) -> None:
        for name, scorer in self.score.items():
            current_score: float = scorer(self.df[self.target_column], prediction)
            log.info_log(f"Prediction {name} score - {current_score:.5f}")

    def fit(self, covariate_columns: types.ColumnNamesType) -> None:
        """
        Fit model for transformations.

        Parameters
        ----------
        covariate_columns: ColumnNamesType
            Columns which will be used for the transformation.
        """
        self._check_columns(covariate_columns)
        self.model.fit(self.df[covariate_columns].values, self.df[self.target_column].values)
        self.bias = np.mean(self.model.predict(self.df[covariate_columns].values))
        self.fitted = True

    def transform(
        self,
        covariate_columns: types.ColumnNamesType,
        inplace: bool = False,
        name: Optional[str] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Transform data using the fitted model.

        Parameters
        ----------
        covariate_columns : ColumnNamesType
            Columns which will be used for the transformations.
        inplace : bool, default: ``False``
            If is ``True``, then method returns ``None`` and
            sets a new column for the original dataframe.
            Otherwise return copied dataframe with a new column.
        name : str, optional
            Name for the new transformed target column, if is not defined
            it will be generated automatically.
        """
        self._check_columns(covariate_columns)
        self._check_fitted()
        prediction: np.ndarray = self(self.df[self.target_column].values, self.df[covariate_columns].values)
        new_target: np.ndarray = prediction + np.mean(self.df[self.target_column]) - np.mean(prediction)
        old_variance: float = np.var(self.df[self.target_column].values)
        new_variance: float = np.var(prediction)
        if self.verbose:
            self._verbose(old_variance, new_variance)
            self._verbose_score(prediction)
        return self._return_result(new_target, inplace, name)

    def fit_transform(
        self, covariate_columns: types.ColumnNamesType, inplace: bool = False, name: Optional[str] = None
    ) -> Union[pd.DataFrame, None]:
        """
        Combinate consequentially ``fit()`` and ``transform()`` methods.

        Parameters
        ----------
        covariate_columns : ColumnNamesType
            Columns which will be used for the transformations.
        inplace : bool, default: ``False``
            If is ``True``, then method returns ``None`` and
            sets a new column for the original dataframe.
            Otherwise return copied dataframe with a new column.
        name : str, optional
            Name for the new transformed target column, if is not defined
            it will be generated automatically.
        """
        self.fit(covariate_columns)
        return self.transform(covariate_columns, inplace, name)
