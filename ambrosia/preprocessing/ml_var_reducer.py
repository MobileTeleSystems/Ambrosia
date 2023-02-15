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

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from ambrosia import types
from ambrosia.tools import log
from ambrosia.tools.ab_abstract_component import AbstractVarianceReducer
from ambrosia.tools.back_tools import wrap_cols


class MLVarianceReducer(AbstractVarianceReducer):
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
    model : model type
        Model which will be used for the transformations.
    params : Dict
        Parameters of instance that will be updated after calling fit() method.
        Include:
        - target column name
        - covariate columns names
        - name of column after the transformation
        - additional train bias equals mean(M(X)).
    scores : Dict[str, Callable]
        Scores which will be used.
    verbose : bool
        Verbose info flag.
    fitted : bool
        Fit status flag.

    Examples
    --------
    We have data table with column 'target' and columns 'feature_1',
    'feature_2', 'feature_3'. Let us assume, that means of all these metrics
    don't change over the time, it can be age for example. We want to reduce
    variance using the predictions some of ML model, then we can use this class:

    >>> transformer = MLVarianceReducer() # By default CatBoost model will be choosen
    >>> transformer.fit_transform(dataframe, 'target', [feature columns], inplace=True, name='new_target')
    >>> transformer.store_params('path_ml_params.json')

    Now to transform the experimental data we use the following commands:

    >>> transformer = MLVarianceReducer()
    >>> transformer.load_params('path_ml_params.json')
    >>> transformer.transform(exp_data, inplace=True)

    Methods
    -------
    get_params_dict()
        Returns dict with instance fitted parameters.
    load_params_dict()
        Load parameters from the dict.
    store_params(store_path)
        Store fitted params in a json file and pickle model file.
    load_params(load_path)
        Load params from a json file and pickled model.
    fit(**fit_params)
        Fit model using a train data.
    transform(dataframe, inplace)
        Transform target column of a data frame.
    fit_transform(dataframe, **fit_params, inplace)
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

    def __create_model(self) -> None:
        """
        Construct variance reducing ML model.
        """
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
        model: Union[str, Any] = "boosting",
        model_params: Optional[Dict] = None,
        scores: Optional[Dict[str, Callable]] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(verbose)
        self.params["covariate_columns"] = None
        self.params["train_bias"] = None
        self.model = model
        self.model_params = {} if model_params is None else model_params
        self.__set_scorer(scores)

    def __str__(self) -> str:
        return f"ML approach reduce for {self.params['target_column']}"

    def __call__(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Transform target values using its predictions based on covariates.

        Class must be fitted.
        """
        self._check_fitted()
        y_hat = y - self.model.predict(X) + self.params["train_bias"]
        return y_hat

    def _verbose_score(self, dataframe: pd.DataFrame, prediction: np.ndarray) -> None:
        for name, scorer in self.score.items():
            current_score: float = scorer(dataframe[self.params["target_column"]], prediction)
            log.info_log(f"Prediction {name} score - {current_score:.5f}")

    def _check_load_params(self, params: Dict) -> None:
        for parameter in self.params:
            if parameter in params:
                self.params[parameter] = params[parameter]
            else:
                raise TypeError(f"params argument must contain: {parameter}")

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
            "target_column": self.params["target_column"],
            "covariate_columns": self.params["covariate_columns"],
            "transformed_name": self.params["transformed_name"],
            "train_bias": self.params["train_bias"],
            "model": self.model,
        }

    def load_params_dict(self, params: Dict) -> None:
        """
        Load instance parameters from the dictionary.

        Parameters
        ----------
        params : Dict
            Dictionary with params.
        """
        self._check_load_params(params)
        if "model" in params:
            self.model = params["model"]
        else:
            raise TypeError(f"params argument must contain: {'model'}")
        self.fitted = True

    def store_params(self, config_store_path: Path, model_store_path: Path) -> None:
        """
        Store params of model as a json file, available only for CatBoost
        model.

        You can reach model using instance.model and store it by yourself.

        Parameters
        ----------
         store_path : Path
            Path where models parameters will be stored in a json format.
        """
        self._check_fitted()
        with open(config_store_path, "w+") as file:
            json.dump(self.params, file)
        joblib.dump(self.model, model_store_path)

    def load_params(self, config_load_path: Path, model_load_path: Path) -> None:
        """
        Load models params from a json file, works only for CatBoost model.

        Parameters
        ----------
        load_path: Path
            Path to a json file with model parameters.
        """
        with open(config_load_path, "r+") as file:
            params = json.load(file)
            self._check_load_params(params)
        self.model = joblib.load(model_load_path)
        self.fitted = True

    def fit(
        self,
        dataframe: pd.DataFrame,
        target_column: types.ColumnNameType,
        covariate_columns: types.ColumnNamesType,
        transformed_name: Optional[types.ColumnNamesType] = None,
    ) -> None:
        """
        Fit model for transformations.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Table with data for model fitting.
        target_column : ColumnNameType
            Column from the dataframe, for which transformation will be
            applied.
        covariate_columns: ColumnNamesType
            Columns which will be used for the transformation.
        transformed_name : ColumnNamesType, optional
            Name for the new transformed target column, if is not defined
            it will be generated automatically.
        """
        covariate_columns = wrap_cols(covariate_columns)
        self._check_cols(dataframe, [target_column] + covariate_columns)
        self.__create_model()
        self.model.fit(dataframe[covariate_columns].values, dataframe[target_column].values)

        self.params["target_column"] = target_column
        self.params["transformed_name"] = transformed_name
        self.params["covariate_columns"] = covariate_columns
        self.params["train_bias"] = np.mean(self.model.predict(dataframe[covariate_columns].values))
        self.fitted = True

    def transform(
        self,
        dataframe: pd.DataFrame,
        inplace: bool = False,
    ) -> Union[pd.DataFrame, None]:
        """
        Transform data using the fitted model.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Table with data for transformation.
        inplace : bool, default: ``False``
            If is ``True``, then method returns ``None`` and
            sets a new column for the original dataframe.
            Otherwise return copied dataframe with a new column.
        """
        self._check_cols(dataframe, [self.params["target_column"]] + self.params["covariate_columns"])
        self._check_fitted()
        prediction: np.ndarray = self(
            dataframe[self.params["target_column"]].values, dataframe[self.params["covariate_columns"]].values
        )
        new_target: np.ndarray = prediction + np.mean(dataframe[self.params["target_column"]]) - np.mean(prediction)
        if self.verbose:
            old_variance: float = np.var(dataframe[self.params["target_column"]].values)
            new_variance: float = np.var(prediction)
            self._verbose(old_variance, new_variance)
            self._verbose_score(dataframe, prediction)
        return self._return_result(dataframe, new_target, inplace)

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
        target_column: types.ColumnNameType,
        covariate_columns: types.ColumnNamesType,
        transformed_name: Optional[types.ColumnNamesType] = None,
        inplace: bool = False,
    ) -> Union[pd.DataFrame, None]:
        """
        Combinate consequentially ``fit()`` and ``transform()`` methods.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Table with data for model fitting and further transformation.
        target_column : ColumnNameType
            Column from the dataframe, for which transformation will be
            applied.
        covariate_columns: ColumnNamesType
            Columns which will be used for the transformation.
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
