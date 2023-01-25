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
Module contains tools for outliers removal from data during a
preprocessing task.
"""
from typing import Dict, List, Union, Iterable

import numpy as np
import pandas as pd
import scipy.stats as sps

from ambrosia import types
from ambrosia.tools import log
from ambrosia.tools.ab_abstract_component import AbstractFittableTransformer


class OldRobustPreprocessor:
    """
    Unit for simple robust transformation for avoiding outliers in data.

    It cuts the alpha percentage of distribution from head and tail for
    each given metric.
    The data distribution structure assumed to present as small alpha
    part of outliers, followed by the normal part of the data with another
    alpha part of outliers at the end of the distribution.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe to transform.
    verbose : bool, default: ``True``
        If ``True`` will show info about the transformation of passed columns.

    Attributes
    ----------
    dataframe : pd.DataFrame
        Dataframe to transform.
    verbose : bool
        Verbose info flag.

    Examples
    --------
    >>> robust = OldRobustPreprocessor(dataframe, verbose=True)
    >>> robust.run('my_column_with_outliers', alpha=0.05, inplace=True)
    or
    >>> robust.run(['column1', 'column2'], alpha=0.001, inplace=True)
    You can pass one or number of columns, if several columns passed
    it will drop 2 * alpha percent of extreme values for each column.
    """

    def __str__(self) -> str:
        return "Robust transformation"

    @staticmethod
    def __calculate_stats(values: np.ndarray) -> Dict[str, float]:
        return {
            "Mean": np.mean(values),
            "Variance": np.var(values),
            "IQR": np.quantile(values, 0.75) - np.quantile(values, 0.25),
            "Range": np.max(values) - np.min(values),
        }

    def __init__(self, dataframe: pd.DataFrame, verbose: bool = True) -> None:
        """ """
        self.dataframe = dataframe
        self.verbose = verbose

    @staticmethod
    def __verbose(was_stats: Dict[str, float], new_stats: Dict[str, float], name: str) -> None:
        """
        Verbose transormations to os.stdout.
        """
        for metric in was_stats.keys():
            was: float = was_stats[metric]
            new: float = new_stats[metric]
            log.info_log(f"Change {metric} {name}: {was:.4f} ===> {new:.4f}")

    @staticmethod
    def __verbose_list(
        was_stats: List[Dict[str, float]], new_stats: List[Dict[str, float]], names: types.ColumnNamesType
    ) -> None:
        """
        Verbose iteratively.
        """
        for name, stat_1, stat_2 in zip(names, was_stats, new_stats):
            log.info_log("\n")
            OldRobustPreprocessor.__verbose(stat_1, stat_2, name)

    @staticmethod
    def __get_stats(df: pd.DataFrame, names: types.ColumnNamesType) -> List[Dict[str, float]]:
        """
        Get metrics for all columns.
        """
        result: List[Dict[str, float]] = []
        for name in names:
            err_msg: str = f"Column name is not in data frame, coumn - {name}"
            assert name in df.columns, err_msg
            result.append(OldRobustPreprocessor.__calculate_stats(df[name].values))
        return result

    def run(
        self, column_names: types.ColumnNamesType, alpha: float = 0.05, inplace: bool = False
    ) -> Union[pd.DataFrame, None]:
        """
        Remove objects from the dataframe which are in the head and tail alpha
        parts of chosen metrics distributions.

        Parameters
        ----------
        column_names : ColumnNamesType
            One or number of columns in the dataframe.
        alpha : float, default: ``0.05``
            The percentage of removed data from head and tail.
        inplace : bool, default: ``False``
            If ``True`` transforms the given dataframe, otherwise copy and
            returns an another one.

        Returns
        -------
        df : Union[pd.DataFrame, None]
            None or transformed dataframe
        """
        if isinstance(column_names, types.ColumnNameType):
            column_names = [column_names]
        if alpha < 0 or alpha >= 0.5:
            raise ValueError(f"Alpha must be from 0 to 0.5, but alpha = {alpha}")

        was_stats: List[Dict[str, float]] = OldRobustPreprocessor.__get_stats(self.dataframe, column_names)

        if not inplace:
            transformed = self.dataframe.copy()

        for column in column_names:
            cur_df: pd.DataFrame = self.dataframe if inplace else transformed
            tail, head = np.quantile(cur_df[column].values, [alpha, 1 - alpha])
            bad_table = cur_df[(cur_df[column] < tail) | (cur_df[column] > head)]
            bad_ids: np.ndarray = bad_table.index.values
            cur_df.drop(bad_ids, inplace=True)

        if self.verbose:
            log.info_log(f"Make robust transformation with alpha = {alpha:.3f}")
            df = self.dataframe if inplace else transformed
            new_stats: Dict[str, float] = OldRobustPreprocessor.__get_stats(df, column_names)
            OldRobustPreprocessor.__verbose_list(was_stats, new_stats, column_names)
        return None if inplace else transformed


class RobustPreprocessor(AbstractFittableTransformer):
    available_tails = ["both", "left", "right"]

    def __str__(self) -> str:
        return "Robust preprocessing"

    @staticmethod
    def __calculate_stats(values: np.ndarray) -> Dict[str, float]:
        return {
            "Mean": np.mean(values),
            "Variance": np.var(values),
            "IQR": np.quantile(values, 0.75) - np.quantile(values, 0.25),
            "Range": np.max(values) - np.min(values),
        }

    def __init__(self, tail: str = "both", verbose: bool = True) -> None:
        """ """
        if tail not in self.available_tails:
            raise ValueError(f"tail must be one of {self.available_tails}")
        self.params = {
            "tail": tail,
            "column_names": None,
            "alpha": None,
            "quantiles": None,
            "verbose": verbose,
        }
        super().__init__()

    def get_params_dict(self) -> Dict:
        """
        Returns a dictionary with params.
        """
        self._check_fitted()
        return self.params

    def load_params_dict(self, params: Dict) -> None:
        """
        Parameters
        ----------
        params : Dict
            Dictionary with params.
        """
        for parameter in self.params:
            if parameter in params:
                self.params[parameter] = params[parameter]
            else:
                raise TypeError(f"params argument must contain: {parameter}")
        self.fitted = True

    @staticmethod
    def __verbose(prev_stats: Dict[str, float], new_stats: Dict[str, float], name: str) -> None:
        """
        Verbose transormations to os.stdout.
        """
        for metric in prev_stats.keys():
            prev: float = prev_stats[metric]
            new: float = new_stats[metric]
            log.info_log(f"Change {metric} {name}: {prev:.4f} ===> {new:.4f}")

    @staticmethod
    def __verbose_list(
        prev_stats: List[Dict[str, float]], new_stats: List[Dict[str, float]], names: types.ColumnNamesType
    ) -> None:
        """
        Verbose iteratively.
        """
        for name, stat_1, stat_2 in zip(names, prev_stats, new_stats):
            log.info_log("\n")
            RobustPreprocessor.__verbose(stat_1, stat_2, name)

    @staticmethod
    def __get_stats(df: pd.DataFrame, names: types.ColumnNamesType) -> List[Dict[str, float]]:
        """
        Get metrics for all columns.
        """
        result: List[Dict[str, float]] = []
        for name in names:
            err_msg: str = f"Column name is not in data frame, coumn - {name}"
            assert name in df.columns, err_msg
            result.append(RobustPreprocessor.__calculate_stats(df[name].values))
        return result

    def __wrap_alpha(self, alpha: Union[float, Iterable]) -> np.ndarray:
        columns_num = len(self.params["column_names"])
        if isinstance(alpha, float):
            alpha = np.array([alpha] * columns_num)
        elif isinstance(alpha, Iterable):
            alpha = np.array(alpha)
        else:
            raise ValueError("Alpha parameter must be float or an iterable")
        if len(alpha) != columns_num:
            raise ValueError("Alpha length must be equal to the columns number")
        if (alpha < 0).any() or (alpha >= 0.5).any():
            raise ValueError(f"Alpha value must be from 0 to 0.5, but alpha vector = {alpha}")
        return alpha

    def fit(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
        alpha: Union[float, np.ndarray] = 0.05,
    ) -> Union[pd.DataFrame, None]:
        """
        Fit.
        """
        self.params["column_names"] = self._wrap_cols(column_names)
        self._check_columns(dataframe, self.params["column_names"])
        columns_num = len(self.params["column_names"])
        self.params["alpha"] = self.__wrap_alpha(alpha)
        if self.params["tail"] == "both":
            self.params["quantiles"] = np.zeros((columns_num, 2))
            for num, col in enumerate(self.params["column_names"]):
                alpha = self.params["alpha"][num] / 2
                self.params["quantiles"][num, :] = np.quantile(dataframe[col].values, [alpha, 1 - alpha])
        else:
            self.params["quantiles"] = np.zeros((columns_num, 1))
            for num, col in enumerate(self.params["column_names"]):
                alpha = self.params["alpha"][num] if self.params["tail"] == "left" else 1 - self.params["alpha"][num]
                self.params["quantiles"][num] = np.quantile(dataframe[col].values, alpha)
        self.fitted = True

    def transform(self, dataframe: pd.DataFrame, inplace: bool = False):
        """
        Transform.
        """
        self._check_fitted()
        self._check_columns(dataframe, self.params["column_names"])
        if self.params["verbose"]:
            prev_stats: List[Dict[str, float]] = RobustPreprocessor.__get_stats(dataframe, self.params["column_names"])

        transformed: pd.DataFrame = dataframe if inplace else dataframe.copy()
        if self.params["tail"] == "both":
            mask = (transformed[self.params["column_names"]] <= self.params["quantiles"][:, 0]).any(axis=1) | (
                transformed[self.params["column_names"]] >= self.params["quantiles"][:, 1]
            ).any(axis=1)
        elif self.params["tail"] == "left":
            mask = (transformed[self.params["column_names"]] <= self.params["quantiles"].T).any(axis=1)
        elif self.params["tail"] == "right":
            mask = (transformed[self.params["column_names"]] >= self.params["quantiles"].T).any(axis=1)
        bad_ids = transformed.loc[mask].index
        transformed.drop(bad_ids, inplace=True)

        if self.params["verbose"]:
            log.info_log(f"Make robust transformation of columns with alphas = {np.round(self.params['alpha'], 3)}")
            new_stats: Dict[str, float] = RobustPreprocessor.__get_stats(transformed, self.params["column_names"])
            RobustPreprocessor.__verbose_list(prev_stats, new_stats, self.params["column_names"])
        return None if inplace else transformed

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
        alpha: Union[float, np.ndarray] = 0.05,
        inplace: bool = False,
    ):
        """
        Fit-Transform.
        """
        self.fit(dataframe, column_names, alpha)
        return self.transform(dataframe, inplace)


class IQRPreprocessor(AbstractFittableTransformer):
    def __str__(self) -> str:
        return "IQR outliers preprocessing"

    def __init__(
        self,
    ) -> None:
        self.params = {"column_names": None, "medians": None, "quartiles": None}
        super().__init__()

    def get_params_dict(self) -> Dict:
        """
        Returns a dictionary with params.
        """
        self._check_fitted()
        return self.params

    def load_params_dict(self, params: Dict) -> None:
        """
        Parameters
        ----------
        params : Dict
            Dictionary with params.
        """
        for parameter in self.params:
            if parameter in params:
                self.params[parameter] = params[parameter]
            else:
                raise TypeError(f"params argument must contain: {parameter}")
        self.fitted = True

    def fit(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
    ) -> None:
        """
        Fit.
        """
        self.params["column_names"] = self._wrap_cols(column_names)
        self._check_columns(dataframe, self.params["column_names"])
        X = dataframe[column_names].values
        self.params["quartiles"] = np.quantile(X, (0.25, 0.75), axis=0).T
        self.params["medians"] = np.median(X, axis=0).T
        self.fitted = True

    def transform(self, dataframe: pd.DataFrame, inplace: bool = False):
        """
        Transform.
        """
        self._check_fitted()
        self._check_columns(dataframe, self.params["column_names"])
        transformed: pd.DataFrame = dataframe if inplace else dataframe.copy()
        iqr = self.params["quartiles"][:, 1] - self.params["quartiles"][:, 0]
        tail = self.params["quartiles"][:, 0] - 1.5 * iqr
        head = self.params["quartiles"][:, 1] + 1.5 * iqr
        mask = (
            (transformed[self.params["column_names"]] < tail) | (transformed[self.params["column_names"]] > head)
        ).any(axis=1)
        bad_ids = transformed.loc[mask].index
        transformed.drop(bad_ids, inplace=True)
        return None if inplace else transformed

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
        inplace: bool = False,
    ):
        """
        Fit-Transform.
        """
        self.fit(dataframe, column_names)
        return self.transform(dataframe, inplace)


class BoxCoxTransformer(AbstractFittableTransformer):
    """
    Unit for Box-Cox transformation of pandas data.
    """

    def __str__(self) -> str:
        return "Box-Cox transformation"

    def __init__(
        self,
    ) -> None:
        self.column_names = None
        self.lambda_ = None
        super().__init__()

    def get_params_dict(self) -> Dict:
        """
        Returns a dictionary with params.
        """
        self._check_fitted()
        return {
            "column_names": self.column_names,
            "lambda_": self.lambda_,
        }

    def load_params_dict(self, params: Dict) -> None:
        """
        Load model parameters from the dictionary.

        Parameters
        ----------
        params : Dict
            Dictionary with params.
        """
        if "column_names" in params:
            self.column_names = params["column_names"]
        else:
            raise TypeError(f"params argument must contain: {'column_names'}")
        if "lambda_" in params:
            self.lambda_ = params["lambda_"]
        else:
            raise TypeError(f"params argument must contain: {'lambda_'}")
        self.fitted = True

    def fit(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
    ) -> None:
        """
        Fit.
        """
        self.column_names = self._wrap_cols(column_names)
        self._check_columns(dataframe, self.column_names)
        columns_num: int = len(column_names)
        self.lambda_ = np.zeros(columns_num)
        X = dataframe[column_names].values
        for num in range(columns_num):
            self.lambda_[num] = sps.boxcox(X[:, num])[1]
        self.fitted = True

    def transform(self, dataframe: pd.DataFrame, inplace: bool = False):
        """
        Transform.
        """
        self._check_fitted()
        self._check_columns(dataframe, self.column_names)
        transformed: pd.DataFrame = dataframe if inplace else dataframe.copy()
        X = transformed[self.column_names].values
        for num in range(len(self.column_names)):
            if self.lambda_[num] == 0:
                X[:, num] = np.log(X[:, num])
            else:
                X[:, num] = (X[:, num] ** self.lambda_[num] - 1) / self.lambda_[num]
        transformed[self.column_names] = X
        return None if inplace else transformed

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
        inplace: bool = False,
    ):
        """
        Fit-Transform.
        """
        self.fit(dataframe, column_names)
        return self.transform(dataframe, inplace)

    def inverse_transform(self, dataframe: pd.DataFrame, inplace: bool = False):
        """
        Inverse transform.
        """
        self._check_fitted()
        self._check_columns(dataframe, self.column_names)
        transformed: pd.DataFrame = dataframe if inplace else dataframe.copy()
        X_tr = transformed[self.column_names].values
        for num in range(len(self.column_names)):
            if self.lambda_[num] == 0:
                X_tr[:, num] = np.exp(X_tr[:, num])
            else:
                X_tr[:, num] = (X_tr[:, num] * self.lambda_[num] + 1) ** (1 / self.lambda_[num])
        transformed[self.column_names] = X_tr
        return None if inplace else transformed


class LogTransformer(AbstractFittableTransformer):
    """
    Unit for logarithmic transformation of the data.
    """

    def __str__(self) -> str:
        return "Logarithmic transformation"

    def __init__(self):
        self.column_names = None
        super().__init__()

    def get_params_dict(self) -> Dict:
        """
        Returns a dictionary with params.
        """
        self._check_fitted()
        return {
            "column_names": self.column_names,
        }

    def load_params_dict(self, params: Dict) -> None:
        """
        Load model parameters from the dictionary.

        Parameters
        ----------
        params : Dict
            Dictionary with params.
        """
        if "column_names" in params:
            self.column_names = params["column_names"]
        else:
            raise TypeError(f"params argument must contain: {'column_names'}")
        self.fitted = True

    def fit(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
    ) -> None:
        """
        Fit.
        """
        self.column_names = self._wrap_cols(column_names)  # ATTENTION HERE
        self._check_columns(dataframe, self.column_names)
        self.fitted = True

    def transform(self, dataframe: pd.DataFrame, inplace: bool = False):
        """
        Transform.
        """
        self._check_fitted()
        self._check_columns(dataframe, self.column_names)
        transformed: pd.DataFrame = dataframe if inplace else dataframe.copy()
        if (transformed[self.column_names] > 0).all(axis=None):
            transformed[self.column_names] = np.log(transformed[self.column_names].values)
        else:
            raise ValueError(f"All values in columns {self.column_names} must be positive")
        return None if inplace else transformed

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
        inplace: bool = False,
    ):
        """
        Fit-Transform.
        """
        self.fit(dataframe, column_names)
        return self.transform(dataframe, inplace)

    def inverse_transform(self, dataframe: pd.DataFrame, inplace: bool = False):
        """
        Inverse transform.
        """
        self._check_fitted()
        self._check_columns(dataframe, self.column_names)
        transformed: pd.DataFrame = dataframe if inplace else dataframe.copy()
        transformed[self.column_names] = np.exp(transformed[self.column_names].values)
        return None if inplace else transformed
