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
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd

from ambrosia import types
from ambrosia.tools import log
from ambrosia.tools.ab_abstract_component import AbstractFittableTransformer
from ambrosia.tools.back_tools import wrap_cols


class RobustPreprocessor(AbstractFittableTransformer):
    """
    Unit for simple robust transformation for avoiding outliers in data.

    It cuts the alpha percentage of distribution from head, tail or both sides
    for each given metric.
    The data distribution structure assumed to present as small alpha
    part of outliers, followed by the normal part of the data with another
    alpha part of outliers at the end of the distribution.

    Parameters
    ----------
    verbose : bool, default: ``True``
        If ``True`` will show info about the transformation of passed columns.

    Attributes
    ----------
    params : Dict
        Dictionary with operational parameters of the instance.
        Updated after calling the ``fit`` method.
    verbose : bool
        Verbose info flag.
    available_tails : List
        List of the available tail type names to preprocess
    non_serializable_params: List
        List of the class parameters that should be converted to lists
        in order to serialize.
    fitted : bool
        Fit flag.

    Examples
    --------
    >>> robust = RobustPreprocessor(verbose=True)
    >>> robust.fit(dataframe, ['column1', 'column2'], alpha=0.05)
    >>> robust.transform(dataframe, inplace=True)

    You can pass one or number of columns, if several columns are passed
    it will drop in total alpha percent of extreme values for each column.
    """

    available_tails: List = ["both", "left", "right"]
    non_serializable_params: List = ["alpha", "quantiles"]

    def __str__(self) -> str:
        return "Robust preprocessing"

    def __init__(self, verbose: bool = True) -> None:
        """
        RobustPreprocessor class constructor.
        """
        self.params = {
            "tail": None,
            "column_names": None,
            "alpha": None,
            "quantiles": None,
        }
        self.verbose = verbose
        super().__init__()

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
            key: (value if key not in RobustPreprocessor.non_serializable_params else value.tolist())
            for key, value in self.params.items()
        }

    def load_params_dict(self, params: Dict) -> None:
        """
        Load prefitted parameters form a dictionary.

        Parameters
        ----------
        params : Dict
            Dictionary with prefitted params.
        """
        for parameter in self.params:
            if parameter in params:
                if parameter in RobustPreprocessor.non_serializable_params:
                    self.params[parameter] = np.array(params[parameter])
                else:
                    self.params[parameter] = params[parameter]
            else:
                raise TypeError(f"params argument must contain: {parameter}")
        self.fitted = True

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

    def __check_tail(self, tail: str) -> str:
        if tail not in self.available_tails:
            raise ValueError(f"tail must be one of {RobustPreprocessor.available_tails}")
        return tail

    def __calculate_quantiles(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        columns_num = len(self.params["column_names"])
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

    def fit(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
        alpha: Union[float, np.ndarray] = 0.05,
        tail: str = "both",
    ):
        """
        Fit to calculate robust parameters for the selected columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to calculate quantiles.
        column_names : ColumnNamesType
            One or number of columns in the dataframe.
        alpha : Union[float, np.ndarray], default: ``0.05``
            The percentage of removed data from head and tail.
        tail : str, default: ``"both"``
            Part of distribution to be removed.
            Can be ``"left"``, ``"right"`` or ``"both"``.

        Returns
        -------
        self : object
            Instance object.
        """
        self.params["column_names"] = wrap_cols(column_names)
        self._check_cols(dataframe, self.params["column_names"])
        self.params["alpha"] = self.__wrap_alpha(alpha)
        self.params["tail"] = self.__check_tail(tail)
        self.__calculate_quantiles(dataframe)
        self.fitted = True
        return self

    def transform(self, dataframe: pd.DataFrame, inplace: bool = False) -> Union[pd.DataFrame, None]:
        """
        Remove objects from the dataframe which are in the head, tail or both
        alpha parts of chosen metrics distributions.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to transform.
        inplace : bool, default: ``False``
            If ``True`` transforms the given dataframe, otherwise copy and
            returns an another one.

        Returns
        -------
        df : Union[pd.DataFrame, None]
            Transformed dataframe or None
        """
        self._check_fitted()
        self._check_cols(dataframe, self.params["column_names"])
        if self.verbose:
            prev_stats: List[Dict[str, float]] = log.RobustLogger.get_stats(dataframe, self.params["column_names"])

        transformed: pd.DataFrame = dataframe if inplace else dataframe.copy()
        if self.params["tail"] == "both":
            mask: pd.Series = (transformed[self.params["column_names"]] < self.params["quantiles"][:, 0]).any(
                axis=1
            ) | (transformed[self.params["column_names"]] > self.params["quantiles"][:, 1]).any(axis=1)
        elif self.params["tail"] == "left":
            mask = (transformed[self.params["column_names"]] < self.params["quantiles"].T).any(axis=1)
        elif self.params["tail"] == "right":
            mask = (transformed[self.params["column_names"]] > self.params["quantiles"].T).any(axis=1)
        bad_ids = transformed.loc[mask].index
        transformed.drop(bad_ids, inplace=True)

        if self.verbose:
            log.info_log(
                f"""Making {self.params['tail']}-tail robust transformation of columns {self.params['column_names']}
                 with alphas = {np.round(self.params['alpha'], 3)}"""
            )
            new_stats: Dict[str, float] = log.RobustLogger.get_stats(transformed, self.params["column_names"])
            log.RobustLogger.verbose_list(prev_stats, new_stats, self.params["column_names"])
        return None if inplace else transformed

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
        alpha: Union[float, np.ndarray] = 0.05,
        tail: str = "both",
        inplace: bool = False,
    ) -> Union[pd.DataFrame, None]:
        """
        Fit preprocessor parameters using given dataframe and transform it.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to calculate quantiles and for further transformation.
        column_names : ColumnNamesType
            One or number of columns in the dataframe.
        alpha : Union[float, np.ndarray], default: ``0.05``
            The percentage of removed data from head and tail.
        tail : str, default: ``"both"``
            Part of distribution to be removed.
            Can be ``"left"``, ``"right"`` or ``"both"``.
        inplace : bool, default: ``False``
            If ``True`` transforms the given dataframe, otherwise copy and
            returns an another one.

        Returns
        -------
        df : Union[pd.DataFrame, None]
            Transformed dataframe or None
        """
        self.fit(dataframe, column_names, alpha, tail)
        return self.transform(dataframe, inplace)


class IQRPreprocessor(AbstractFittableTransformer):
    """
    Unit for IQR transformation of the data to exclude outliers.

    It cuts the points from the distribution which are behind the range of
    0.25 quantile - 1,5 * iqr and 0.75 quantile + 1,5 * iqr
    for each given metric.


    Parameters
    ----------
    verbose : bool, default: ``True``
        If ``True`` will show info about the transformation of passed columns.

    Attributes
    ----------
    params : Dict
        Dictionary with operational parameters of the instance.
        Updated after calling the ``fit`` method.
    verbose : bool
        Verbose info flag.
    non_serializable_params: List
        List of the class parameters that should be converted to lists
        in order to serialize.
    fitted : bool
        Fit flag.

    Examples
    --------
    >>> iqr = IQRPreprocessor(verbose=True)
    >>> iqr.fit(dataframe, ['column1', 'column2'])
    >>> iqr.transform(dataframe, inplace=True)

    You can pass one or number of columns, if several columns are passed
    it will drop extreme values for each column.
    """

    non_serializable_params: List = ["medians", "quartiles"]

    def __str__(self) -> str:
        return "IQR outliers preprocessing"

    def __init__(self, verbose: bool = True) -> None:
        """
        IQRPreprocessor class constructor.
        """
        self.params = {"column_names": None, "medians": None, "quartiles": None}
        self.verbose = verbose
        super().__init__()

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
            key: (value if key not in IQRPreprocessor.non_serializable_params else value.tolist())
            for key, value in self.params.items()
        }

    def load_params_dict(self, params: Dict) -> None:
        """
        Load prefitted parameters form a dictionary.

        Parameters
        ----------
        params : Dict
            Dictionary with prefitted params.
        """
        for parameter in self.params:
            if parameter in params:
                if parameter in IQRPreprocessor.non_serializable_params:
                    self.params[parameter] = np.array(params[parameter])
                else:
                    self.params[parameter] = params[parameter]
            else:
                raise TypeError(f"params argument must contain: {parameter}")
        self.fitted = True

    def __calculate_params(
        self,
        dataframe: pd.DataFrame,
    ):
        X: np.ndarray = dataframe[self.params["column_names"]].values
        self.params["quartiles"] = np.quantile(X, (0.25, 0.75), axis=0).T
        self.params["medians"] = np.median(X, axis=0).T

    def fit(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
    ):
        """
        Fit to calculate iqr parameters for the selected columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to calculate quantiles.
        column_names : ColumnNamesType
            One or number of columns in the dataframe.

        Returns
        -------
        self : object
            Instance object.
        """
        self.params["column_names"] = wrap_cols(column_names)
        self._check_cols(dataframe, self.params["column_names"])
        self.__calculate_params(dataframe)
        self.fitted = True
        return self

    def transform(self, dataframe: pd.DataFrame, inplace: bool = False) -> Union[pd.DataFrame, None]:
        """
        Remove objects from the dataframe which are behind maximum and minimum
        values of boxplots for each metric distribution.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to transform.
        inplace : bool, default: ``False``
            If ``True`` transforms the given dataframe, otherwise copy and
            returns an another one.

        Returns
        -------
        df : Union[pd.DataFrame, None]
            Transformed dataframe or None
        """
        self._check_fitted()
        self._check_cols(dataframe, self.params["column_names"])
        if self.verbose:
            prev_stats: List[Dict[str, float]] = log.RobustLogger.get_stats(dataframe, self.params["column_names"])

        transformed: pd.DataFrame = dataframe if inplace else dataframe.copy()
        iqr: np.ndarray = self.params["quartiles"][:, 1] - self.params["quartiles"][:, 0]
        tail: np.ndarray = self.params["quartiles"][:, 0] - 1.5 * iqr
        head: np.ndarray = self.params["quartiles"][:, 1] + 1.5 * iqr
        mask: pd.Series = (
            (transformed[self.params["column_names"]] < tail) | (transformed[self.params["column_names"]] > head)
        ).any(axis=1)
        bad_ids = transformed.loc[mask].index
        transformed.drop(bad_ids, inplace=True)

        if self.verbose:
            log.info_log(f"Making IQR transformation of columns {self.params['column_names']}")
            new_stats: Dict[str, float] = log.RobustLogger.get_stats(transformed, self.params["column_names"])
            log.RobustLogger.verbose_list(prev_stats, new_stats, self.params["column_names"])
        return None if inplace else transformed

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
        inplace: bool = False,
    ) -> Union[pd.DataFrame, None]:
        """
        Fit preprocessor parameters using given dataframe and transform it.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to calculate quantiles and for further transformation.
        column_names : ColumnNamesType
            One or number of columns in the dataframe.
        inplace : bool, default: ``False``
            If ``True`` transforms the given dataframe, otherwise copy and
            returns an another one.

        Returns
        -------
        df : Union[pd.DataFrame, None]
            Transformed dataframe or None
        """
        self.fit(dataframe, column_names)
        return self.transform(dataframe, inplace)
