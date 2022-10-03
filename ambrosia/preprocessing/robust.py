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
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from ambrosia import types
from ambrosia.tools import log


class RobustPreprocessor:
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
    >>> robust = RobustPreprocessor(dataframe, verbose=True)
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

        was_stats: List[Dict[str, float]] = RobustPreprocessor.__get_stats(self.dataframe, column_names)

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
            new_stats: Dict[str, float] = RobustPreprocessor.__get_stats(df, column_names)
            RobustPreprocessor.__verbose_list(was_stats, new_stats, column_names)
        return None if inplace else transformed
