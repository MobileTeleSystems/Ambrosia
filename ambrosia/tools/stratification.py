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

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

import ambrosia.tools.ab_abstract_component as ab_abstract
from ambrosia.tools import log


class Stratification(ab_abstract.StratificationUtil):
    """
    Stratification implementation
    https://en.wikipedia.org/wiki/Stratified_sampling

    Attributes
    ----------
    dataframe: pd.DataFrame
        Given data frame for stratification
    columns: List[Any]
        Columns for stratification
    threshold: Optional[int]
        If amount for current value of stratification is <= than threshold
        It wont be used for stratification. If None such regularization.
    verbose: bool
        Whenether to print information of not used strats
    strats: Dict[Any, pd.DataFrame]
        Dictionary with startification tables, values for current strat -> table

    Methods
    -------
    fit(dataframe: pd.DataFrame, columns: List[Any]) -> None
        Store data frame and stratification data
    strat_sizes() -> Dict[Any, int]
        Calculate stratification sizes
    get_test_inds(self, test_id: Iterable, id_column: Any=None) -> Dict[Any, Tuple[List, int]]
        Find test group ids for each stratification group and size of rest a group
    is_trained() -> bool
        Return true if fit method was called before
    is_not_trained() -> bool
        Return true if fit method was not called before
    size() -> int
        Return size considering droped rows using threshold in stratification
        If threshold = None => size() = len(dataframe)
    groups()
        Returns items for stratification
    """

    def __init__(self, threshold: Optional[int] = None, verbose: bool = False):
        """
        Parameters
        ----------
        threshold : Optional[int]
            Threshold for stratification group sizes
        verbose : bool, default: ``False``
            Whenether to print information of not used strats
        """
        super().__init__()
        self.dataframe = None
        self.columns = None
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, dataframe: pd.DataFrame, columns: Optional[List[Any]] = None) -> None:
        """
        Store stratification data.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Pandas Data Frame for stratification
        columns : Optional[List[Any]]
            Columns of dataframe for straitification
        """
        self.dataframe = dataframe
        self.columns = columns
        if columns is None:
            self.strats = {ab_abstract.EmptyStratValue.NO_STRATIFICATION: dataframe}
            return
        self.strats: Dict[Any, pd.DataFrame] = {}
        for values, table in dataframe.groupby(columns):
            if self.threshold is not None and self.dataframe.shape[0] <= self.threshold:
                if self.verbose:
                    log.info_log("Stratification group with values:")
                    for val, name in zip(values, columns):
                        log.info_log(f"Column {name}: {val}")
                continue
            self.strats[values] = table

    def strat_sizes(self) -> Dict[Any, int]:
        """
        Returns sizes for each strat
        """
        self._check_fit()
        sizes: Dict[Any, int] = {}
        for value, table in self.strats.items():
            sizes[value] = table.shape[0]
        return sizes

    @staticmethod
    def __corresponding_strat(test_id: Iterable, strat_id: Iterable) -> List:
        """
        Filter test id for given stratification group.
        """
        return list(filter(lambda x: x in strat_id, test_id))

    def get_test_inds(self, test_id: Iterable, id_column: Any = None) -> Dict[Tuple, Tuple[List, int]]:
        """
        Returns test ids for each strat and amount of remaining ids in this strat.
        Basically use id of data frame, if id_column is set use it

        Parameters
        ----------
        test_id : Iterable
            Ids for test group
        id_column : Any, default: None

        Returns
        -------
        test_ids : Dict[Tuple, Tuple[List, int]]
            Dictionary for each value from stratification with test ids and
            not used ids from this strat
        """
        if self.empty_strat():
            other_amount: int = self.size() - len(test_id)
            return {ab_abstract.EmptyStratValue.NO_STRATIFICATION: (list(test_id), other_amount)}

        if id_column is not None:
            error_column_name: str = f"""Column - {id_column},
             is not in list of columns - {", ".join(self.dataframe.columns)}"""
            if id_column not in self.dataframe.columns:
                raise ValueError(error_column_name)
        test_ids: Dict[Tuple, Tuple[List, int]] = {}
        for value, table in self.strats.items():
            if id_column is not None:
                ids: List = Stratification.__corresponding_strat(test_id, table[id_column].values)
            else:
                ids: List = Stratification.__corresponding_strat(test_id, table.index.values)
            test_ids[value] = (ids, table.shape[0] - len(ids))
        return test_ids
