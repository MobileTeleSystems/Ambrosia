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

from typing import Any, Dict, Iterable, Optional

import pyspark.sql.functions as spark_funcs
from pyspark.sql import Window

import ambrosia.tools.ab_abstract_component as ab_abstract
from ambrosia import types

EMPTY_VALUE: int = 0
STRAT_GROUPS: str = "__ambrosia_strat"


class Stratification(ab_abstract.StratificationUtil):
    """
    Stratification implementation for spark tables
    https://en.wikipedia.org/wiki/Stratified_sampling
    """

    def fit(self, dataframe: types.SparkDataFrame, columns: Optional[Iterable[types.ColumnNameType]] = None):
        if columns is None:
            self.strats = {ab_abstract.EmptyStratValue.NO_STRATIFICATION: dataframe}
            return

        window = Window.orderBy(*columns).partitionBy(spark_funcs.lit(EMPTY_VALUE))
        with_groups = dataframe.withColumn(STRAT_GROUPS, spark_funcs.dense_rank().over(window))
        amount_of_strats: int = with_groups.select(spark_funcs.max(STRAT_GROUPS)).collect()[0][0]

        self.strats: Dict[int, types.SparkDataFrame] = {}
        for strat_value in range(1, amount_of_strats + 1):
            strat_table = with_groups.where(spark_funcs.col(STRAT_GROUPS) == strat_value)
            self.strats[strat_value] = strat_table.drop(STRAT_GROUPS)

    def strat_sizes(self) -> Dict[int, int]:
        """
        Returns size of each stratification group
        """
        self._check_fit()
        sizes: Dict[Any, int] = {}
        for value, table in self.strats.items():
            sizes[value] = table.count()
        return sizes
