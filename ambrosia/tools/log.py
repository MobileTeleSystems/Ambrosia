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

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from ambrosia import types

NAME: str = "ambrosia_LOGGER"
PREFIX: str = "ambrosia LOGGER"


def info_log(message: str):
    logger = logging.Logger(NAME)
    logger.warning(f"{PREFIX}: {message}")


class RobustLogger:
    """
    Temporary class with methods for calculating and logging changes
    in the characteristics of metric distributions during the preprocessing.
    """

    @staticmethod
    def verbose(prev_stats: Dict[str, float], new_stats: Dict[str, float], name: str) -> None:
        """
        Verbose transormations to os.stdout.
        """
        for metric in prev_stats.keys():
            prev: float = prev_stats[metric]
            new: float = new_stats[metric]
            info_log(f"Change {metric} {name}: {prev:.4f} ===> {new:.4f}")

    @staticmethod
    def verbose_list(
        prev_stats: List[Dict[str, float]],
        new_stats: List[Dict[str, float]],
        names: types.ColumnNamesType,
    ) -> None:
        """
        Verbose iteratively.
        """
        for name, stat_1, stat_2 in zip(names, prev_stats, new_stats):
            info_log("\n")
            RobustLogger.verbose(stat_1, stat_2, name)

    @staticmethod
    def __calculate_stats(values: np.ndarray) -> Dict[str, float]:
        return {
            "Mean": np.mean(values),
            "Variance": np.var(values),
            "IQR": np.quantile(values, 0.75) - np.quantile(values, 0.25),
            "Range": np.max(values) - np.min(values),
        }

    @staticmethod
    def get_stats(
        df: pd.DataFrame,
        names: types.ColumnNamesType,
    ) -> List[Dict[str, float]]:
        """
        Get metrics for all columns.
        """
        result: List[Dict[str, float]] = []
        for name in names:
            err_msg: str = f"Column name is not in data frame, coumn - {name}"
            assert name in df.columns, err_msg
            result.append(RobustLogger.__calculate_stats(df[name].values))
        return result
