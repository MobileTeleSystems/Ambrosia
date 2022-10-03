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

from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from yaml import YAMLObjectMetaclass

from ambrosia import types
from ambrosia.tools import log

AVAILABLE: List[str] = ["pandas", "spark"]
DATA: str = "dataframe"


class ABMetaClass(ABCMeta, YAMLObjectMetaclass):
    """
    A metaclass to solve the problem 'metaclass conflict'.
    If you want to derive from ABToolAbstract set metaclass=ABMetaClass.
    """


class ABToolAbstract(ABC):
    _SavedArgumentType = Any
    _GivenArgument_Type = Any
    _PrepareArgumentsType = Dict[str, Tuple[_SavedArgumentType, _GivenArgument_Type]]
    _PRECISION_DIGITS = 4

    @abstractmethod
    def run(self):
        """
        Each derived class must implement this method.
        """

    @staticmethod
    def _prepare_arguments(_args: _PrepareArgumentsType) -> types._UsageArgumentsType:
        """
        Protected method for derived classes.
        Choose values for attributes (saved from __init__, passed to method run)
        If passed value not None, then it will be used, otherwise - saved value

        Parameters
        ----------
        _args : _PrepareArgumentsType
            Dictionary with
                keys - name of attribute of derived class
                values - (saved argument from constructor, given argument for run method)

        Returns
        -------
        chosen_args : _UsageArgumentsType
            Dictionary with
                keys - name of attribute
                values - chosen arg
        """
        chosen_args: types._UsageArgumentsType = {}
        for arg_name, [saved_argument, given_argument] in _args.items():
            exception_message: str = f"""Value for argument - {arg_name},
             was not set! Define it via setter method or pass as argument"""
            choice = given_argument if given_argument is not None else saved_argument
            if choice is None:
                raise ValueError(exception_message)
            chosen_args[arg_name] = choice
        return chosen_args


class AbstractVarianceReduction(ABC):
    """
    Abstract class for Variance Reduction.

    Attributes
    ----------
    df : pd.DataFrame
        Given data frame with target column
    target_column : str
        Column from df, must be str type, it matches Y (Y => Y_hat)
    cov : pd.DataFrame
        Covariance matrix of given data frame
    """

    EPSILON: float = 1e-5

    @abstractmethod
    def store_params(self, store_path: Path) -> None:
        """
        Store model params into file.
        """

    @abstractmethod
    def load_params(self, load_path: Path) -> None:
        """
        Load model params from file.
        """

    @abstractmethod
    def fit(self):
        """
        Fit class parameters on some data.
        """

    @abstractmethod
    def fit_transform(self):
        """
        Fit class parameters on some data and transform it.
        """

    @abstractmethod
    def __call__(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:  # pylint: disable=C0103
        """
        Returns transformed y values.

        Returns
        -------
        y_hat: np.ndarray
            y_hat = y - transformation(X) + MEAN(transformation(X))
        """

    def __init__(self, dataframe: pd.DataFrame, target_column: types.ColumnNameType, verbose: bool = True) -> None:
        self.df = dataframe
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column {target_column} is not in Data Frame columns list")
        self.target_column = target_column
        self.fitted: bool = False
        self.verbose: bool = verbose

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Call fit method !!!")

    def _check_columns(self, columns: types.ColumnNameType) -> None:
        for column in columns:
            if column not in self.df.columns:
                raise ValueError(f"Column {column} is not in Data Frame columns list")

    def _return_result(
        self, new_target: np.ndarray, inplace: bool, name: Union[types.ColumnNameType, None]
    ) -> Union[pd.DataFrame, None]:
        if name is None:
            name = self.target_column + "_transformed"
        if inplace:
            self.df.loc[:, name] = new_target
            return None
        new_df: pd.DataFrame = self.df.copy()
        new_df.loc[:, name] = new_target
        return new_df

    def _verbose(self, old_variance: float, new_variance: float) -> None:
        """
        Verbose method for transform operation Log.
        """
        part_of_variance: float = new_variance / (old_variance + AbstractVarianceReduction.EPSILON)
        log.info_log(f"After transformation {self}, the variance is {(part_of_variance * 100):.4f} % of the original")
        log.info_log(f"Variance transformation {old_variance:.4f} ===> {new_variance:.4f}")

    @abstractmethod
    def transform(
        self, covariate_columns: List[types.ColumnNameType], inplace: bool = False, name: Optional[str] = None
    ) -> None:
        pass


class DataframeHandler:
    @staticmethod
    def _handle_cases(__func_pandas: Callable, __func_spark: Callable, *args, **kwargs):
        """
        Helps handle cases with different types of dataframe in kwargs,
        available types - pandas, spark.
        """
        if isinstance(kwargs[DATA], pd.DataFrame):
            return __func_pandas(*args, **kwargs)
        elif isinstance(kwargs[DATA], types.SparkDataFrame):
            return __func_spark(*args, **kwargs)
        else:
            raise TypeError(f'Type of table must be one of {", ".join(AVAILABLE)}')

    @staticmethod
    def _handle_on_table(
        __func_pandas: Callable, __func_spark: Callable, variable: types.SparkOrPandas, *args, **kwargs
    ):
        """
        Helps handle cases with different types of dataframe as additional variable,
        available types - pandas, spark.
        """
        if isinstance(variable, pd.DataFrame):
            return __func_pandas(*args, **kwargs)
        elif isinstance(variable, types.SparkDataFrame):
            return __func_spark(*args, **kwargs)
        else:
            raise TypeError(f'Type of table must be one of {", ".join(AVAILABLE)}')


class SimpleDesigner(ABC, DataframeHandler):
    """
    Simple designer is the interface for designers for each dataframe and method.

    kwargs must contain parameter dataframe.
    """

    @abstractmethod
    def size_design(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def effect_design(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def power_design(self, **kwargs) -> pd.DataFrame:
        pass


class EmptyStratValue(Enum):
    NO_STRATIFICATION = 0


class StratificationUtil(ABC):
    """
    Stratification element
    https://en.wikipedia.org/wiki/Stratified_sampling
    """

    def __init__(self):
        self.strats = None

    @abstractmethod
    def fit(self, dataframe: types.SparkOrPandas, columns) -> None:
        pass

    @abstractmethod
    def strat_sizes(self) -> Dict[Any, int]:
        pass

    def is_trained(self) -> bool:
        """
        Returns whether the fit method was called.
        """
        return self.strats is not None

    def empty_strat(self) -> bool:
        """
        Check if there was no stratification.
        """
        return list(self.strats.keys()) == [EmptyStratValue.NO_STRATIFICATION]

    def _check_fit(self) -> None:
        """
        If fit method was not called before throw RuntimeError.
        """
        if not self.is_trained():
            raise RuntimeError("Call fit method !")

    def groups(self):
        """
        Returns
        -------
        items : Dict[tuple, SparkOrPandas] items
            iterable object (stratification value or group number, table for current stratification group)
        """
        self._check_fit()
        return self.strats.items()

    def size(self) -> int:
        """
        Calculate total rows amount considering filtering by threshold.

        Returns
        -------
        size: int
            Total size of filtered data
        """
        total_size: int = 0
        for size in self.strat_sizes().values():
            total_size += size
        return total_size

    def get_group_sizes(self, group_size: int) -> Dict[Any, int]:
        """
        Calculate size of group for each strat by total size of group.
        size(G_j) = group_size
        group_j_size = round(group_size * strat_size / total_size)

        Parameters
        ----------
        group_size : int
            Size for groups A, B, C ...

        Returns
        -------
        group_sizes : Dict[Any, int]
            Sizes for groups corresponding to stratification groups
        """
        filtered_size: int = self.size()
        strat_sizes: Dict[Tuple, int] = self.strat_sizes()
        group_sizes: Dict[Tuple, int] = {}
        for strat_value in self.strats:
            group_sizes[strat_value] = int(np.floor(group_size * strat_sizes[strat_value] / filtered_size))
        return group_sizes


class StatCriterion(ABC):
    """
    StatCriterion is the interface for arbitrary statistical criteria.
    """

    @abstractmethod
    def calculate_pvalue(self, group_a: Iterable[float], group_b: Iterable[float], **kwargs) -> np.ndarray:
        pass


class ABStatCriterion(StatCriterion):
    """
    ABStatCriterion is is the abstract class for statistical criterion used for design and test.
    """

    required_attributes = ["alias", "implemented_effect_types"]

    @classmethod
    def _send_type_error_msg(cls):
        error_msg = f"Choose effect_type from {cls.implemented_effect_types}"  # pylint: disable=E1101
        return error_msg

    @abstractmethod
    def calculate_effect(self, group_a: Iterable[float], group_b: Iterable[float], effect_type: str) -> np.ndarray:
        pass

    @abstractmethod
    def calculate_conf_interval(
        self, group_a: Iterable[float], group_b: Iterable[float], alpha: types.StatErrorType, effect_type: str, **kwargs
    ) -> List[Tuple]:
        pass

    def get_results(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        alpha: types.StatErrorType = 0.05,
        effect_type: str = "absolute",
        **kwargs,
    ) -> types.StatCriterionResult:
        return {
            "first_type_error": alpha,
            "pvalue": self.calculate_pvalue(group_a, group_b, **kwargs),
            "effect": self.calculate_effect(group_a, group_b, effect_type),
            "confidence_interval": self.calculate_conf_interval(group_a, group_b, alpha, effect_type, **kwargs),
        }
