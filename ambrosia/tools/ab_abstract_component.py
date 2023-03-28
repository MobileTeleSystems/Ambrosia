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

import json
from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from yaml import YAMLObjectMetaclass

import ambrosia.tools.pvalue_tools as pvalue_pkg
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


class AbstractFittableTransformer(ABC):
    """
    Abstract class for fittable transformer.

    Attributes
    ----------
    fitted : bool
        Indicator for the fit status of transformer.
    """

    def __init__(self):
        self.fitted: bool = False

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Call fit method before!")

    def _check_cols(self, dataframe: pd.DataFrame, columns: types.ColumnNameType) -> None:
        for column in columns:
            if column not in dataframe:
                raise ValueError(f"Column {column} is not in Data Frame columns list")

    @abstractmethod
    def get_params_dict(self) -> Dict:
        """
        Returns a dictionary with params.
        """

    @abstractmethod
    def load_params_dict(self, params: Dict) -> None:
        """
        Load model parameters from the dictionary.

        Parameters
        ----------
        params : Dict
            Dictionary with params.
        """

    @abstractmethod
    def fit(self):
        """
        Fit class parameters on some data.
        """

    @abstractmethod
    def transform(self):
        """
        Transform data using fitted parameters.
        """

    @abstractmethod
    def fit_transform(self):
        """
        Fit class parameters on some data and transform it.
        """

    def store_params(self, store_path: Path) -> None:
        """
        Parameters
        ----------
        store_path : Path
            Path where parameters will be stored in a json format.
        """
        with open(store_path, "w+") as file:
            json.dump(self.get_params_dict(), file)

    def load_params(self, load_path: Path) -> None:
        """
        Parameters
        ----------
        load_path : Path
            Path to json file with parameters.
        """
        with open(load_path, "r+") as file:
            params = json.load(file)
            self.load_params_dict(params)


class AbstractVarianceReducer(AbstractFittableTransformer):
    """
    Abstract class for Variance Reduction.

    Parameters
    ----------
    verbose : bool, default: ``True``
        If ``True`` will print in sys.stdout the information
        about the variance reduction.

    Attributes
    ----------
    params : Dict
        Parameters of the VarianceReducer that will be updated after
        calling ``fit`` method.
        These parameters are sufficient for data frames transformations
        and are used when loading and saving methods of instance are called.
        By the default these params dictionary contains  ``'target_column'``
        and ``'transformed_name'`` keys.
    verbose : bool
        Verbose info flag.
    fitted : bool
        Indicator for the fit status of transformer.
    """

    EPSILON: float = 1e-5

    @abstractmethod
    def __call__(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:  # pylint: disable=C0103
        """
        Returns transformed y values.

        Returns
        -------
        y_hat: np.ndarray
            y_hat = y - transformation(X) + MEAN(transformation(X))
        """

    def __init__(self, verbose: bool = True) -> None:
        self.params = {"target_column": None, "transformed_name": None}
        self.verbose: bool = verbose
        super().__init__()

    def _return_result(
        self, dataframe: pd.DataFrame, new_target: np.ndarray, inplace: bool
    ) -> Union[pd.DataFrame, None]:
        """
        Prepare and return resulted data frame with transformed target column.
        """
        if self.params["transformed_name"] is None:
            name = self.params["target_column"] + "_transformed"
        else:
            name = self.params["transformed_name"]
        df: pd.DataFrame = dataframe if inplace else dataframe.copy()
        df.loc[:, name] = new_target
        return df

    def _verbose(self, old_variance: float, new_variance: float) -> None:
        """
        Verbose method for transform operation Log.
        """
        part_of_variance: float = new_variance / (old_variance + AbstractVarianceReducer.EPSILON)
        log.info_log(f"After transformation {self}, the variance is {(part_of_variance * 100):.4f} % of the original")
        log.info_log(f"Variance transformation {old_variance:.4f} ===> {new_variance:.4f}")


def choose_on_table(alternatives: List[Any], dataframe) -> Any:
    """
    alternatives: [alternative_pandas, alternative_spark, ...]
    """
    if isinstance(dataframe, pd.DataFrame):
        return alternatives[0]
    elif isinstance(dataframe, types.SparkDataFrame):
        return alternatives[1]
    raise TypeError(f'Type of table must be one of {", ".join(AVAILABLE)}')


class DataframeHandler:
    @staticmethod
    def _handle_cases(__func_pandas: Callable, __func_spark: Callable, *args, **kwargs):
        """
        Helps handle cases with different types of dataframe in kwargs,
        available types - pandas, spark.
        """
        __func = choose_on_table([__func_pandas, __func_spark], kwargs[DATA])
        return __func(*args, **kwargs)

    @staticmethod
    def _handle_on_table(
        __func_pandas: Callable, __func_spark: Callable, variable: types.SparkOrPandas, *args, **kwargs
    ):
        """
        Helps handle cases with different types of dataframe as additional variable,
        available types - pandas, spark.
        """
        __func = choose_on_table([__func_pandas, __func_spark], variable)
        return __func(*args, **kwargs)


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

    def _make_ci(self, left_ci: np.ndarray, right_ci: np.ndarray, alternative: str) -> List:
        left_ci, right_ci = pvalue_pkg.choose_from_bounds(left_ci, right_ci, alternative)
        conf_intervals = list(zip(left_ci, right_ci))
        return conf_intervals

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
            "pvalue": self.calculate_pvalue(group_a, group_b, effect_type=effect_type, **kwargs),
            "effect": self.calculate_effect(group_a, group_b, effect_type=effect_type),
            "confidence_interval": self.calculate_conf_interval(
                group_a, group_b, alpha=alpha, effect_type=effect_type, **kwargs
            ),
        }
