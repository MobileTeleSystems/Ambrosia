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

import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ambrosia.tools.theoretical_tools import (
    get_table_sample_size, make_labels_effects, get_minimal_effects_table,
    get_power_table,
)


class TheoreticalBinary(ABC):
    '''
    Base class for theoretical methods for binary methods
    To get corresponding solver for given method
        >>> solver = TheoreticalBinary.METHODS[method]()
    All inheritors must implement methods
        >>> [get_table_sample_size_on_effect, get_table_effect_on_sample_size, get_table_power_on_size_and_delta]
    '''

    METHODS = {}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        TheoreticalBinary.METHODS[cls._method_name] = cls

    @abstractmethod
    def get_table_sample_size_on_effect(self, 
                      p_a: float,
                      second_errors: tp.Iterable[float],
                      first_errors: tp.Iterable[float],
                      delta_relative_values: tp.Iterable[float]) -> pd.DataFrame:
        '''
        Return table with sizes designed for experiment

        Parameters
        ----------
        p_a: float
            Conversion for group A
        second_errors: Iterable of floats
            2nd type errors.
            e.x.: [0.01, 0.05, 0.1]
        first_errors: Iterable of floats
            1st type errors.
            e.x.: [0.01, 0.05, 0.1]
        delta_relative_values: Iterable of floats
            List of effects which we want to detect.
            e.x.: [0.95, 1.05]

        Returns
        -------
        df_results: Pandas Dataframe
            Table with minimal sample sizes for each effect and error from input data.
        '''
        pass

    @abstractmethod
    def get_table_effect_on_sample_size(self,
                        p_a: float,
                        sample_sizes: tp.Iterable[int],
                        second_errors: tp.Iterable[float],
                        first_errors: tp.Iterable[float]) -> pd.DataFrame:
        '''
        Return table with effects designed for experiment

        Parameters
        ----------
        p_a: float
            Conversion for group A
        sample_sizes: Iterable of integers
            List of sample sizes which we want to check.
            e.x.: [100, 200, 1000]
        second_errors: Iterable of floats
            2nd type errors.
            e.x.: [0.01, 0.05, 0.1]
        first_errors: Iterable of floats
            1st type errors.
            e.x.: [0.01, 0.05, 0.1]

        Returns
        -------
        df_results: Pandas Dataframe
            Table with minimal effects for each sample size and error from input data.
        '''
        pass

    @abstractmethod
    def get_table_power_on_size_and_delta(self,
                       p_a: float,
                       sample_sizes: tp.Iterable[int],
                       delta_relative_values: tp.Iterable[float],
                       first_errors: tp.Iterable[float]) -> pd.DataFrame:
        '''
        Return table with powers designed for experiment

        Parameters
        ----------
        p_a: float
            Conversion for group A
        sample_sizes: Iterable of integers
            List of sample sizes which we want to check.
            e.x.: [100, 200, 1000]
        delta_relative_values: Iterable of floats
            List of effects which we want to detect.
            e.x.: [0.95, 1.05]
        first_errors: Iterable of floats
            1st type errors.
            e.x.: [0.01, 0.05, 0.1]

        Returns
        -------
        df_results: Pandas Dataframe
            Table with power for each sample size, first type error and effects from input data.
        '''
        pass



class TheoreticalVariationStabilization(TheoreticalBinary):

    _method_name: str = "asin_var_stabil"

    def _transform(self, x: float) -> float:
        return 2 * np.arcsin(np.sqrt(x))

    def _reverse(self, y: float) -> float:
        return np.sin(y / 2) ** 2

    def _get_mean_rel_effect(self, p_a: float):
        return self._transform(p_a)

    def _reverse_effect(self, df: pd.DataFrame, p_a: float):
        p_b = self._reverse(df.to_numpy().astype(float) * self._transform(p_a))
        df[::] = (p_b - p_a) / p_a * 100
        df = df.applymap(lambda x: f"{x:.2f} %")
        return df

    def _recalc_asin_effects(self, p_a: float, effects: tp.Iterable[float]) -> np.ndarray:
        effects = np.array(effects)
        p_b = p_a * effects
        effects_asin = 1 + (self._transform(p_a) - self._transform(p_b)) / self._transform(p_a)
        return effects_asin

    def get_table_sample_size_on_effect(self,
                      p_a: float,
                      second_errors: tp.Iterable[float],
                      first_errors: tp.Iterable[float],
                      delta_relative_values: tp.Iterable[float]) -> pd.DataFrame:
        effects_asin = self._recalc_asin_effects(p_a, delta_relative_values)
        labels = make_labels_effects(delta_relative_values)
        size = get_table_sample_size(self._transform(p_a),
                                     std=1,
                                     effects=effects_asin,
                                     first_errors=first_errors,
                                     second_errors=np.array(second_errors),
                                     effects_labels=labels)
        return size

    def get_table_effect_on_sample_size(self,
                        p_a: float,
                        sample_sizes: tp.Iterable[int],
                        second_errors: tp.Iterable[float],
                        first_errors: tp.Iterable[float]) -> pd.DataFrame:
        effects = get_minimal_effects_table(self._transform(p_a),
                                            std=1,
                                            sample_sizes=sample_sizes,
                                            first_errors=first_errors,
                                            second_errors=np.array(second_errors),
                                            as_numeric=True)
        effects = self._reverse_effect(effects, p_a)
        return effects

    def get_table_power_on_size_and_delta(self,
                       p_a: float,
                       sample_sizes: tp.Iterable[int],
                       delta_relative_values: tp.Iterable[float],
                       first_errors: tp.Iterable[float]) -> pd.DataFrame:        
        if isinstance(first_errors, float):
            first_errors = [first_errors]

        effects_asin = self._recalc_asin_effects(p_a, delta_relative_values)
        labels = make_labels_effects(delta_relative_values)
        powers = get_power_table(self._transform(p_a),
                                 std=1,
                                 sample_sizes=sample_sizes,
                                 first_errors=first_errors,
                                 effects=effects_asin,
                                 effects_labels=labels)
        return powers
