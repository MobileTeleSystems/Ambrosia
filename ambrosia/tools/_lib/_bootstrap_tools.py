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

from typing import Callable, List

import ambrosia.tools._lib._tools_aide as pkg_solvers


class EmpiricSolution:
    """
    Unit helps design experiment via empiric distribution.
    """

    def __init__(self, power_calulation: Callable, desired_power: float, variating_param_name: List[str]) -> None:
        """
        power_calculation(**kwargs) - Empiric estiation of power of criterion
        """
        self.power_calc = power_calulation
        self.desired_power = desired_power
        self.var_param_name = variating_param_name

    def power(self, **kwargs) -> float:
        return self.power_calc(**kwargs)


class EmpiricSizeSolution(EmpiricSolution):
    """
    Special unit designing sample size via bootstrap.
    """

    def calc_upper_bound(self, **kwargs) -> int:
        deg: int = pkg_solvers.helper_bin_search_upper_bound_size(
            self.power_calc, self.desired_power, self.var_param_name, **kwargs
        )
        return 2**deg

    def calc_binary(self, **kwargs) -> int:
        upper_bound: int = self.calc_upper_bound(**kwargs) * 2
        return pkg_solvers.helper_binary_search_optimal_size(
            self.power_calc, self.desired_power, upper_bound, self.var_param_name, **kwargs
        )


class EmpiricEffectSolution(EmpiricSolution):
    """
    Special unit designing MDE via bootstrap.
    """

    def calc_upper_bound(self, **kwargs) -> float:
        deg: float = pkg_solvers.helper_bin_searh_upper_bound_effect(self.power_calc, self.desired_power, **kwargs)
        return 2**deg

    def calc_binary(self, **kwargs) -> int:
        upper_bound: float = self.calc_upper_bound(**kwargs)
        return pkg_solvers.helper_binary_search_effect_with_injection(
            self.power_calc, self.desired_power, upper_bound, self.var_param_name, **kwargs
        )
