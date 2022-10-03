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

from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Tuple

import pandas as pd
from tqdm.auto import tqdm

ROUND_DIGITS_TABLE: int = 3
ROUND_DIGITS_PERCENT: int = 1


class Selector:
    """
    Variate params for designing experiment.
    """

    def __init__(
        self, solver: Callable, selecting_params: Dict[str, Iterable], n_jobs: int = 1, use_tqdm: bool = True, **kwargs
    ) -> None:
        self.solver = solver
        self.sel_params = selecting_params
        self.kwargs = kwargs
        self.n_jobs = n_jobs
        self.tqdm = use_tqdm

    def set_params(self, params: Tuple[Any, ...]) -> None:
        for param_name, param_value in zip(self.sel_params.keys(), params):
            self.kwargs[param_name] = param_value

    def iterate_params(self) -> Tuple[Tuple[Any, ...], List[Any]]:
        parameters: Tuple = tuple(product(*list(self.sel_params.values())))
        result = []
        iter_set = parameters if not self.tqdm else tqdm(parameters)
        for params in iter_set:
            self.set_params(params)
            result.append(self.solver(**self.kwargs))
        return parameters, result

    @staticmethod
    def handle_numeric(report: pd.DataFrame, as_numeric: bool) -> None:
        if not as_numeric:
            report["effect"] = (round((report["effect"] - 1) * 100, ROUND_DIGITS_PERCENT)).astype(str) + "%"
        report["errors"] = tuple(zip(report["alpha"], report["beta"]))

    def get_table_size(self, as_numeric: bool = False) -> pd.DataFrame:
        parameters, group_sizes_list = self.iterate_params()
        report = pd.DataFrame(list(parameters), columns=["effect", "alpha", "beta"]).join(
            pd.DataFrame(list(group_sizes_list), columns=["sample_sizes"])
        )
        self.handle_numeric(report, as_numeric)
        report = report.pivot(index="effect", columns="errors", values="sample_sizes")
        report = report.sort_values(report.columns[0])
        return report

    def get_table_effect(self, as_numeric: bool = False) -> pd.DataFrame:
        parameters, effects = self.iterate_params()
        report = pd.DataFrame(list(parameters), columns=["group_sizes", "alpha", "beta"]).join(
            pd.DataFrame(list(effects), columns=["effect"])
        )
        self.handle_numeric(report, as_numeric)
        report = report.pivot(index="group_sizes", columns="errors", values="effect")
        report = report.sort_index()
        return report
