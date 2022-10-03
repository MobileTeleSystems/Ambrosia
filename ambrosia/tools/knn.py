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

from typing import Any, List

import nmslib
import numpy as np


class NMTree:
    """
    KNN solution from NMSLIB.
    """

    def __init__(self, points: np.ndarray, payload: np.ndarray, ef_search: int):
        self.__was = set()
        self.__index = nmslib.init(space="l2")
        self.__index.addDataPointBatch(points)
        self.__index.createIndex()
        self.__index.setQueryTimeParams({"efSearch": ef_search})
        self.__payload = payload

    def query_batch(
        self,
        points: np.ndarray,
        payload: np.ndarray,
        out: List[List[Any]],
        group_size: int,
        amount: int = 1,
        threads: int = 1,
    ) -> None:
        """
        Write to out list indices for groups.

        Parameters
        ----------
        points : np.ndarray
            Batch of points for query
        payload : np.ndarray
            Some usable information
        out : List[List[Any]]
            Output list for indices
        group_size : int
            Sizes for groups
        amount : int, default: ``1``
            Amount groups exclude one
        threads : int, default: ``1``
            Amount of threads to be used
        """
        neighbours = self.__index.knnQueryBatch(points, k=group_size * amount, num_threads=threads)
        indices: List[np.ndarray] = [x[0] for x in neighbours]
        for i, index in enumerate(indices):
            current_answer: List = []
            for j in index:
                if self.__payload[j] in self.__was:
                    continue
                self.__was.add(self.__payload[j])
                current_answer.append(self.__payload[j])
                if len(current_answer) == amount:
                    out.append(current_answer + [payload[i]])
                    break
