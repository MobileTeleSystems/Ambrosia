import contextlib
from typing import Any, Callable, Dict, Iterable, Optional, Union

import joblib
import numpy as np
from tqdm.auto import tqdm

from ambrosia import types
from ambrosia.tools.decorators import tqdm_parallel_decorator


def create_seed_sequence(length: int, entropy: Optional[Union[int, Iterable[int]]] = None) -> np.ndarray:
    """
    Create a seed sequence using ``numpy.random.SeedSequence`` class.
    Parameters
    ----------
    length : int
        Total length of a sequence.
    entropy : Union[int,Iterable[int]], optional
        The entropy for creating a ``SeedSequence``.
        Used to get a deterministic result.
    Returns
    -------
    seed_sequence : List
        The seed sequence of requested length.
    """
    rng = np.random.SeedSequence(entropy)
    seed_sequence: np.ndarray = rng.generate_state(length)
    return seed_sequence


from ambrosia import types


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument.
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):  # pylint: disable=W0235
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def handle_nested_multiprocessing(
    n_jobs: int, criterion: int, func: Callable, desc: str, total: int, **kwargs
) -> Dict[str, Any]:
    """
    Handle parameters for nested bootstrap parallelism of experiment design computation tasks.
    """
    progress_bar = tqdm(desc=desc, total=total)
    if criterion == "bootstrap":
        nested_n_jobs, n_jobs = n_jobs, 1
        func = tqdm_parallel_decorator(func)
        kwargs["progress_bar"] = progress_bar
    else:
        nested_n_jobs: int = 1
        progress_bar = tqdm_joblib(progress_bar)
    return {
        "n_jobs": n_jobs,
        "nested_n_jobs": nested_n_jobs,
        "parallel_func": func,
        "progress_bar": progress_bar,
        "kwargs": kwargs,
    }


def wrap_cols(cols: types.ColumnNamesType) -> types.ColumnNamesType:
    """
    Handle one or number of columns as list.
    """
    if isinstance(cols, types.ColumnNameType):
        cols = [cols]
    return cols
