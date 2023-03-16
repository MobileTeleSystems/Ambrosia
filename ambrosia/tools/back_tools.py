import contextlib
from typing import Callable, Tuple

import joblib
from tqdm.auto import tqdm

from ambrosia import types
from ambrosia.tools.decorators import tqdm_parallel_decorator


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


def handle_bootstrap_multiprocessing(
    n_jobs: int, criterion: int, func: Callable, desc: str, total: int, **kwargs
) -> Tuple[int, int]:
    """
    Handle parameters for bootstrap parallelism of experiment design computation tasks.
    """
    progress_bar = tqdm(desc=desc, total=total)
    if criterion == "bootstrap":
        bootstrap_n_jobs, n_jobs = n_jobs, 1
        func = tqdm_parallel_decorator(func)
        kwargs["progress_bar"] = progress_bar
    else:
        bootstrap_n_jobs: int = 1
        progress_bar = tqdm_joblib(progress_bar)
    return n_jobs, bootstrap_n_jobs, func, progress_bar, kwargs


def wrap_cols(cols: types.ColumnNamesType) -> types.ColumnNamesType:
    """
    Handle one or number of columns as list.
    """
    if isinstance(cols, types.ColumnNameType):
        cols = [cols]
    return cols
