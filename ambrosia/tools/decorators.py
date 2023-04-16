import inspect
from functools import wraps


def filter_kwargs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        arg_names = inspect.getfullargspec(func).args
        correct_kwargs = {}
        for arg in arg_names:
            if arg in kwargs:
                correct_kwargs[arg] = kwargs[arg]
        return func(*args, **correct_kwargs)

    return wrapper


def tqdm_parallel_decorator(func):
    def wrapper(progress_bar, *args, **kwargs):
        output = func(*args, **kwargs)
        progress_bar.update(1)
        return output

    return wrapper
