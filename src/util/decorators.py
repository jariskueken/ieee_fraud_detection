import time
from functools import wraps
import logging


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        stop = time.time()

        elapsed = stop - start
        logging.debug(f'function {func.__name__} ran in {elapsed:.4f} seconds')
        return result
    return wrapper

