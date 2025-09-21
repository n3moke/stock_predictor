import time
from time import perf_counter
from functools import wraps
from typing import Callable, Any
from loguru import logger # type: ignore

# Wrapper function for measuring time of exceution of prompt

def measure_time(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kargs) -> Any:
        prompt_start_time: float = perf_counter()
        result: Any = await func(*args, **kargs)
        prompt_end_time:float = perf_counter()

        logger.warning(f"prompt of {func.__name__}() took {prompt_end_time - prompt_start_time:.3f} seconds")
        #after measuring time of excecution return function result
        return result
    
    return wrapper