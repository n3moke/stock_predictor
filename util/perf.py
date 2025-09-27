from datetime import datetime
import os
from time import perf_counter
from functools import wraps
from typing import Callable, Any
from loguru import logger # type: ignore

# decorator for measuring time of exceution of prompt with optional save to file
def measure_time(write_log: bool = False):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kargs) -> Any:
            from llm import LLM
            prompt_start_time: float = perf_counter()
            logger.warning(args)
            result: Any = await func(*args, **kargs)
            prompt_end_time:float = perf_counter()
            if write_log:
                for a in args:
                    if isinstance(a,list) and any(isinstance(d, dict) and 'stock' in d for d in a):
                        stock = a[0]['stock']
                    if isinstance(a,LLM.LLM_TYPE):
                        llm_type = a.value.replace(":","_")
                    
                    
                if stock is not None and llm_type is not None:
                        timestamp = datetime.now().strftime('%Y-%d-%m-%d_%H_%M_%S')
                        logger.warning(f"stock: {stock}, llm_type: {llm_type}, timestamp: {timestamp}")
                        log_string = f"{timestamp},{llm_type},{stock},{prompt_end_time - prompt_start_time:.3f}"
                        current_dir = os.path.dirname(os.path.abspath(__file__))

                        # Get the parent directory
                        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
                        os.makedirs(f'{parent_dir}/results', exist_ok=True)
                        # Construct full path to the file in the parent directory
                        file_path = os.path.join(parent_dir, "results\measurements.csv")
                        logger.info(f"creating (if not exits) and writing result of measure time into {file_path}")
                        
                        if not os.path.isfile(file_path):
                            logger.error("inside if")
                            with open(file_path, "w") as file:
                                file.write("timestamp_filename,llm_type,stock,measurement_in_s\n")
                        with open(f"{file_path}", "a") as file:   
                            file.write(log_string + "\n")  # Add a newline after each entry

            logger.warning(f"prompt of {func.__name__}() took {prompt_end_time - prompt_start_time:.3f} seconds")
            #after measuring time of excecution return function result
            return result
            
        return wrapper
    return decorator