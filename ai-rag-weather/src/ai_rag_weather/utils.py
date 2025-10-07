import time
from functools import wraps
from typing import Callable, TypeVar, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

"""Utility decorators and functions for timing, retrying, and text cleaning.

This module provides decorators to measure function execution time and implement retry logic,
along with a utility function for cleaning text by normalizing whitespace.
"""

F = TypeVar("F", bound=Callable[..., Any])

def timer(func: F) -> F:
    """Measure the execution time of a function.

    Args:
        func: The function to be timed.

    Returns:
        A wrapped function that returns a tuple of the original function's result and the elapsed time in seconds.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    return wrapper

def with_retries(*, attempts: int = 3, wait: float = 0.5):
    """Apply retry logic to a function with exponential backoff.

    Args:
        attempts: Number of retry attempts (default: 3).
        wait: Initial wait time multiplier for exponential backoff (default: 0.5 seconds).

    Returns:
        A decorator that retries the function on exceptions, with exponential backoff.
    """
    def decorator(func: F) -> F:
        return retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=wait),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )(func)
    return decorator

def clean_text(text: str) -> str:
    """Normalize whitespace in a text string.

    Args:
        text: The input string to clean.

    Returns:
        A string with normalized whitespace (multiple spaces collapsed to single spaces).
    """
    return " ".join(text.split())