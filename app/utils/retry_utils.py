"""
Retry utilities for API calls
"""
import time
import random
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

def exponential_backoff_retry(func: Callable, max_retries: int = 3, base_delay: float = 1.0) -> Any:
    """
    Retry function with exponential backoff for rate limiting
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            
            if attempt == max_retries - 1:
                # Last attempt failed
                raise e
            
            if "rate_limit" in error_str or "429" in error_str:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limited, retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            else:
                # Non-rate-limit error, don't retry
                raise e
    
    return None