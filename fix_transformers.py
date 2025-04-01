"""
Compatibility module for transformers library changes.
Provides a stable interface for cached_path function 
which has moved between different versions.
"""

import logging

logger = logging.getLogger(__name__)

def get_cached_path():
    """
    Get the cached_path function from transformers, handling API changes.
    
    The function has moved between different versions of the transformers library:
    - In older versions: transformers.file_utils
    - In newer versions: transformers.utils.hub
    
    Returns:
        The cached_path function or a simple fallback implementation
    """
    try:
        # Try the newer location first (post v4.0)
        from transformers.utils.hub import cached_path
        logger.info("Using cached_path from transformers.utils.hub")
        return cached_path
    except ImportError:
        try:
            # Try the older location
            from transformers.file_utils import cached_path
            logger.info("Using cached_path from transformers.file_utils")
            return cached_path
        except ImportError:
            logger.warning("Could not import cached_path from transformers, using fallback")
            # Define a simple alternative if not available
            def simple_cached_path(file_path, *args, **kwargs):
                """Simple implementation that just returns the path"""
                return file_path
            return simple_cached_path