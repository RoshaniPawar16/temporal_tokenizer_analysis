"""
Utilities to fix compatibility issues with the transformers library.
"""

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def get_cached_path():
    """
    Get the cached_path function from transformers, handling API changes.
    
    Returns:
        Function to resolve cached paths for tokenizer files
    """
    # Try different locations based on transformers version
    try:
        from transformers.file_utils import cached_path
        return cached_path
    except ImportError:
        try:
            from transformers.utils.hub import cached_path
            return cached_path
        except ImportError:
            try:
                from transformers import cached_path
                return cached_path
            except ImportError:
                logger.warning("Could not import cached_path from transformers, using fallback")
                # Simple fallback implementation
                def fallback_cached_path(url_or_filename):
                    if url_or_filename.startswith("http"):
                        import tempfile
                        import os
                        import requests
                        
                        # Create cache directory
                        cache_dir = os.path.join(tempfile.gettempdir(), "transformers_cache")
                        os.makedirs(cache_dir, exist_ok=True)
                        
                        # Create filename from URL
                        filename = os.path.basename(url_or_filename)
                        cached_file = os.path.join(cache_dir, filename)
                        
                        # Download if not already cached
                        if not os.path.exists(cached_file):
                            logger.info(f"Downloading {url_or_filename} to {cached_file}")
                            response = requests.get(url_or_filename)
                            with open(cached_file, "wb") as f:
                                f.write(response.content)
                        
                        return cached_file
                    else:
                        return url_or_filename
                
                return fallback_cached_path