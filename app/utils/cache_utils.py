"""
Caching utilities for LRU-based model management.

Simple helpers for cache management (though we use OrderedDict in services).
"""

import logging
from typing import TypeVar, Generic, Dict, Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """
    Simple LRU (Least Recently Used) cache implementation.
    
    When max_size is exceeded, the oldest entry is evicted.
    """
    
    def __init__(self, max_size: int = 10):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
        """
        self.max_size = max_size
        self.cache: Dict[K, V] = OrderedDict()
        logger.debug(f"LRUCache initialized with max_size={max_size}")
    
    def get(self, key: K) -> Optional[V]:
        """
        Get value from cache.
        
        Accessing a value marks it as recently used.
        
        Args:
            key: Cache key
        
        Returns:
            Value if found, None otherwise
        """
        if key not in self.cache:
            return None
        
        # Move to end (mark as recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: K, value: V) -> None:
        """
        Put value into cache.
        
        If key already exists, updates it and marks as recently used.
        If cache is full, evicts oldest entry.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update existing key and move to end
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Add new key
            self.cache[key] = value
            # Check if we exceeded max size
            if len(self.cache) > self.max_size:
                # Remove oldest (first) entry
                oldest_key, _ = self.cache.popitem(last=False)
                logger.debug(f"LRU evicted: {oldest_key}")
    
    def evict(self, key: K) -> bool:
        """
        Manually evict an entry from cache.
        
        Args:
            key: Key to evict
        
        Returns:
            True if evicted, False if key not found
        """
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Cache evicted: {key}")
            return True
        return False
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        self.cache.clear()
        logger.debug("Cache cleared")
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def keys(self):
        """Get all cache keys."""
        return self.cache.keys()
    
    def values(self):
        """Get all cache values."""
        return self.cache.values()
    
    def items(self):
        """Get all cache items."""
        return self.cache.items()
