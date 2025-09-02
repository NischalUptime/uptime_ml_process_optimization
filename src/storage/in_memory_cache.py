"""
In-memory cache for optimization artifacts.
Simple in-memory implementation without Redis dependency.
"""

import structlog
from typing import Any, Dict, Optional, Callable
from datetime import datetime, timedelta
import threading


class InMemoryCache:
    """In-memory cache for strategy configurations and timestamps."""
    
    def __init__(self):
        """Initialize cache with in-memory storage."""
        self.logger = structlog.get_logger()
        
        # In-memory storage
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()  # Thread-safe access
        
        # Cache key prefixes
        self.PREFIX_CONFIG = "strategy:config:"
        self.PREFIX_TIMESTAMP = "strategy:timestamp:"
        self.PREFIX_MODEL = "strategy:model:"
        self.PREFIX_SCALER = "strategy:scaler:"
        self.PREFIX_VERSION = "strategy:version:"
        self.PREFIX_LAG_OFFSET_BOUNDS = "strategy:lag_offset_bounds:"
        self.logger.info("Initialized in-memory cache")
    
    def get_last_run_timestamp_with_cache(self, loader_func: Callable[[], Optional[datetime]]) -> Optional[datetime]:
        """
        Get last run timestamp with caching support.
        
        Args:
            loader_func: Function to load timestamp if not cached
            
        Returns:
            Last run timestamp or None
        """
        cache_key = f"{self.PREFIX_TIMESTAMP}last_run"
        
        try:
            with self._lock:
                # Try to get from cache first
                if cache_key in self._cache:
                    self.logger.debug("Retrieved last run timestamp from memory cache")
                    return self._cache[cache_key]
                
                # Not in cache, use loader function
                timestamp = loader_func()
                if timestamp is not None:
                    # Cache the result
                    self._cache[cache_key] = timestamp
                    self.logger.debug("Cached last run timestamp in memory")
                
                return timestamp
                
        except Exception as e:
            self.logger.error(f"Error in timestamp caching: {e}")
            # Fallback to loader function
            return loader_func()
    
    def set_cached_last_run_timestamp(self, timestamp: datetime) -> bool:
        """
        Set/update the cached last run timestamp.
        
        Args:
            timestamp: Timestamp to cache
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = f"{self.PREFIX_TIMESTAMP}last_run"
        
        try:
            with self._lock:
                self._cache[cache_key] = timestamp
                self.logger.debug(f"Updated cached last run timestamp: {timestamp}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to cache timestamp: {e}")
            return False
    
    def get_cached_config(self, config_version: str) -> Optional[Dict]:
        """
        Get cached strategy configuration by version.
        
        Args:
            config_version: Configuration version identifier
            
        Returns:
            Cached configuration or None
        """
        cache_key = f"{self.PREFIX_CONFIG}{config_version}"
        
        try:
            with self._lock:
                config = self._cache.get(cache_key)
                if config is not None:
                    self.logger.debug(f"Retrieved config version {config_version} from memory cache")
                return config
        except Exception as e:
            self.logger.error(f"Failed to get cached config: {e}")
            return None
    
    def set_cached_config(self, config_version: str, config_data: Dict) -> bool:
        """
        Cache strategy configuration by version.
        
        Args:
            config_version: Configuration version identifier
            config_data: Configuration data to cache
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = f"{self.PREFIX_CONFIG}{config_version}"
        
        try:
            with self._lock:
                self._cache[cache_key] = config_data
                self.logger.debug(f"Cached config version {config_version}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to cache config: {e}")
            return False
    
    def get_cached_model(self, model_path: str) -> Optional[Any]:
        """
        Get cached model by path.
        
        Args:
            model_path: Model file path identifier
            
        Returns:
            Cached model or None
        """
        cache_key = f"{self.PREFIX_MODEL}{model_path}"
        
        try:
            with self._lock:
                model = self._cache.get(cache_key)
                if model is not None:
                    self.logger.debug(f"Retrieved model {model_path} from memory cache")
                return model
        except Exception as e:
            self.logger.error(f"Failed to get cached model: {e}")
            return None
    
    def set_cached_model(self, model_path: str, model_data: Any) -> bool:
        """
        Cache model by path.
        
        Args:
            model_path: Model file path identifier
            model_data: Model data to cache
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = f"{self.PREFIX_MODEL}{model_path}"
        
        try:
            with self._lock:
                self._cache[cache_key] = model_data
                self.logger.debug(f"Cached model {model_path}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to cache model: {e}")
            return False
    
    def get_cached_scaler(self, scaler_path: str) -> Optional[Any]:
        """
        Get cached scaler by path.
        
        Args:
            scaler_path: Scaler file path identifier
            
        Returns:
            Cached scaler or None
        """
        cache_key = f"{self.PREFIX_SCALER}{scaler_path}"
        
        try:
            with self._lock:
                scaler = self._cache.get(cache_key)
                if scaler is not None:
                    self.logger.debug(f"Retrieved scaler {scaler_path} from memory cache")
                return scaler
        except Exception as e:
            self.logger.error(f"Failed to get cached scaler: {e}")
            return None
    
    def set_cached_scaler(self, scaler_path: str, scaler_data: Any) -> bool:
        """
        Cache scaler by path.
        
        Args:
            scaler_path: Scaler file path identifier
            scaler_data: Scaler data to cache
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = f"{self.PREFIX_SCALER}{scaler_path}"
        
        try:
            with self._lock:
                self._cache[cache_key] = scaler_data
                self.logger.debug(f"Cached scaler {scaler_path}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to cache scaler: {e}")
            return False
    
    def invalidate_cached_model(self, model_path: str) -> bool:
        """
        Invalidate a specific cached model.
        
        Args:
            model_path: Path to the model file in MinIO
            
        Returns:
            True if successfully invalidated, False otherwise
        """
        cache_key = f"{self.PREFIX_MODEL}{model_path}"
        
        try:
            with self._lock:
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    self.logger.info(f"Invalidated cached model: {model_path}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to invalidate cached model: {e}")
            return False
    
    def invalidate_cached_scaler(self, scaler_path: str) -> bool:
        """
        Invalidate a specific cached scaler.
        
        Args:
            scaler_path: Path to the scaler file in MinIO
            
        Returns:
            True if successfully invalidated, False otherwise
        """
        cache_key = f"{self.PREFIX_SCALER}{scaler_path}"
        
        try:
            with self._lock:
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    self.logger.info(f"Invalidated cached scaler: {scaler_path}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to invalidate cached scaler: {e}")
            return False
    
    def invalidate_cached_config(self, config_version: str) -> bool:
        """
        Invalidate a specific cached config.
        
        Args:
            config_version: Version of the config to invalidate
            
        Returns:
            True if successfully invalidated, False otherwise
        """
        cache_key = f"{self.PREFIX_CONFIG}{config_version}"
        
        try:
            with self._lock:
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    self.logger.info(f"Invalidated cached config: {config_version}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to invalidate cached config: {e}")
            return False
    
    def check_version_and_invalidate_if_needed(self, current_version: str) -> bool:
        """
        Check if version has changed and invalidate cache if needed.
        
        Args:
            current_version: Current strategy configuration version
            
        Returns:
            True if cache was invalidated, False otherwise
        """
        version_key = f"{self.PREFIX_VERSION}current"
        
        try:
            with self._lock:
                cached_version = self._cache.get(version_key)
                
                if cached_version != current_version:
                    self.logger.info(f"Version changed from {cached_version} to {current_version}, invalidating cache...")
                    
                    # Clear all strategy-related caches
                    cleared_counts = self.clear_all_caches()
                    
                    # Set new version
                    self._cache[version_key] = current_version
                    
                    # Log detailed breakdown of what was cleared
                    total_cleared = sum(cleared_counts.values())
                    self.logger.info(f"Cache invalidated due to version change. Cleared {total_cleared} items:")
                    for cache_type, count in cleared_counts.items():
                        if count > 0:
                            self.logger.info(f"  - {cache_type}: {count} items")
                    
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking version: {e}")
            return False
    
    def get_current_cached_version(self) -> Optional[str]:
        """Get the currently cached strategy version."""
        version_key = f"{self.PREFIX_VERSION}current"
        try:
            with self._lock:
                return self._cache.get(version_key)
        except Exception as e:
            self.logger.error(f"Error getting cached version: {e}")
            return None
    
    def clear_all_caches(self) -> Dict[str, int]:
        """
        Clear all strategy-related caches.
        
        Returns:
            Dictionary with count of cleared items by type
        """
        cleared = {}
        
        try:
            with self._lock:
                # Count and clear different cache types
                keys_to_remove = []
                
                for key in self._cache.keys():
                    if key.startswith(self.PREFIX_CONFIG):
                        keys_to_remove.append(key)
                        cleared['configs'] = cleared.get('configs', 0) + 1
                    elif key.startswith(self.PREFIX_TIMESTAMP):
                        keys_to_remove.append(key)
                        cleared['timestamps'] = cleared.get('timestamps', 0) + 1
                    elif key.startswith(self.PREFIX_MODEL):
                        keys_to_remove.append(key)
                        cleared['models'] = cleared.get('models', 0) + 1
                    elif key.startswith(self.PREFIX_SCALER):
                        keys_to_remove.append(key)
                        cleared['scalers'] = cleared.get('scalers', 0) + 1
                
                # Remove the keys
                for key in keys_to_remove:
                    del self._cache[key]
                
                total_cleared = sum(cleared.values())
                self.logger.info(f"Cleared {total_cleared} cache items")
                
        except Exception as e:
            self.logger.error(f"Failed to clear caches: {e}")
        
        return cleared
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            with self._lock:
                # Get current cached version
                current_version = self.get_current_cached_version()
                
                # Get last run timestamp
                timestamp_key = f"{self.PREFIX_TIMESTAMP}last_run"
                cached_timestamp = self._cache.get(timestamp_key)
                
                # Count keys by prefix
                key_counts = {
                    'configs': {'active_items': 0, 'expired_items': 0},
                    'timestamps': {'active_items': 0, 'expired_items': 0},
                    'models': {'active_items': 0, 'expired_items': 0},
                    'scalers': {'active_items': 0, 'expired_items': 0},
                    'versions': {'active_items': 0, 'expired_items': 0}
                }
                
                for key in self._cache.keys():
                    if key.startswith(self.PREFIX_CONFIG):
                        key_counts['configs']['active_items'] += 1
                    elif key.startswith(self.PREFIX_TIMESTAMP):
                        key_counts['timestamps']['active_items'] += 1
                    elif key.startswith(self.PREFIX_MODEL):
                        key_counts['models']['active_items'] += 1
                    elif key.startswith(self.PREFIX_SCALER):
                        key_counts['scalers']['active_items'] += 1
                    elif key.startswith(self.PREFIX_VERSION):
                        key_counts['versions']['active_items'] += 1
                
                # Calculate total memory usage (rough estimate)
                total_items = len(self._cache)
                
                # Format memory stats
                memory_stats = {
                    'active_items': f"{total_items} items",
                    'expired_items': 0  # No TTL in memory cache
                }
                
                # Format strategy cache counts
                strategy_counts = {
                    'active_items': sum(counts['active_items'] for counts in key_counts.values()),
                    'expired_items': 0  # No TTL in memory cache
                }
                
                return {
                    'current_config_version': current_version,
                    'cached_last_run_timestamp': str(cached_timestamp) if cached_timestamp else None,
                    'memory_stats': memory_stats,
                    'cache_counts': strategy_counts,
                    'cache_details': key_counts,
                    'cache_efficiency': 100.0  # Always 100% for in-memory cache
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {
                'current_config_version': None,
                'cached_last_run_timestamp': None,
                'memory_stats': {'active_items': 0, 'expired_items': 0},
                'cache_counts': {'active_items': 0, 'expired_items': 0},
                'cache_details': {},
                'cache_efficiency': 0.0
            }

    def set_lag_offset_bounds(self, bounds: Dict[str, Any]) -> None:
        """
        Set the lag offset bounds for the cache.

        Args:
            bounds: A dictionary containing the lag offset bounds.
        """
        try:
            with self._lock:
                self._cache[self.PREFIX_LAG_OFFSET_BOUNDS] = bounds
        except Exception as e:
            self.logger.error(f"Failed to set lag offset bounds: {e}")

    def get_lag_offset_bounds(self) -> Optional[Dict[str, Any]]:
        """
        Get the lag offset bounds from the cache.

        Returns:
            A dictionary containing the lag offset bounds, or None if not set.
        """
        try:
            with self._lock:
                return self._cache.get(self.PREFIX_LAG_OFFSET_BOUNDS)
        except Exception as e:
            self.logger.error(f"Failed to get lag offset bounds: {e}")
            return None

# Global cache instance
_cache: Optional[InMemoryCache] = None


def get_cache() -> InMemoryCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = InMemoryCache()
    return _cache
