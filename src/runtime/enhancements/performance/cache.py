"""Advanced caching system for OM1 performance optimization."""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import pickle


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "LRU"  # Least Recently Used
    LFU = "LFU"  # Least Frequently Used
    TTL = "TTL"  # Time To Live
    SIZE = "SIZE"  # Size-based eviction


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class Cache:
    """Base cache implementation."""
    
    def __init__(self, max_size: int = 1000, policy: CachePolicy = CachePolicy.LRU):
        self.max_size = max_size
        self.policy = policy
        self.entries: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger("cache")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        async with self._lock:
            if key not in self.entries:
                return None
            
            entry = self.entries[key]
            
            # Check if expired
            if entry.is_expired():
                del self.entries[key]
                return None
            
            # Update access statistics
            entry.update_access()
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set a value in the cache."""
        async with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except (pickle.PickleError, TypeError):
                size_bytes = len(str(value))
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            # Evict if necessary
            if len(self.entries) >= self.max_size:
                await self._evict()
            
            self.entries[key] = entry
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        async with self._lock:
            if key in self.entries:
                del self.entries[key]
                return True
            return False
    
    async def clear(self):
        """Clear all entries from the cache."""
        async with self._lock:
            self.entries.clear()
    
    async def _evict(self):
        """Evict entries based on the policy."""
        if not self.entries:
            return
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently used
            oldest_key = min(self.entries.keys(), key=lambda k: self.entries[k].last_accessed)
            del self.entries[oldest_key]
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            least_used_key = min(self.entries.keys(), key=lambda k: self.entries[k].access_count)
            del self.entries[least_used_key]
        elif self.policy == CachePolicy.SIZE:
            # Remove largest entries
            largest_key = max(self.entries.keys(), key=lambda k: self.entries[k].size_bytes)
            del self.entries[largest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.entries:
            return {
                "size": 0,
                "max_size": self.max_size,
                "hit_rate": 0.0,
                "total_size_bytes": 0
            }
        
        total_accesses = sum(entry.access_count for entry in self.entries.values())
        total_size = sum(entry.size_bytes for entry in self.entries.values())
        
        return {
            "size": len(self.entries),
            "max_size": self.max_size,
            "total_accesses": total_accesses,
            "total_size_bytes": total_size,
            "average_access_count": total_accesses / len(self.entries) if self.entries else 0
        }


class LLMCache:
    """Specialized cache for LLM calls."""
    
    def __init__(self, max_size: int = 100, ttl: float = 3600):
        self.cache = Cache(max_size=max_size, policy=CachePolicy.LRU)
        self.ttl = ttl
        self._logger = logging.getLogger("llm_cache")
    
    def _generate_key(self, prompt: str, llm_config: Dict[str, Any]) -> str:
        """Generate a cache key for an LLM call."""
        # Create a hash of the prompt and configuration
        key_data = {
            "prompt": prompt,
            "llm_type": llm_config.get("type", "unknown"),
            "model": llm_config.get("config", {}).get("model", "unknown"),
            "temperature": llm_config.get("config", {}).get("temperature", 1.0),
            "max_tokens": llm_config.get("config", {}).get("max_tokens", 1000)
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def get_cached_response(self, prompt: str, llm_config: Dict[str, Any]) -> Optional[Any]:
        """Get a cached LLM response."""
        key = self._generate_key(prompt, llm_config)
        return await self.cache.get(key)
    
    async def cache_response(self, prompt: str, llm_config: Dict[str, Any], response: Any):
        """Cache an LLM response."""
        key = self._generate_key(prompt, llm_config)
        await self.cache.set(key, response, ttl=self.ttl)
        self._logger.debug(f"Cached LLM response for prompt hash: {key[:8]}...")
    
    async def clear(self):
        """Clear the LLM cache."""
        await self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM cache statistics."""
        return self.cache.get_stats()


class SensorDataCache:
    """Cache for sensor data with time-based expiration."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 30.0):
        self.cache = Cache(max_size=max_size, policy=CachePolicy.TTL)
        self.default_ttl = default_ttl
        self._logger = logging.getLogger("sensor_cache")
    
    async def get_latest_data(self, sensor_type: str) -> Optional[Any]:
        """Get the latest data for a sensor type."""
        return await self.cache.get(f"sensor_{sensor_type}")
    
    async def cache_data(self, sensor_type: str, data: Any, ttl: Optional[float] = None):
        """Cache sensor data."""
        key = f"sensor_{sensor_type}"
        ttl = ttl or self.default_ttl
        await self.cache.set(key, data, ttl=ttl)
        self._logger.debug(f"Cached {sensor_type} data")
    
    async def clear_sensor(self, sensor_type: str):
        """Clear data for a specific sensor."""
        await self.cache.delete(f"sensor_{sensor_type}")
    
    async def clear_all(self):
        """Clear all sensor data."""
        await self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sensor cache statistics."""
        return self.cache.get_stats()


class BatchProcessor:
    """Batch processor for efficient operation batching."""
    
    def __init__(self, batch_size: int = 10, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.pending_operations: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._logger = logging.getLogger("batch_processor")
    
    async def start(self):
        """Start the batch processor."""
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush_loop())
            self._logger.info("Started batch processor")
    
    async def stop(self):
        """Stop the batch processor and flush remaining operations."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        
        # Flush remaining operations
        await self._flush_pending()
        self._logger.info("Stopped batch processor")
    
    async def add_operation(self, operation: Dict[str, Any]):
        """Add an operation to the batch."""
        async with self._lock:
            self.pending_operations.append(operation)
            
            if len(self.pending_operations) >= self.batch_size:
                await self._flush_pending()
    
    async def _flush_loop(self):
        """Main flush loop."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_pending()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in flush loop: {e}")
    
    async def _flush_pending(self):
        """Flush pending operations."""
        async with self._lock:
            if not self.pending_operations:
                return
            
            operations = self.pending_operations.copy()
            self.pending_operations.clear()
        
        if operations:
            await self._process_batch(operations)
    
    async def _process_batch(self, operations: List[Dict[str, Any]]):
        """Process a batch of operations."""
        self._logger.debug(f"Processing batch of {len(operations)} operations")
        
        # Group operations by type
        grouped_ops = {}
        for op in operations:
            op_type = op.get("type", "unknown")
            if op_type not in grouped_ops:
                grouped_ops[op_type] = []
            grouped_ops[op_type].append(op)
        
        # Process each group
        for op_type, ops in grouped_ops.items():
            try:
                await self._process_operation_group(op_type, ops)
            except Exception as e:
                self._logger.error(f"Error processing {op_type} operations: {e}")
    
    async def _process_operation_group(self, op_type: str, operations: List[Dict[str, Any]]):
        """Process a group of operations of the same type."""
        # This is a placeholder - implement specific batch processing logic
        # based on operation types
        self._logger.debug(f"Processed {len(operations)} {op_type} operations")


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.llm_cache = LLMCache()
        self.sensor_cache = SensorDataCache()
        self.batch_processor = BatchProcessor()
        self._logger = logging.getLogger("performance_optimizer")
    
    async def start(self):
        """Start all optimization components."""
        await self.batch_processor.start()
        self._logger.info("Started performance optimizer")
    
    async def stop(self):
        """Stop all optimization components."""
        await self.batch_processor.stop()
        self._logger.info("Stopped performance optimizer")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics for all optimization components."""
        return {
            "llm_cache": self.llm_cache.get_stats(),
            "sensor_cache": self.sensor_cache.get_stats(),
            "batch_processor": {
                "pending_operations": len(self.batch_processor.pending_operations)
            }
        }


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()
