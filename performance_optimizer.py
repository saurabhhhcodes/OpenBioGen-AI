"""
Performance Optimization Module for OpenBioGen AI
Provides caching, parallel processing, and performance monitoring
"""

import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Callable
import pickle
import hashlib
import os
from datetime import datetime, timedelta
import json

class SmartCache:
    """Advanced caching system with TTL and intelligent invalidation"""
    
    def __init__(self, cache_dir: str = "cache", default_ttl: int = 3600):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "size": 0}
    
    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate unique cache key"""
        key_data = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get file path for cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.cache")
    
    def _is_cache_valid(self, cache_path: str, ttl: int) -> bool:
        """Check if cache file is still valid"""
        if not os.path.exists(cache_path):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - file_time < timedelta(seconds=ttl)
    
    def get(self, func_name: str, args: tuple, kwargs: dict, ttl: int = None) -> Optional[Any]:
        """Get cached result"""
        cache_key = self._get_cache_key(func_name, args, kwargs)
        ttl = ttl or self.default_ttl
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            cached_data, timestamp = self.memory_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=ttl):
                self.cache_stats["hits"] += 1
                return cached_data
            else:
                del self.memory_cache[cache_key]
        
        # Check file cache
        cache_path = self._get_cache_path(cache_key)
        if self._is_cache_valid(cache_path, ttl):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.memory_cache[cache_key] = (cached_data, datetime.now())
                    self.cache_stats["hits"] += 1
                    return cached_data
            except Exception:
                pass
        
        self.cache_stats["misses"] += 1
        return None
    
    def set(self, func_name: str, args: tuple, kwargs: dict, result: Any):
        """Cache result"""
        cache_key = self._get_cache_key(func_name, args, kwargs)
        
        # Store in memory cache
        self.memory_cache[cache_key] = (result, datetime.now())
        
        # Store in file cache
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            self.cache_stats["size"] += 1
        except Exception:
            pass
    
    def clear_expired(self):
        """Clear expired cache entries"""
        current_time = datetime.now()
        
        # Clear memory cache
        expired_keys = []
        for key, (_, timestamp) in self.memory_cache.items():
            if current_time - timestamp > timedelta(seconds=self.default_ttl):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        # Clear file cache
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.cache'):
                filepath = os.path.join(self.cache_dir, filename)
                if not self._is_cache_valid(filepath, self.default_ttl):
                    try:
                        os.remove(filepath)
                        self.cache_stats["size"] -= 1
                    except Exception:
                        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = self.cache_stats["hits"] / (self.cache_stats["hits"] + self.cache_stats["misses"]) if (self.cache_stats["hits"] + self.cache_stats["misses"]) > 0 else 0
        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache)
        }

# Global cache instance
smart_cache = SmartCache()

def cached(ttl: int = 3600):
    """Decorator for intelligent caching"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get from cache
            cached_result = smart_cache.get(func.__name__, args, kwargs, ttl)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            smart_cache.set(func.__name__, args, kwargs, result)
            return result
        return wrapper
    return decorator

class ParallelProcessor:
    """Advanced parallel processing for bioinformatics tasks"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min(self.max_workers, os.cpu_count() or 1))
    
    def parallel_map(self, func: Callable, items: List[Any], use_processes: bool = False) -> List[Any]:
        """Execute function on list of items in parallel"""
        executor = self.process_executor if use_processes else self.thread_executor
        
        try:
            futures = [executor.submit(func, item) for item in items]
            results = [future.result() for future in futures]
            return results
        except Exception as e:
            print(f"Parallel processing error: {e}")
            # Fallback to sequential processing
            return [func(item) for item in items]
    
    def parallel_batch_predict(self, predict_func: Callable, gene_disease_pairs: List[tuple], batch_size: int = 10) -> List[Dict[str, Any]]:
        """Process gene-disease predictions in parallel batches"""
        def process_batch(batch):
            return [predict_func(gene, disease) for gene, disease in batch]
        
        # Split into batches
        batches = [gene_disease_pairs[i:i + batch_size] for i in range(0, len(gene_disease_pairs), batch_size)]
        
        # Process batches in parallel
        batch_results = self.parallel_map(process_batch, batches)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results
    
    def cleanup(self):
        """Clean up executors"""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)

class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.metrics[operation] = {"start_time": time.time()}
    
    def end_timer(self, operation: str, **metadata):
        """End timing an operation"""
        if operation in self.metrics:
            end_time = time.time()
            duration = end_time - self.metrics[operation]["start_time"]
            self.metrics[operation].update({
                "duration": duration,
                "end_time": end_time,
                **metadata
            })
            return duration
        return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        total_runtime = time.time() - self.start_time
        
        report = {
            "total_runtime": total_runtime,
            "operations": {},
            "summary": {
                "total_operations": len(self.metrics),
                "average_operation_time": 0,
                "slowest_operation": None,
                "fastest_operation": None
            }
        }
        
        if self.metrics:
            durations = []
            for op_name, op_data in self.metrics.items():
                if "duration" in op_data:
                    duration = op_data["duration"]
                    durations.append(duration)
                    report["operations"][op_name] = op_data
            
            if durations:
                report["summary"]["average_operation_time"] = sum(durations) / len(durations)
                report["summary"]["slowest_operation"] = max(self.metrics.items(), key=lambda x: x[1].get("duration", 0))
                report["summary"]["fastest_operation"] = min(self.metrics.items(), key=lambda x: x[1].get("duration", float('inf')))
        
        return report
    
    def save_report(self, filename: str = None):
        """Save performance report to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        os.makedirs("reports", exist_ok=True)
        filepath = os.path.join("reports", filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.get_performance_report(), f, indent=2, default=str)
        
        return filepath

# Global instances
parallel_processor = ParallelProcessor()
performance_monitor = PerformanceMonitor()

def monitor_performance(operation_name: str):
    """Decorator to monitor function performance"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            performance_monitor.start_timer(operation_name)
            try:
                result = func(*args, **kwargs)
                performance_monitor.end_timer(operation_name, success=True)
                return result
            except Exception as e:
                performance_monitor.end_timer(operation_name, success=False, error=str(e))
                raise
        return wrapper
    return decorator
