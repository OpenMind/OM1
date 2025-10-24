"""Metrics collection and monitoring system for OM1."""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from threading import Lock


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A metric value with metadata."""
    name: str
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class HistogramBucket:
    """A histogram bucket."""
    upper_bound: float
    count: int


class Counter:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = Lock()
    
    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        with self._lock:
            self._value += value
    
    def get_value(self) -> float:
        with self._lock:
            return self._value


class Gauge:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = Lock()
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        with self._lock:
            self._value = value
    
    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        with self._lock:
            self._value += value
    
    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        with self._lock:
            self._value -= value
    
    def get_value(self) -> float:
        with self._lock:
            return self._value


class Histogram:
    def __init__(self, name: str, description: str = "", buckets: Optional[List[float]] = None):
        self.name = name
        self.description = description
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf')]
        self._observations = deque(maxlen=10000)
        self._lock = Lock()
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        with self._lock:
            self._observations.append(value)
    
    def get_buckets(self) -> List[HistogramBucket]:
        with self._lock:
            observations = list(self._observations)
        
        buckets = []
        for bucket in self.buckets:
            count = sum(1 for obs in observations if obs <= bucket)
            buckets.append(HistogramBucket(upper_bound=bucket, count=count))
        
        return buckets
    
    def get_sum(self) -> float:
        with self._lock:
            return sum(self._observations)
    
    def get_count(self) -> int:
        with self._lock:
            return len(self._observations)


class MetricsCollector:
    def __init__(self):
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self._logger = logging.getLogger("metrics_collector")
        self._lock = Lock()
    
    def create_counter(self, name: str, description: str = "") -> Counter:
        with self._lock:
            if name not in self.counters:
                self.counters[name] = Counter(name, description)
                self._logger.debug(f"Created counter metric: {name}")
            return self.counters[name]
    
    def create_gauge(self, name: str, description: str = "") -> Gauge:
        with self._lock:
            if name not in self.gauges:
                self.gauges[name] = Gauge(name, description)
                self._logger.debug(f"Created gauge metric: {name}")
            return self.gauges[name]
    
    def create_histogram(self, name: str, description: str = "", buckets: Optional[List[float]] = None) -> Histogram:
        with self._lock:
            if name not in self.histograms:
                self.histograms[name] = Histogram(name, description, buckets)
                self._logger.debug(f"Created histogram metric: {name}")
            return self.histograms[name]
    
    def get_counter(self, name: str) -> Optional[Counter]:
        return self.counters.get(name)
    
    def get_gauge(self, name: str) -> Optional[Gauge]:
        return self.gauges.get(name)
    
    def get_histogram(self, name: str) -> Optional[Histogram]:
        return self.histograms.get(name)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        metrics = {}
        
        with self._lock:
            for name, counter in self.counters.items():
                metrics[f"counter_{name}"] = {
                    "type": "counter",
                    "value": counter.get_value(),
                    "description": counter.description
                }
            
            for name, gauge in self.gauges.items():
                metrics[f"gauge_{name}"] = {
                    "type": "gauge",
                    "value": gauge.get_value(),
                    "description": gauge.description
                }
            
            for name, histogram in self.histograms.items():
                metrics[f"histogram_{name}"] = {
                    "type": "histogram",
                    "buckets": [{"upper_bound": b.upper_bound, "count": b.count} for b in histogram.get_buckets()],
                    "sum": histogram.get_sum(),
                    "count": histogram.get_count(),
                    "description": histogram.description
                }
        
        return metrics


class PerformanceProfiler:
    """Performance profiler for measuring execution times."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._logger = logging.getLogger("performance_profiler")
        
        # Create performance metrics
        self.execution_time_histogram = self.metrics.create_histogram(
            "execution_time_seconds",
            "Execution time in seconds",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
        )
        
        self.llm_call_duration = self.metrics.create_histogram(
            "llm_call_duration_seconds",
            "LLM call duration in seconds",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float('inf')]
        )
        
        self.action_execution_time = self.metrics.create_histogram(
            "action_execution_time_seconds",
            "Action execution time in seconds",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, float('inf')]
        )
        
        self.memory_usage = self.metrics.create_gauge(
            "memory_usage_bytes",
            "Memory usage in bytes"
        )
        
        self.cpu_usage = self.metrics.create_gauge(
            "cpu_usage_percent",
            "CPU usage percentage"
        )
    
    def time_execution(self, operation_name: str):
        """Context manager for timing operations."""
        return ExecutionTimer(self.metrics, operation_name)
    
    def record_llm_call(self, duration: float, llm_name: str = "unknown"):
        """Record LLM call duration."""
        self.llm_call_duration.observe(duration)
        self._logger.debug(f"LLM call to {llm_name} took {duration:.3f}s")
    
    def record_action_execution(self, duration: float, action_name: str = "unknown"):
        """Record action execution time."""
        self.action_execution_time.observe(duration)
        self._logger.debug(f"Action {action_name} took {duration:.3f}s")
    
    async def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            import psutil
            
            # Memory usage
            memory_info = psutil.virtual_memory()
            self.memory_usage.set(memory_info.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
        except ImportError:
            self._logger.warning("psutil not available, system metrics disabled")
        except Exception as e:
            self._logger.error(f"Failed to update system metrics: {e}")


class ExecutionTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, metrics: MetricsCollector, operation_name: str):
        self.metrics = metrics
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics.execution_time_histogram.observe(duration)


# Global metrics collector instance
metrics_collector = MetricsCollector()
performance_profiler = PerformanceProfiler(metrics_collector)
