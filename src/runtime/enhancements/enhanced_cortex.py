"""Enhanced Cortex Runtime with additional features."""

import asyncio
import logging
import time
from typing import Optional, Dict, Any

from ...single_mode.cortex import CortexRuntime
from ...single_mode.config import RuntimeConfig
from .resilience.circuit_breaker import CircuitBreaker, retry_with_exponential_backoff, RetryConfig
from .resilience.health_check import HealthMonitor, LLMHealthChecker, SensorHealthChecker
from .observability.metrics import metrics_collector, performance_profiler
from .observability.tracing import tracer, trace_operation, TraceableLLM
from .performance.cache import performance_optimizer, LLMCache
from .config.validation import config_manager


class EnhancedCortexRuntime(CortexRuntime):
    
    def __init__(self, config: RuntimeConfig):
        super().__init__(config)
        
        self.health_monitor = HealthMonitor()
        self.llm_circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            name="LLM"
        )
        
        self.metrics = metrics_collector
        self.tracer = tracer
        self.profiler = performance_profiler
        
        self.performance_optimizer = performance_optimizer
        self.llm_cache = LLMCache()
        
        self._create_metrics()
        self._setup_health_checks()
        self._wrap_llm()
        
        self._logger = logging.getLogger("enhanced_cortex")
    
    def _create_metrics(self):
        self.cortex_ticks = self.metrics.create_counter(
            "cortex_ticks_total",
            "Total number of cortex ticks"
        )
        
        self.llm_calls = self.metrics.create_counter(
            "llm_calls_total",
            "Total number of LLM calls"
        )
        
        self.llm_call_duration = self.metrics.create_histogram(
            "llm_call_duration_seconds",
            "LLM call duration in seconds"
        )
        
        self.action_executions = self.metrics.create_counter(
            "action_executions_total",
            "Total number of action executions"
        )
        
        self.errors = self.metrics.create_counter(
            "errors_total",
            "Total number of errors"
        )
        
        self.active_connections = self.metrics.create_gauge(
            "active_connections",
            "Number of active connections"
        )
    
    def _setup_health_checks(self):
        if hasattr(self.config, 'cortex_llm') and self.config.cortex_llm:
            llm_checker = LLMHealthChecker(self.config.cortex_llm, "CortexLLM")
            self.health_monitor.register_checker(llm_checker)
        
        self.health_monitor.register_circuit_breaker("llm", self.llm_circuit_breaker)
        
        for input_config in getattr(self.config, 'agent_inputs', []):
            if hasattr(input_config, 'provider'):
                sensor_checker = SensorHealthChecker(
                    input_config.provider,
                    f"Sensor_{input_config.type}"
                )
                self.health_monitor.register_checker(sensor_checker)
    
    def _wrap_llm(self):
        if hasattr(self.config, 'cortex_llm') and self.config.cortex_llm:
            self.config.cortex_llm = TraceableLLM(self.config.cortex_llm, "CortexLLM")
    
    async def run(self) -> None:
        await self.performance_optimizer.start()
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._health_monitoring_loop())
        await super().run()
    
    async def _metrics_collection_loop(self):
        while True:
            try:
                await self.profiler.update_system_metrics()
                self.active_connections.set(len(self.health_monitor.checkers))
                await asyncio.sleep(10.0)
            except Exception as e:
                self._logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(30.0)
    
    async def _health_monitoring_loop(self):
        while True:
            try:
                results = await self.health_monitor.check_all()
                
                unhealthy = self.health_monitor.get_unhealthy_components()
                if unhealthy:
                    self._logger.warning(f"Unhealthy components: {unhealthy}")
                
                degraded = self.health_monitor.get_degraded_components()
                if degraded:
                    self._logger.info(f"Degraded components: {degraded}")
                
                await asyncio.sleep(30.0)
            except Exception as e:
                self._logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60.0)
    
    @trace_operation("cortex_tick")
    async def _tick(self) -> None:
        start_time = time.time()
        
        try:
            self.cortex_ticks.inc()
            
            try:
                finished_promises, _ = await self.action_orchestrator.flush_promises()
            except Exception as e:
                self._logger.error(f"Error collecting inputs: {e}")
                self.errors.inc()
                return
            
            try:
                prompt = self.fuser.fuse(self.config.agent_inputs, finished_promises)
                if prompt is None:
                    self._logger.warning("No prompt to fuse")
                    return
            except Exception as e:
                self._logger.error(f"Error fusing inputs: {e}")
                self.errors.inc()
                return
            
            cached_response = await self.llm_cache.get_cached_response(
                prompt, 
                getattr(self.config.cortex_llm, 'config', {})
            )
            
            if cached_response:
                self._logger.debug("Using cached LLM response")
                output = cached_response
            else:
                try:
                    output = await self._call_llm_with_resilience(prompt)
                    
                    await self.llm_cache.cache_response(
                        prompt,
                        getattr(self.config.cortex_llm, 'config', {}),
                        output
                    )
                except Exception as e:
                    self._logger.error(f"LLM call failed: {e}")
                    self.errors.inc()
                    return
            
            if output is None:
                self._logger.warning("No output from LLM")
                return
            
            try:
                await self.simulator_orchestrator.promise(output.actions)
                await self.action_orchestrator.promise(output.actions)
                self.action_executions.inc()
            except Exception as e:
                self._logger.error(f"Error executing actions: {e}")
                self.errors.inc()
            
        finally:
            duration = time.time() - start_time
            self.profiler.execution_time_histogram.observe(duration)
    
    async def _call_llm_with_resilience(self, prompt: str):
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=10.0
        )
        
        async def llm_call():
            start_time = time.time()
            try:
                result = await self.llm_circuit_breaker.call(
                    self.config.cortex_llm.ask,
                    prompt
                )
                
                duration = time.time() - start_time
                self.llm_calls.inc()
                self.llm_call_duration.observe(duration)
                self.profiler.record_llm_call(duration, "cortex_llm")
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                self.llm_call_duration.observe(duration)
                raise
        
        return await retry_with_exponential_backoff(llm_call, retry_config)
    
    def get_health_status(self) -> Dict[str, Any]:
        overall_status = self.health_monitor.get_overall_status()
        unhealthy = self.health_monitor.get_unhealthy_components()
        degraded = self.health_monitor.get_degraded_components()
        
        return {
            "overall_status": overall_status.value,
            "unhealthy_components": unhealthy,
            "degraded_components": degraded,
            "timestamp": time.time()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "runtime_metrics": self.metrics.get_all_metrics(),
            "optimization_stats": self.performance_optimizer.get_optimization_stats(),
            "health_status": self.get_health_status()
        }
    
    def get_traces(self, limit: int = 100) -> Dict[str, Any]:
        spans = self.tracer.get_recent_spans(limit)
        return {
            "spans": [
                {
                    "name": span.name,
                    "duration": span.duration,
                    "status": span.status.value,
                    "attributes": span.attributes,
                    "events": span.events
                }
                for span in spans
            ],
            "total_spans": len(spans)
        }
    
    async def graceful_shutdown(self):
        self._logger.info("Starting graceful shutdown...")
        await self.performance_optimizer.stop()
        self._logger.info("Enhanced runtime shutdown complete")
