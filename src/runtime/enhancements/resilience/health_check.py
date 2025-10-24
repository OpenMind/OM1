"""Health check system for monitoring component health and availability."""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from .circuit_breaker import CircuitBreaker


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    timestamp: float
    details: Optional[Dict[str, Any]] = None
    response_time: Optional[float] = None


class HealthChecker:
    def __init__(self, name: str, timeout: float = 5.0):
        self.name = name
        self.timeout = timeout
        self._logger = logging.getLogger(f"health_check.{name}")
    
    async def check(self) -> HealthCheckResult:
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self._perform_check(),
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            result.response_time = response_time
            return result
        except asyncio.TimeoutError:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                timestamp=time.time(),
                response_time=self.timeout
            )
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                timestamp=time.time(),
                response_time=time.time() - start_time
            )
    
    async def _perform_check(self) -> HealthCheckResult:
        raise NotImplementedError


class LLMHealthChecker(HealthChecker):
    def __init__(self, llm_client, name: str = "LLM", timeout: float = 10.0):
        super().__init__(name, timeout)
        self.llm_client = llm_client
    
    async def _perform_check(self) -> HealthCheckResult:
        try:
            test_prompt = "Health check"
            response = await self.llm_client.ask(test_prompt)
            
            if response and response.actions:
                return HealthCheckResult(
                    component=self.name,
                    status=HealthStatus.HEALTHY,
                    message="LLM service is responding normally",
                    timestamp=time.time(),
                    details={"response_length": len(str(response))}
                )
            else:
                return HealthCheckResult(
                    component=self.name,
                    status=HealthStatus.DEGRADED,
                    message="LLM service responded but with empty result",
                    timestamp=time.time()
                )
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"LLM service check failed: {str(e)}",
                timestamp=time.time()
            )


class SensorHealthChecker(HealthChecker):
    def __init__(self, sensor_provider, name: str = "Sensor", timeout: float = 5.0):
        super().__init__(name, timeout)
        self.sensor_provider = sensor_provider
    
    async def _perform_check(self) -> HealthCheckResult:
        try:
            if hasattr(self.sensor_provider, 'get_latest_data'):
                data = self.sensor_provider.get_latest_data()
                if data is not None:
                    return HealthCheckResult(
                        component=self.name,
                        status=HealthStatus.HEALTHY,
                        message="Sensor is providing data",
                        timestamp=time.time(),
                        details={"data_available": True}
                    )
                else:
                    return HealthCheckResult(
                        component=self.name,
                        status=HealthStatus.DEGRADED,
                        message="Sensor is not providing data",
                        timestamp=time.time()
                    )
            else:
                return HealthCheckResult(
                    component=self.name,
                    status=HealthStatus.UNKNOWN,
                    message="Sensor provider does not support health checking",
                    timestamp=time.time()
                )
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Sensor check failed: {str(e)}",
                timestamp=time.time()
            )


class HealthMonitor:
    def __init__(self):
        self.checkers: Dict[str, HealthChecker] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._logger = logging.getLogger("health_monitor")
        self._last_results: Dict[str, HealthCheckResult] = {}
    
    def register_checker(self, checker: HealthChecker):
        self.checkers[checker.name] = checker
        self._logger.info(f"Registered health checker: {checker.name}")
    
    def register_circuit_breaker(self, name: str, circuit_breaker: CircuitBreaker):
        self.circuit_breakers[name] = circuit_breaker
        self._logger.info(f"Registered circuit breaker: {name}")
    
    async def check_all(self) -> Dict[str, HealthCheckResult]:
        results = {}
        
        for name, checker in self.checkers.items():
            try:
                result = await checker.check()
                results[name] = result
                self._last_results[name] = result
            except Exception as e:
                self._logger.error(f"Health check failed for {name}: {e}")
                results[name] = HealthCheckResult(
                    component=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check error: {str(e)}",
                    timestamp=time.time()
                )
        
        for name, cb in self.circuit_breakers.items():
            status = HealthStatus.HEALTHY if cb.is_closed else HealthStatus.UNHEALTHY
            results[f"circuit_breaker_{name}"] = HealthCheckResult(
                component=f"circuit_breaker_{name}",
                status=status,
                message=f"Circuit breaker is {cb.state.value}",
                timestamp=time.time(),
                details={"failure_count": cb.failure_count}
            )
        
        return results
    
    async def check_component(self, name: str) -> Optional[HealthCheckResult]:
        if name in self.checkers:
            result = await self.checkers[name].check()
            self._last_results[name] = result
            return result
        return None
    
    def get_last_result(self, name: str) -> Optional[HealthCheckResult]:
        return self._last_results.get(name)
    
    def get_overall_status(self) -> HealthStatus:
        if not self._last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self._last_results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_unhealthy_components(self) -> List[str]:
        return [
            name for name, result in self._last_results.items()
            if result.status == HealthStatus.UNHEALTHY
        ]
    
    def get_degraded_components(self) -> List[str]:
        return [
            name for name, result in self._last_results.items()
            if result.status == HealthStatus.DEGRADED
        ]


health_monitor = HealthMonitor()
