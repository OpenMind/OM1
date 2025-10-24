#!/usr/bin/env python3
"""Enhanced Runtime Example"""

import asyncio
import logging

from .enhanced_cortex import EnhancedCortexRuntime
from ...single_mode.config import RuntimeConfig
from .resilience import CircuitBreaker, health_monitor
from .observability import metrics_collector, tracer
from .safety import safety_manager
from .performance import performance_optimizer


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("enhanced_runtime_example")
    logger.info("Starting Enhanced Runtime Example")
    
    try:
        config = RuntimeConfig.load("spot")
        runtime = EnhancedCortexRuntime(config)
        
        logger.info("Starting enhanced runtime...")
        await runtime.start()
        
        health_status = runtime.get_health_status()
        logger.info(f"Health Status: {health_status}")
        
        metrics = runtime.get_metrics()
        logger.info(f"Runtime Metrics: {metrics['runtime_metrics']}")
        
        test_actions = [
            "walk forward slowly",
            "run at high speed", 
            "jump up and down",
            "backflip",
            "pick up the red ball"
        ]
        
        for action in test_actions:
            result = safety_manager.validate_action(action)
            logger.info(f"Action '{action}': {result.value}")
        
        test_inputs = [
            "Hello, how are you?",
            "Please walk forward <script>alert('xss')</script>",
            "Execute: rm -rf /",
            "Normal robot command: sit down"
        ]
        
        for input_text in test_inputs:
            sanitized = safety_manager.validate_and_sanitize_input(input_text)
            logger.info(f"Input: {input_text[:50]}...")
            logger.info(f"Safe: {sanitized['is_safe']}, Sanitized: {sanitized['sanitization_applied']}")
        
        traces = runtime.get_traces(limit=5)
        logger.info(f"Recent Traces: {len(traces['spans'])} spans")
        
        opt_stats = runtime.get_metrics()['optimization_stats']
        logger.info(f"Optimization Stats: {opt_stats}")
        
        logger.info("Running for 30 seconds to generate activity...")
        await asyncio.sleep(30)
        
        final_health = runtime.get_health_status()
        final_metrics = runtime.get_metrics()
        safety_status = safety_manager.get_safety_status()
        
        logger.info(f"Final Health: {final_health}")
        logger.info(f"Final Metrics: {final_metrics['runtime_metrics']}")
        logger.info(f"Safety Status: {safety_status}")
        
        logger.info("Shutting down...")
        await runtime.graceful_shutdown()
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise
    finally:
        logger.info("Enhanced Runtime Example completed")


async def demonstrate_circuit_breaker():
    logger = logging.getLogger("circuit_breaker_demo")
    
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0, name="Demo")
    
    async def failing_function():
        raise Exception("Simulated failure")
    
    logger.info("Testing circuit breaker...")
    
    for i in range(5):
        try:
            await cb.call(failing_function)
        except Exception as e:
            logger.info(f"Attempt {i+1}: {e}")
            logger.info(f"Circuit state: {cb.state.value}")
    
    logger.info("Waiting for circuit recovery...")
    await asyncio.sleep(6)
    
    try:
        await cb.call(failing_function)
    except Exception as e:
        logger.info(f"After recovery: {e}")


async def demonstrate_metrics():
    logger = logging.getLogger("metrics_demo")
    
    counter = metrics_collector.create_counter("demo_operations", "Demo operations counter")
    gauge = metrics_collector.create_gauge("demo_value", "Demo value gauge")
    histogram = metrics_collector.create_histogram("demo_duration", "Demo duration histogram")
    
    for i in range(10):
        counter.inc()
        gauge.set(i * 10)
        histogram.observe(i * 0.1)
        await asyncio.sleep(0.1)
    
    all_metrics = metrics_collector.get_all_metrics()
    logger.info(f"Demo Metrics: {all_metrics}")


if __name__ == "__main__":
    asyncio.run(main())
    
    print("\nAdditional Demonstrations")
    
    asyncio.run(demonstrate_circuit_breaker())
    asyncio.run(demonstrate_metrics())
