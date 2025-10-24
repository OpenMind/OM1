"""Distributed tracing system for OM1."""

import asyncio
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class SpanStatus(Enum):
    """Span status values."""
    OK = "OK"
    ERROR = "ERROR"
    UNSET = "UNSET"


@dataclass
class SpanContext:
    """Context for a span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)


@dataclass
class Span:
    """A tracing span."""
    name: str
    context: SpanContext
    start_time: float
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get span duration in seconds."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    def add_attribute(self, key: str, value: Any):
        """Add an attribute to the span."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the span."""
        event = {
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        }
        self.events.append(event)
    
    def set_status(self, status: SpanStatus, message: Optional[str] = None):
        """Set the span status."""
        self.status = status
        if message:
            self.error_message = message
    
    def finish(self):
        """Finish the span."""
        self.end_time = time.time()


class Tracer:
    """Distributed tracer."""
    
    def __init__(self, service_name: str = "om1"):
        self.service_name = service_name
        self.spans: List[Span] = []
        self._logger = logging.getLogger("tracer")
        self._lock = asyncio.Lock()
    
    def start_span(self, name: str, parent_context: Optional[SpanContext] = None) -> Span:
        """
        Start a new span.
        
        Parameters
        ----------
        name : str
            Name of the span
        parent_context : Optional[SpanContext]
            Parent span context for creating child spans
            
        Returns
        -------
        Span
            The started span
        """
        if parent_context:
            context = SpanContext(
                trace_id=parent_context.trace_id,
                span_id=self._generate_span_id(),
                parent_span_id=parent_context.span_id,
                baggage=parent_context.baggage.copy()
            )
        else:
            context = SpanContext(
                trace_id=self._generate_trace_id(),
                span_id=self._generate_span_id()
            )
        
        span = Span(
            name=name,
            context=context,
            start_time=time.time()
        )
        
        return span
    
    def finish_span(self, span: Span):
        """Finish a span."""
        span.finish()
        asyncio.create_task(self._record_span(span))
    
    async def _record_span(self, span: Span):
        """Record a completed span."""
        async with self._lock:
            self.spans.append(span)
            self._logger.debug(f"Recorded span: {span.name} ({span.duration:.3f}s)")
    
    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return str(uuid.uuid4())
    
    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return str(uuid.uuid4())[:16]
    
    def get_spans_by_trace_id(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace ID."""
        return [span for span in self.spans if span.context.trace_id == trace_id]
    
    def get_recent_spans(self, limit: int = 100) -> List[Span]:
        """Get recent spans."""
        return self.spans[-limit:]
    
    def clear_spans(self):
        """Clear all recorded spans."""
        self.spans.clear()


class TraceContext:
    """Context manager for tracing operations."""
    
    def __init__(self, tracer: Tracer, name: str, parent_context: Optional[SpanContext] = None):
        self.tracer = tracer
        self.name = name
        self.parent_context = parent_context
        self.span: Optional[Span] = None
    
    def __enter__(self) -> Span:
        self.span = self.tracer.start_span(self.name, self.parent_context)
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type is not None:
                self.span.set_status(SpanStatus.ERROR, str(exc_val))
            else:
                self.span.set_status(SpanStatus.OK)
            
            self.tracer.finish_span(self.span)


# Global tracer instance
tracer = Tracer()


def trace_operation(name: str, parent_context: Optional[SpanContext] = None):
    """Decorator for tracing operations."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with TraceContext(tracer, name, parent_context) as span:
                    span.add_attribute("function", func.__name__)
                    span.add_attribute("module", func.__module__)
                    
                    try:
                        result = await func(*args, **kwargs)
                        span.add_attribute("success", True)
                        return result
                    except Exception as e:
                        span.set_status(SpanStatus.ERROR, str(e))
                        raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with TraceContext(tracer, name, parent_context) as span:
                    span.add_attribute("function", func.__name__)
                    span.add_attribute("module", func.__module__)
                    
                    try:
                        result = func(*args, **kwargs)
                        span.add_attribute("success", True)
                        return result
                    except Exception as e:
                        span.set_status(SpanStatus.ERROR, str(e))
                        raise
            return sync_wrapper
    return decorator


class TraceableLLM:
    """Wrapper for LLM calls with tracing."""
    
    def __init__(self, llm_client, name: str = "LLM"):
        self.llm_client = llm_client
        self.name = name
        self.tracer = tracer
    
    async def ask(self, prompt: str, parent_context: Optional[SpanContext] = None):
        """Ask the LLM with tracing."""
        with TraceContext(self.tracer, f"{self.name}.ask", parent_context) as span:
            span.add_attribute("llm.name", self.name)
            span.add_attribute("prompt.length", len(prompt))
            
            start_time = time.time()
            try:
                result = await self.llm_client.ask(prompt)
                duration = time.time() - start_time
                
                span.add_attribute("response.length", len(str(result)) if result else 0)
                span.add_attribute("duration", duration)
                span.add_attribute("success", True)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                span.add_attribute("duration", duration)
                span.add_attribute("success", False)
                span.set_status(SpanStatus.ERROR, str(e))
                raise


class TraceableAction:
    """Wrapper for actions with tracing."""
    
    def __init__(self, action, name: str = "Action"):
        self.action = action
        self.name = name
        self.tracer = tracer
    
    async def execute(self, action_data, parent_context: Optional[SpanContext] = None):
        """Execute action with tracing."""
        with TraceContext(self.tracer, f"{self.name}.execute", parent_context) as span:
            span.add_attribute("action.name", self.name)
            span.add_attribute("action.type", type(self.action).__name__)
            
            start_time = time.time()
            try:
                result = await self.action.execute(action_data)
                duration = time.time() - start_time
                
                span.add_attribute("duration", duration)
                span.add_attribute("success", True)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                span.add_attribute("duration", duration)
                span.add_attribute("success", False)
                span.set_status(SpanStatus.ERROR, str(e))
                raise
