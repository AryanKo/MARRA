import logging
from functools import wraps
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

logger = logging.getLogger(__name__)

def setup_observability(app=None):
    try:
        # Configure OpenTelemetry to send traces to the Phoenix Docker container
        import os
        endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:4318/v1/traces")
        tracer_provider = TracerProvider()
        
        # Use SimpleSpanProcessor for dev/local environments to immediately flush spans
        span_processor = SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        tracer_provider.add_span_processor(span_processor)
        
        trace.set_tracer_provider(tracer_provider)
        
        # Instrument LangChain/LangGraph
        LangChainInstrumentor().instrument()
        
        # Instrument FastAPI
        if app:
            FastAPIInstrumentor.instrument_app(app)
            
        logger.info("Observability enabled: Sending OTLP traces to Phoenix at http://127.0.0.1:4318/v1/traces")
    except Exception as e:
        logger.error(f"Failed to setup observability: {e}")

def trace_span(name: str):
    """
    A decorator to explicitly create an OpenTelemetry span around a function.
    Usage: @trace_span(name="my_span_name")
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(func.__module__)
            with tracer.start_as_current_span(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
