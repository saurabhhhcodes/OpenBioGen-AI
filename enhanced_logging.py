"""
Enhanced Logging and Error Handling System for OpenBioGen AI
Provides comprehensive logging, error tracking, and performance monitoring
"""

import logging
import traceback
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional
from datetime import datetime
import json
import os

class EnhancedLogger:
    """Advanced logging system with error tracking and performance monitoring"""
    
    def __init__(self, name: str = "OpenBioGenAI"):
        self.logger = logging.getLogger(name)
        self.setup_logging()
        self.error_counts = {}
        self.performance_metrics = {}
    
    def setup_logging(self):
        """Setup comprehensive logging configuration"""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Configure logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # File handler for all logs
        file_handler = logging.FileHandler('logs/openbio_ai.log')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Error file handler
        error_handler = logging.FileHandler('logs/errors.log')
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with full context and stack trace"""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        error_info = {
            "error_type": error_type,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "stack_trace": traceback.format_exc()
        }
        
        self.logger.error(f"Error in {context}: {error_info}")
        
        # Save error details to file
        with open("logs/error_details.json", "a") as f:
            json.dump(error_info, f)
            f.write("\n")
    
    def log_performance(self, function_name: str, execution_time: float, **kwargs):
        """Log performance metrics"""
        if function_name not in self.performance_metrics:
            self.performance_metrics[function_name] = []
        
        self.performance_metrics[function_name].append({
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        })
        
        self.logger.info(f"Performance: {function_name} took {execution_time:.3f}s")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered"""
        return {
            "error_counts": self.error_counts,
            "total_errors": sum(self.error_counts.values()),
            "most_common_error": max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
        }

# Global logger instance
enhanced_logger = EnhancedLogger()

def log_performance(func: Callable) -> Callable:
    """Decorator to log function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            enhanced_logger.log_performance(func.__name__, execution_time)
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            enhanced_logger.log_error(e, f"Function: {func.__name__}")
            enhanced_logger.log_performance(func.__name__, execution_time, error=True)
            raise
    return wrapper

def safe_execute(func: Callable, *args, default_return=None, context: str = "", **kwargs):
    """Safely execute a function with comprehensive error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        enhanced_logger.log_error(e, context)
        return default_return

class HealthChecker:
    """System health monitoring"""
    
    @staticmethod
    def check_system_health() -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # Check dependencies
        try:
            import streamlit
            import pandas
            import numpy
            import plotly
            health_status["components"]["dependencies"] = "healthy"
        except ImportError as e:
            health_status["components"]["dependencies"] = f"error: {e}"
            health_status["overall_status"] = "degraded"
        
        # Check file system
        try:
            os.makedirs("temp", exist_ok=True)
            with open("temp/health_check.txt", "w") as f:
                f.write("test")
            os.remove("temp/health_check.txt")
            health_status["components"]["filesystem"] = "healthy"
        except Exception as e:
            health_status["components"]["filesystem"] = f"error: {e}"
            health_status["overall_status"] = "degraded"
        
        # Check memory usage
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            health_status["components"]["memory"] = {
                "status": "healthy" if memory_percent < 80 else "warning",
                "usage_percent": memory_percent
            }
        except ImportError:
            health_status["components"]["memory"] = "monitoring_unavailable"
        
        return health_status
