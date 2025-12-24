"""
Middleware for the ACE API including rate limiting and error handling.
"""
import time
from collections import defaultdict
from typing import Callable
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware

from errors import ACEException, sanitize_error_message, log_error
from utils import logger


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using token bucket algorithm."""
    
    def __init__(self, app, requests_per_minute: int = 60, burst_size: int = 10):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens = defaultdict(lambda: burst_size)
        self.last_update = defaultdict(lambda: time.time())
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Skip rate limiting for health checks
        if request.url.path in ["/", "/health", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        client_id = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Refill tokens based on time passed
        time_passed = current_time - self.last_update[client_id]
        tokens_to_add = (time_passed / 60.0) * self.requests_per_minute
        self.tokens[client_id] = min(
            self.burst_size,
            self.tokens[client_id] + tokens_to_add
        )
        self.last_update[client_id] = current_time
        
        # Check if request is allowed
        if self.tokens[client_id] < 1:
            logger.warning(f"Rate limit exceeded for {client_id}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed"
                }
            )
        
        # Consume token
        self.tokens[client_id] -= 1
        
        return await call_next(request)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware to catch and sanitize all exceptions."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        try:
            return await call_next(request)
        except ACEException as e:
            # Custom ACE exceptions - already sanitized
            log_error(logger, e, {"path": request.url.path, "method": request.method})
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=sanitize_error_message(e, include_details=False)
            )
        except HTTPException:
            # Re-raise HTTP exceptions (they're already properly formatted)
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            log_error(logger, e, {"path": request.url.path, "method": request.method})
            error_response = sanitize_error_message(e, include_details=False)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response
            )

