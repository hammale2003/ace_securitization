"""
Custom exceptions and error handling utilities for the ACE system.
"""
from typing import Optional, Dict, Any


class ACEException(Exception):
    """Base exception for ACE system errors."""
    def __init__(self, message: str, error_code: str = "INTERNAL_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(ACEException):
    """Raised when configuration is invalid."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONFIG_ERROR", details)


class LLMError(ACEException):
    """Raised when LLM API calls fail."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "LLM_ERROR", details)


class PlaybookError(ACEException):
    """Raised when playbook operations fail."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "PLAYBOOK_ERROR", details)


class ValidationError(ACEException):
    """Raised when input validation fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)


def sanitize_error_message(error: Exception, include_details: bool = False) -> Dict[str, Any]:
    """
    Sanitize error messages to prevent leaking internal details.
    
    Args:
        error: The exception to sanitize
        include_details: Whether to include detailed error info (for internal logging only)
    
    Returns:
        Dict with sanitized error information
    """
    # User-friendly error messages
    user_messages = {
        "ValueError": "Invalid input provided",
        "KeyError": "Required information is missing",
        "FileNotFoundError": "Required file not found",
        "PermissionError": "Insufficient permissions",
        "ConnectionError": "Unable to connect to external service",
        "TimeoutError": "Operation timed out",
        "JSONDecodeError": "Invalid data format",
    }
    
    error_type = type(error).__name__
    user_message = user_messages.get(error_type, "An error occurred processing your request")
    
    response = {
        "error": user_message,
        "error_code": getattr(error, 'error_code', 'INTERNAL_ERROR')
    }
    
    # Only include details in development/debugging
    if include_details:
        response["details"] = {
            "type": error_type,
            "message": str(error)
        }
    
    return response


def log_error(logger, error: Exception, context: Optional[Dict[str, Any]] = None):
    """
    Log error with full context for debugging.
    
    Args:
        logger: Logger instance
        error: The exception
        context: Additional context to log
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    log_data = {
        "error_type": error_type,
        "error_message": error_msg,
        "error_code": getattr(error, 'error_code', 'UNKNOWN')
    }
    
    if context:
        log_data.update(context)
    
    logger.error(f"Error occurred: {log_data}", exc_info=True)

