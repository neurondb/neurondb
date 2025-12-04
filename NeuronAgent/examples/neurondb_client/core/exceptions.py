"""Custom exceptions for NeuronAgent client"""


class NeuronAgentError(Exception):
    """Base exception for all NeuronAgent errors"""
    pass


class AuthenticationError(NeuronAgentError):
    """Authentication failed"""
    def __init__(self, message="Authentication failed. Check your API key."):
        self.message = message
        super().__init__(self.message)


class NotFoundError(NeuronAgentError):
    """Resource not found"""
    def __init__(self, resource_type="Resource", resource_id=None):
        msg = f"{resource_type} not found"
        if resource_id:
            msg += f": {resource_id}"
        self.message = msg
        super().__init__(self.message)


class ServerError(NeuronAgentError):
    """Server error"""
    def __init__(self, status_code, message="Server error"):
        self.status_code = status_code
        self.message = f"{message} (status: {status_code})"
        super().__init__(self.message)


class ValidationError(NeuronAgentError):
    """Validation error"""
    def __init__(self, message="Validation failed", errors=None):
        self.message = message
        self.errors = errors or []
        super().__init__(self.message)


class ConnectionError(NeuronAgentError):
    """Connection error"""
    def __init__(self, message="Failed to connect to server"):
        self.message = message
        super().__init__(self.message)


class TimeoutError(NeuronAgentError):
    """Request timeout"""
    def __init__(self, message="Request timed out"):
        self.message = message
        super().__init__(self.message)

