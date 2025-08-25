class RequestException(Exception):
    """Base exception for request errors."""

    pass


class RateLimitException(RequestException):
    """Raised when rate limit is hit."""

    pass


class BlockedException(RequestException):
    """Raised when request is blocked (captcha, IP ban)."""

    pass


class ParseException(Exception):
    """Raised when parsing fails."""

    pass


class PaginationException(Exception):
    """Raised when pagination fails."""

    pass


class ValidationException(Exception):
    """Raised when data validation fails."""

    pass
