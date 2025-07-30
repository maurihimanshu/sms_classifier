"""
API middleware implementation.
"""
from typing import Callable, Dict, Optional
import time
from datetime import datetime, timedelta
import jwt
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging
import json
from ..config.base_config import BaseConfig

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiting implementation."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        minute_ago = now - 60

        # Initialize or clean old requests
        if client_id not in self.requests:
            self.requests[client_id] = []
        else:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > minute_ago
            ]

        # Check rate limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False

        # Add new request
        self.requests[client_id].append(now)
        return True

class JWTAuth:
    """JWT authentication handler."""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.security = HTTPBearer()

    def create_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT token."""
        expires = datetime.utcnow() + (
            expires_delta
            if expires_delta
            else timedelta(minutes=self.config.ACCESS_TOKEN_EXPIRE_MINUTES)
        )

        to_encode = {
            "sub": user_id,
            "exp": expires
        }

        return jwt.encode(
            to_encode,
            self.config.SECRET_KEY,
            algorithm=self.config.ALGORITHM
        )

    async def authenticate(
        self,
        credentials: HTTPAuthorizationCredentials
    ) -> Dict:
        """Authenticate JWT token."""
        try:
            payload = jwt.decode(
                credentials.credentials,
                self.config.SECRET_KEY,
                algorithms=[self.config.ALGORITHM]
            )

            if payload.get("exp") < time.time():
                raise HTTPException(
                    status_code=401,
                    detail="Token expired"
                )

            return payload

        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )

class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware."""

    def __init__(
        self,
        app,
        config: BaseConfig,
        exclude_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.auth_handler = JWTAuth(config)
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through authentication."""
        # Skip authentication for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Get and validate token
        auth = request.headers.get("Authorization")
        if not auth:
            raise HTTPException(
                status_code=401,
                detail="Authentication required"
            )

        try:
            scheme, credentials = auth.split()
            if scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=401,
                    detail="Invalid authentication scheme"
                )

            payload = await self.auth_handler.authenticate(
                HTTPAuthorizationCredentials(
                    scheme=scheme,
                    credentials=credentials
                )
            )

            # Add user info to request state
            request.state.user = payload

            return await call_next(request)

        except ValueError:
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization header"
            )

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        exclude_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.exclude_paths = exclude_paths or ["/health"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiting."""
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Get client identifier (user ID or IP)
        client_id = (
            getattr(request.state, "user", {}).get("sub")
            or request.client.host
        )

        # Check rate limit
        if not self.rate_limiter.is_allowed(client_id):
            raise HTTPException(
                status_code=429,
                detail="Too many requests"
            )

        return await call_next(request)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through logging."""
        # Start timing
        start_time = time.time()

        # Log request
        await self._log_request(request)

        # Process request
        try:
            response = await call_next(request)
            await self._log_response(request, response, start_time)
            return response

        except Exception as e:
            # Log error
            logger.error(
                "Request error",
                extra={
                    "url": str(request.url),
                    "method": request.method,
                    "error": str(e)
                }
            )
            raise

    async def _log_request(self, request: Request) -> None:
        """Log request details."""
        try:
            body = await request.body()
            body_str = body.decode() if body else ""

            logger.info(
                "Request received",
                extra={
                    "url": str(request.url),
                    "method": request.method,
                    "headers": dict(request.headers),
                    "body": body_str[:1000] if body_str else None,
                    "client_host": request.client.host
                }
            )
        except Exception as e:
            logger.error(f"Error logging request: {e}")

    async def _log_response(
        self,
        request: Request,
        response: Response,
        start_time: float
    ) -> None:
        """Log response details."""
        try:
            duration = time.time() - start_time

            logger.info(
                "Response sent",
                extra={
                    "url": str(request.url),
                    "method": request.method,
                    "status_code": response.status_code,
                    "duration_ms": int(duration * 1000),
                    "response_size": len(response.body) if hasattr(response, 'body') else 0
                }
            )
        except Exception as e:
            logger.error(f"Error logging response: {e}")

def create_middleware(app, config: BaseConfig):
    """Add middleware to application."""
    # Add authentication
    app.add_middleware(
        AuthMiddleware,
        config=config,
        exclude_paths=["/health", "/docs", "/openapi.json"]
    )

    # Add rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=60,
        exclude_paths=["/health"]
    )

    # Add logging
    app.add_middleware(LoggingMiddleware)