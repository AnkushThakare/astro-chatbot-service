from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTError, jwt

try:
    from jose import JWTClaimsError
except ImportError:
    JWTClaimsError = JWTError  # type: ignore[assignment,misc]

from src.core.config import settings

bearer_scheme = HTTPBearer(auto_error=False)


class TokenValidationError(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class MissingTokenError(TokenValidationError):
    pass


class MalformedTokenError(TokenValidationError):
    pass


class ExpiredTokenError(TokenValidationError):
    pass


class InvalidSignatureTokenError(TokenValidationError):
    pass


class InvalidTokenError(TokenValidationError):
    pass


@dataclass
class AuthenticatedUser:
    user_id: str
    role: str
    token_type: str
    raw_token: str
    raw_claims: dict[str, Any]


def decode_jwt_token(token: str) -> AuthenticatedUser:
    if token.count(".") != 2:
        raise MalformedTokenError("Malformed bearer token")

    decode_kwargs: dict[str, Any] = {
        "token": token,
        "key": settings.JWT_SECRET_KEY,
        "algorithms": [settings.JWT_ALGORITHM],
        "options": {"require_exp": False, "verify_aud": bool(settings.JWT_AUDIENCE)},
    }
    if settings.JWT_AUDIENCE:
        decode_kwargs["audience"] = settings.JWT_AUDIENCE
    if settings.JWT_ISSUER:
        decode_kwargs["issuer"] = settings.JWT_ISSUER

    try:
        claims = jwt.decode(**decode_kwargs)
    except ExpiredSignatureError as exc:
        raise ExpiredTokenError("Bearer token expired") from exc
    except JWTError as exc:
        detail = str(exc)
        if "signature" in detail.lower():
            raise InvalidSignatureTokenError("Invalid bearer token signature") from exc
        if JWTClaimsError is not JWTError and isinstance(exc, JWTClaimsError):
            raise InvalidTokenError(f"Invalid bearer token claims: {exc}") from exc
        raise InvalidTokenError(f"Invalid bearer token: {exc}") from exc

    user_id = claims.get("sub") or claims.get("user_id")
    if not user_id:
        raise InvalidTokenError("JWT is missing subject claim")
    exp_value = claims.get("exp")
    if isinstance(exp_value, (int, float)):
        if exp_value + settings.JWT_EXP_GRACE_SECONDS <= time.time():
            raise ExpiredTokenError("Bearer token expired")

    return AuthenticatedUser(
        user_id=str(user_id),
        role=str(claims.get("role", "customer")),
        token_type=str(claims.get("type", "access")),
        raw_token=token,
        raw_claims=claims,
    )


def seconds_until_token_expiry(user: AuthenticatedUser) -> int | None:
    exp_value = user.raw_claims.get("exp")
    if not isinstance(exp_value, (int, float)):
        return None
    return int(exp_value - time.time())


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthenticatedUser:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=MissingTokenError("Missing bearer token").detail,
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_jwt_token(credentials.credentials)
    except TokenValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=exc.detail,
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def get_optional_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthenticatedUser | None:
    if credentials is None:
        return None
    try:
        user = decode_jwt_token(credentials.credentials)
    except TokenValidationError:
        # Token present but invalid (e.g. forwarded from core-service proxy).
        # Treat as anonymous rather than rejecting — the request is still valid.
        return None
    request.state.user = user
    return user
