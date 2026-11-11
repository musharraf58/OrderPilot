from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.deps import CurrentUser, get_store
from app.clients.store import StoreClient, StoreError, StoreUnavailable

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class MeResponse(BaseModel):
    customer_id: str


@router.post("/login")
async def login(
    body: LoginRequest, store: Annotated[StoreClient, Depends(get_store)]
) -> LoginResponse:
    """Forward the credentials to the store and return its token. We store nothing."""
    try:
        token = await store.login(body.email, body.password)
    except StoreUnavailable as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The store is unavailable. Try again later."
        ) from exc
    except StoreError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password.") from exc
    return LoginResponse(access_token=token.access_token)


@router.get("/me")
async def me(user: CurrentUser) -> MeResponse:
    return MeResponse(customer_id=str(user.customer_id))
