from dataclasses import dataclass
from uuid import UUID

from pydantic import SecretStr

from app.clients.store import StoreClient


@dataclass(frozen=True, slots=True)
class UserContext:
    """Who is making the request. Built by our code, never by the model.

    Tools receive this as a separate argument that is not part of the model-visible
    schema, so the model cannot choose, change or even see whose data it acts on.
    """

    customer_id: UUID
    store_token: SecretStr

    @classmethod
    async def from_token(cls, client: StoreClient, token: str) -> "UserContext":
        """Build a context from a store token; customer_id comes only from /auth/me."""
        user = await client.get_me(token)
        return cls(customer_id=user.id, store_token=SecretStr(token))
