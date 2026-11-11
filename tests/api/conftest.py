import pytest

from app.agent.llm import ScriptedLLM
from app.db.repositories import ConversationRepository
from tests.conftest import AUTH
from tests.factories import CUSTOMER_ID, me


@pytest.fixture
def chat(api_app, api, store, sessionmaker):
    """The API with a test DB; call chat.script([...]) to give it a ScriptedLLM."""
    store.get("/auth/me").respond(200, json=me())
    api_app.state.sessionmaker = sessionmaker

    class Chat:
        client = api
        sessions = sessionmaker
        llm: ScriptedLLM | None = None

        def script(self, steps, chunk_size=4):
            self.llm = ScriptedLLM(steps, chunk_size=chunk_size)
            api_app.state.llm = self.llm
            return self.llm

        async def new(self):
            r = await api.post("/conversations", headers=AUTH)
            assert r.status_code == 201
            return r.json()["id"]

        async def send(self, conversation_id, message):
            r = await api.post(
                f"/conversations/{conversation_id}/messages", json={"text": message}, headers=AUTH
            )
            return r

        async def rows(self, conversation_id):
            async with sessionmaker() as s:
                return await ConversationRepository(s).load_messages(CUSTOMER_ID, conversation_id)

    return Chat()
