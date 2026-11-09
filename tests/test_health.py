from app.db.session import get_session


class OkSession:
    async def execute(self, *_args, **_kwargs):
        return None


class BrokenSession:
    async def execute(self, *_args, **_kwargs):
        raise ConnectionError("db down")


def override_session(app, session):
    async def _dep():
        yield session

    app.dependency_overrides[get_session] = _dep


async def test_health_ok(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_ready_ok(app, client):
    override_session(app, OkSession())
    r = await client.get("/health/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}


async def test_ready_returns_503_when_db_fails(app, client):
    override_session(app, BrokenSession())
    r = await client.get("/health/ready")
    assert r.status_code == 503
    assert r.json() == {"status": "unavailable"}


async def test_health_stays_200_when_db_fails(app, client):
    override_session(app, BrokenSession())
    assert (await client.get("/health")).status_code == 200
