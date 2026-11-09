from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.middleware import RequestIDMiddleware


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    app = FastAPI(title="OrderPilot")
    app.add_middleware(RequestIDMiddleware)
    return app


app = create_app()
