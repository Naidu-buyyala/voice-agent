from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone

from contextlib import asynccontextmanager
from app.config.settings import get_settings
from app.db.database import engine
from app.db.models import Base
from app.api.routes.conversations import router as conversations_router
from app.api.routes.tasks import router as tasks_router
from app.api.routes.uber_auth import router as uber_auth_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure database tables exist on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        description="Universal AI Action Agent - Backend MVP for ride booking.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS configuration for REST clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API Routers
    app.include_router(conversations_router, prefix="/api/v1")
    app.include_router(tasks_router, prefix="/api/v1")
    app.include_router(uber_auth_router, prefix="/api/v1")

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {
            "status": "ok",
            "app": settings.APP_NAME,
            "environment": settings.APP_ENV,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    from fastapi.responses import FileResponse
    from pathlib import Path

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        ui_file = Path("app/static/index.html")
        if ui_file.exists():
            return FileResponse(ui_file)
        return {"message": "Action Agent API is active. Visit /docs for Swagger."}

    return app


app = create_app()
