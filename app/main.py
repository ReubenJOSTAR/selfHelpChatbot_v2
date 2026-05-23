from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat


def create_app() -> FastAPI:
    """
    WHY a factory function instead of a global `app = FastAPI()`:

    A factory function lets you create different app instances for
    different environments (testing, development, production) with
    different settings. This costs nothing now and saves real pain later.
    """

    app = FastAPI(
        title="WalkWithJesus API",
        description="Scripture-grounded Christian AI guidance using RAG",
        version="1.0.0",
        docs_url="/docs",       # Swagger UI — visit http://localhost:8000/docs
        redoc_url="/redoc",     # ReDoc UI  — visit http://localhost:8000/redoc
    )

    # ---------------------------------------------------
    # CORS Middleware
    # WHY: Browsers block requests from a different origin
    # (e.g. React on localhost:3000 → FastAPI on localhost:8000)
    # unless the backend explicitly allows it.
    # This config allows your future frontend to connect.
    # In production, replace "*" with your actual frontend domain.
    # ---------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # Replace with ["https://yourfrontend.com"] in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------
    # Register Routes
    # WHY prefix="/api":
    # Groups all routes under /api/...
    # Easy to version later: /api/v1/chat, /api/v2/chat
    # Frontend always calls /api/... — clean and consistent
    # ---------------------------------------------------
    app.include_router(
        chat.router,
        prefix="/api",
        tags=["Chat"],
    )

    return app


# Create the app instance used by uvicorn
app = create_app()
