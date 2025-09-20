from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
import logging

from backend import predict, load_model, health_check, load_default_model
from config import Settings
from logger import setup_logger

settings = Settings()
app = FastAPI(title="ML Service")

setup_logger(settings.LOG_PATH_INFERENCE, settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Ошибка при обработке запроса {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": exc.__class__.__name__},
    )


@app.get("/health")
async def health():
    return await health_check()


@app.post("/predict")
async def predict_route(request):
    return await predict(request)


@app.post("/load_model")
async def load_model_route(request):
    return await load_model(request)


@app.on_event("startup")
def startup_event():
    load_default_model()


def start():
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)


if __name__ == "__main__":
    start()
