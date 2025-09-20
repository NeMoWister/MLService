from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from catboost import CatBoostClassifier
from pydantic import BaseModel, BaseSettings
import uvicorn
import os
from dotenv import load_dotenv
import logging
from logger import setup_logger
from models import PredictionRequest, PredictionResponse, LoadModelRequest, ModelWrapper
from config import Settings


settings = Settings()

app = FastAPI(title="ML Service")
setup_logger(
    settings.LOG_PATH_INFERENCE, settings.LOG_LEVEL
)
logger = logging.getLogger(__name__)

current_model: ModelWrapper | None = None

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Ошибка при обработке запроса {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": exc.__class__.__name__},
    )


@app.get("/health")
async def health_check():
    if current_model is None:
        logger.warning("Health-check: модель не загружена")
        return {"status": "No model loaded"}
    logger.info(f"Health-check: модель {current_model.name} доступна")
    return {"status": "ok", "model": current_model.name}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if current_model is None:
        logger.error("Попытка сделать предсказание без загруженной модели")
        raise HTTPException(status_code=400, detail="No model loaded")
    try:
        result = current_model.predict(request.time, request.amount)
        logger.info(
            f"Предсказание: features={{'time': {request.time}, 'amount': {request.amount}}} → {result}"
        )
        return PredictionResponse(predictions=result)
    except Exception as e:
        logger.exception(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")


@app.post("/load_model")
async def load_model(request: LoadModelRequest):
    global current_model
    try:
        current_model = ModelWrapper(request.model_path)
        logger.info(f"Загружена новая модель {current_model.name}")
        return {"status": "Model loaded", "model": current_model.name}
    except Exception as e:
        logger.exception(f"Ошибка загрузки модели: {e}")
        raise HTTPException(status_code=400, detail=f"Model loading error: {e}")


def start():
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)



@app.on_event("startup")
def load_default_model():
    global current_model
    default_path = settings.LATEST_DIR
    try:
        current_model = ModelWrapper(default_path)
        logger.info(f"Модель по умолчанию {current_model.name} загружена при старте")
    except Exception as e:
        logger.error(f"Не удалось загрузить модель по умолчанию: {e}")
        current_model = None


if __name__ == "__main__":
    start()
