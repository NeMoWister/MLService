import logging
import joblib
import pandas as pd
from fastapi import HTTPException
from models import PredictionRequest, PredictionResponse, LoadModelRequest, ModelWrapper
from config import Settings

settings = Settings()
logger = logging.getLogger(__name__)

current_model: ModelWrapper | None = None


async def health_check():
    if current_model is None:
        logger.warning("Health-check: модель не загружена")
        return {"status": "No model loaded"}
    logger.info(f"Health-check: модель {current_model.name} доступна")
    return {"status": "ok", "model": current_model.name}


async def predict(request: PredictionRequest):
    global current_model
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


async def load_model(request: LoadModelRequest):
    global current_model
    try:
        current_model = ModelWrapper(request.model_path)
        logger.info(f"Загружена новая модель {current_model.name}")
        return {"status": "Model loaded", "model": current_model.name}
    except Exception as e:
        logger.exception(f"Ошибка загрузки модели: {e}")
        raise HTTPException(status_code=400, detail=f"Model loading error: {e}")


def load_default_model():
    global current_model
    default_path = settings.LATEST_DIR
    try:
        current_model = ModelWrapper(default_path)
        logger.info(f"Модель по умолчанию {current_model.name} загружена при старте")
    except Exception as e:
        logger.error(f"Не удалось загрузить модель по умолчанию: {e}")
        current_model = None
