from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from catboost import CatBoostClassifier
from pydantic import BaseModel
import uvicorn
import os
import yaml
import logging


def load_config(path: str = "configs/inference.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(log_path: str, log_level: str = "INFO"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


app = FastAPI(title="ML Service")
config = load_config()
setup_logger(
    config.get("logging", {}).get("log_path", "logs/service.log"),
    config.get("logging", {}).get("log_level", "INFO"),
)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    time: int
    amount: float


class PredictionResponse(BaseModel):
    predictions: float


class LoadModelRequest(BaseModel):
    model_path: str


class ModelWrapper:
    '''
    Принимает папку или конкретный файл. в случае, если передана папка берет первую модель. нужно для загрузки модели по дефолту в случае перезапуска
    '''
    def __init__(self, path: str):
        if os.path.isdir(path):
            files = [f for f in os.listdir(path) if f.endswith(".cbm")]
            if not files:
                raise FileNotFoundError(f"В папке {path} нет .cbm моделей")
            path = os.path.join(path, files[0])

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Модель {path} не найдена")

        self.model = CatBoostClassifier()
        self.model.load_model(path)
        self.model_path = path
        self.name = os.path.basename(path)
        logger.info(f"Модель {self.name} загружена из {path}")


    def predict(self, time: int, amount: float) -> float:
        return self.model.predict([[time, amount]])[0]


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
    uvicorn.run(
        "main:app",
        host=config["service"]["host"],
        port=config["service"]["port"],
        reload=config["service"]["reload"],
    )


@app.on_event("startup")
def load_default_model():
    global current_model
    default_path = config["model"]["default_path"]
    try:
        current_model = ModelWrapper(default_path)
        logger.info(f"Модель по умолчанию {current_model.name} загружена при старте")
    except Exception as e:
        logger.error(f"Не удалось загрузить модель по умолчанию: {e}")
        current_model = None


if __name__ == "__main__":
    start()
