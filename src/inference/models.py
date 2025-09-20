from catboost import CatBoostClassifier
from pydantic import BaseModel
import logger
import os


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
