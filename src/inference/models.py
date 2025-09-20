from pydantic import BaseModel
from catboost import CatBoostClassifier


class PredictionRequest(BaseModel):
    time: float
    amount: float


class PredictionResponse(BaseModel):
    predictions: float


class LoadModelRequest(BaseModel):
    model_path: str


class ModelWrapper:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        self.name = model_path.split("/")[-1]

    def predict(self, time: float, amount: float):
        features = [[time, amount]]
        return float(self.model.predict(features)[0])
