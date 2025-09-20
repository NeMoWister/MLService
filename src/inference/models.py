from pydantic import BaseModel
import joblib
import pandas as pd
import os


class PredictionRequest(BaseModel):
    time: float
    amount: float


class PredictionResponse(BaseModel):
    predictions: float


class LoadModelRequest(BaseModel):
    model_path: str


class ModelWrapper:
    def __init__(self, model_path: str):
        if os.path.isdir(model_path):
            files = [f for f in os.listdir(model_path) if f.endswith(".joblib")]
            if not files:
                raise ValueError("В директории нет .joblib модели")
            model_file = sorted(files)[-1]  # берем самую свежую
            model_path = os.path.join(model_path, model_file)

        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.name = os.path.basename(model_path)

    def predict(self, time: float, amount: float):
        features = pd.DataFrame([{"time": time, "amount": amount}])
        return float(self.model.predict(features)[0])
