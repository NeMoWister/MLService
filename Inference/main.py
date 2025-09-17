from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List, Any
from catboost import CatBoostClassifier
from pydantic import BaseModel
import uvicorn
import os


app = FastAPI(title='ML Service')


class PredictionRequest(BaseModel): # для каждой фичи нужно прописать значения отдельно
    time: int
    amount: float
 

class PredictionResponse(BaseModel):
    predictions: float


class LoadModelRequest(BaseModel):
    model_path: str


class ModelWrapper:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Model file {path} not found')
        self.model = CatBoostClassifier()
        self.model_path = path
        self.model.load_model(self.model_path)
        self.name = os.path.basename(path)


    def predict(self, time: int, amount: float) -> float:
        return self.model.predict([[time, amount]])[0]
    

def start():
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)

current_model: ModelWrapper = None


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            'error': str(exc),
            'type': exc.__class__.__name__
        }
    )


@app.get('/health')
async def health_check():
    if current_model is None:
        return {'status': 'No model loaded'}
    return {'status': 'ok', 'model': current_model.name}


@app.post('/predict', response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if current_model is None:
        raise HTTPException(status_code=400, detail='No model loaded')
    try:
        result = current_model.predict(request.time, request.amount)
        return PredictionResponse(predictions=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Prediction error: {e}')
    

@app.post('/load_model') # вопрос: нужно ли какие то права настроить, на тот случай если кто то левый захочет её сменить
async def load_model(request: LoadModelRequest):
    global current_model
    try:
        current_model = ModelWrapper(request.model_path)
        # нужно ли возвращать ответ о том что модель загружена? по сути, этот эндпоинт используется только скриптом обучения
        return JSONResponse(
        content = {
            'status': 'Model loaded'
        })
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Model loading error: {e}')
    
    
if __name__ == '__main__':
    start()