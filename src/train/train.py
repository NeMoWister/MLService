import pandas as pd
import numpy as np
import random
import shutil
import time
import os
import requests
import ast
import logging
import joblib
from logger import setup_logger

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from catboost import CatBoostClassifier

from config import settings


def divide_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def main():
    np.random.seed(settings.SEED)
    random.seed(settings.SEED)

    setup_logger(settings.LOG_PATH_TRAIN, settings.LOG_LEVEL)
    logging.info("Запуск обучения модели")

    try:
        df = pd.read_csv(settings.INPUT_PATH)
        logging.info(f"Загружено {len(df)} строк из {settings.INPUT_PATH}")
    except Exception as e:
        logging.exception("Ошибка при чтении данных")
        raise e

    if settings.DROP_COLUMNS:
        drop_columns = [c.strip() for c in settings.DROP_COLUMNS.split(",")]
        df.drop(columns=drop_columns, inplace=True, errors="ignore")
        logging.info(f"Удалены колонки: {drop_columns}")

    target_column = settings.TARGET_COLUMN
    positive_df = df[df[target_column] == 1]
    negative_df = df[df[target_column] == 0]

    neg_sample_size = settings.NEGATIVE_SAMPLE or len(negative_df)
    negative_sample = negative_df.sample(n=neg_sample_size, random_state=settings.SEED)

    balanced_df = pd.concat([positive_df, negative_sample], ignore_index=True)
    if settings.SHUFFLE:
        balanced_df = balanced_df.sample(frac=1, random_state=settings.SEED).reset_index(drop=True)

    logging.info(f"Балансировка: {len(positive_df)} положительных, {neg_sample_size} отрицательных")

    X, y = divide_data(balanced_df, target_column)
    stratify = y if settings.STRATIFY else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.TEST_SIZE,
        random_state=settings.SEED,
        stratify=stratify
    )

    model_params = ast.literal_eval(settings.MODEL_PARAMS)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("squared", FunctionTransformer(lambda x: np.power(x, 2))),
        ("model", CatBoostClassifier(**model_params))
    ])

    pipeline.fit(X_train, y_train)
    logging.info("Модель обучена")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {}
    for m in settings.METRICS.split(","):
        m = m.strip()
        if m == "roc_auc":
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        elif m == "accuracy":
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
        elif m == "precision":
            metrics["precision"] = precision_score(y_test, y_pred)
        elif m == "recall":
            metrics["recall"] = recall_score(y_test, y_pred)
        elif m == "f1":
            metrics["f1"] = f1_score(y_test, y_pred)

    logging.info(f"Метрики: {metrics}")

    os.makedirs(settings.LATEST_DIR, exist_ok=True)
    os.makedirs(settings.ARCHIVE_DIR, exist_ok=True)

    existing_models = os.listdir(settings.LATEST_DIR)
    if existing_models:
        old_model = os.path.join(settings.LATEST_DIR, existing_models[0])
        shutil.move(old_model, settings.ARCHIVE_DIR)
        logging.info(f"Старая модель перемещена в {settings.ARCHIVE_DIR}")

    timestamp = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    save_path = os.path.join(settings.LATEST_DIR, f"{timestamp}.joblib")
    joblib.dump(pipeline, save_path)
    logging.info(f"Новая модель сохранена в {save_path}")

    if settings.INFERENCE_URL:
        payload = {"model_path": settings.LATEST_DIR}
        try:
            response = requests.post(settings.INFERENCE_URL, json=payload)
            if response.status_code == 200:
                logging.info("Модель успешно обновлена в inference сервисе")
            else:
                logging.error(f"Ошибка при обновлении модели: {response.status_code}, {response.text}")
        except Exception as e:
            logging.exception("Ошибка при запросе к inference сервису")


if __name__ == "__main__":
    main()
