import pandas as pd
import numpy as np
import random
import logging
import shutil
import time
import os
import requests
import ast

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv


def setup_logger(log_path: str, log_level: str = "INFO"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, log_level.upper()))
    logging.getLogger().addHandler(console)


def divide_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def main():
    load_dotenv()

    SEED = int(os.getenv("SEED", 42))
    np.random.seed(SEED)
    random.seed(SEED)

    setup_logger(os.getenv("LOG_PATH_TRAIN", "/app/logs/train.log"),
                 os.getenv("LOG_LEVEL", "INFO"))
    logging.info("Запуск обучения модели")

    try:
        df = pd.read_csv(os.getenv("INPUT_PATH"))
        logging.info(f"Загружено {len(df)} строк из {os.getenv('INPUT_PATH')}")
    except Exception as e:
        logging.exception("Ошибка при чтении данных")
        raise e

    drop_columns = os.getenv("DROP_COLUMNS")
    if drop_columns:
        drop_columns = [c.strip() for c in drop_columns.split(",")]
        df.drop(columns=drop_columns, inplace=True, errors="ignore")
        logging.info(f"Удалены колонки: {drop_columns}")

    target_column = os.getenv("TARGET_COLUMN", "Class")
    positive_df = df[df[target_column] == 1]
    negative_df = df[df[target_column] == 0]

    neg_sample_size = int(os.getenv("NEGATIVE_SAMPLE", len(negative_df)))
    negative_sample = negative_df.sample(n=neg_sample_size, random_state=SEED)

    balanced_df = pd.concat([positive_df, negative_sample], ignore_index=True)
    if os.getenv("SHUFFLE", "True").lower() == "true":
        balanced_df = balanced_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    logging.info(f"Балансировка: {len(positive_df)} положительных, {neg_sample_size} отрицательных")

    X, y = divide_data(balanced_df, target_column)
    stratify = y if os.getenv("STRATIFY", "False").lower() == "true" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(os.getenv("TEST_SIZE", 0.2)),
        random_state=SEED,
        stratify=stratify
    )

    train_pool = Pool(X_train.values, y_train)
    valid_pool = Pool(X_test.values, y_test)

    model_params = ast.literal_eval(os.getenv("MODEL_PARAMS", "{}"))
    model = CatBoostClassifier(**model_params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    logging.info("Модель обучена")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {}
    for m in os.getenv("METRICS", "roc_auc,accuracy").split(","):
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

    latest_dir = os.getenv("LATEST_DIR", "/app/models/latest")
    archive_dir = os.getenv("ARCHIVE_DIR", "/app/models/archive")
    os.makedirs(latest_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)

    existing_models = os.listdir(latest_dir)
    if existing_models:
        old_model = os.path.join(latest_dir, existing_models[0])
        shutil.move(old_model, archive_dir)
        logging.info(f"Старая модель перемещена в {archive_dir}")

    timestamp = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    save_format = os.getenv("SAVE_FORMAT", "cbm")
    save_path = os.path.join(latest_dir, f"{timestamp}.{save_format}")
    model.save_model(save_path)
    logging.info(f"Новая модель сохранена в {save_path}")

    url = os.getenv("INFERENCE_URL")
    if url:
        payload = {"model_path": latest_dir}
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                logging.info("Модель успешно обновлена в inference сервисе")
            else:
                logging.error(f"Ошибка при обновлении модели: {response.status_code}, {response.text}")
        except Exception as e:
            logging.exception("Ошибка при запросе к inference сервису")


if __name__ == "__main__":
    main()
