# train/train.py

import pandas as pd
import numpy as np
import random
import logging
import shutil
import time
import os
import yaml

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def divide_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


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


def main(config_path: str = None):
    # print(os.path.dirname(os.path.abspath(__file__)))
    # time.sleep(10000000)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if config_path is None:
        config_path = os.path.join(base_dir, "app", "configs", "train.yaml")

    SEED = config["seed"]
    np.random.seed(SEED)
    random.seed(SEED)

    setup_logger(config["logging"]["log_path"], config["logging"]["log_level"])
    logging.info("Запуск обучения модели")

    try:
        df = pd.read_csv(config["data"]["input_path"])
        logging.info(f"Загружено {len(df)} строк из {config['data']['input_path']}")
    except Exception as e:
        logging.exception("Ошибка при чтении данных")
        raise e

    if "drop_columns" in config["data"]:
        df.drop(columns=config["data"]["drop_columns"], inplace=True, errors="ignore")
        logging.info(f"Удалены колонки: {config['data']['drop_columns']}")

    positive_df = df[df[config["data"]["target_column"]] == 1]
    negative_df = df[df[config["data"]["target_column"]] == 0]

    neg_sample_size = config["data"]["balance"]["negative_sample"]
    negative_sample = negative_df.sample(n=neg_sample_size, random_state=SEED)

    balanced_df = pd.concat([positive_df, negative_sample], ignore_index=True)
    if config["data"]["balance"]["shuffle"]:
        balanced_df = balanced_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    logging.info(f"Балансировка: {len(positive_df)} положительных, {neg_sample_size} отрицательных")

    X, y = divide_data(balanced_df, config["data"]["target_column"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["split"]["test_size"],
        random_state=SEED,
        stratify=y if config["split"].get("stratify", False) else None
    )

    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_test, y_test)

    model = CatBoostClassifier(**config["model"]["params"])
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    logging.info("Модель обучена")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {}
    if "roc_auc" in config["metrics"]:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    if "accuracy" in config["metrics"]:
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
    if "precision" in config["metrics"]:
        metrics["precision"] = precision_score(y_test, y_pred)
    if "recall" in config["metrics"]:
        metrics["recall"] = recall_score(y_test, y_pred)
    if "f1" in config["metrics"]:
        metrics["f1"] = f1_score(y_test, y_pred)

    logging.info(f"Метрики: {metrics}")

    latest_dir = config["output"]["latest_dir"]
    archive_dir = config["output"]["archive_dir"]
    os.makedirs(latest_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)

    existing_models = os.listdir(latest_dir)
    if existing_models:
        old_model = os.path.join(latest_dir, existing_models[0])
        shutil.move(old_model, archive_dir)
        logging.info(f"Старая модель перемещена в {archive_dir}")

    timestamp = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    save_path = os.path.join(latest_dir, f"{timestamp}.{config['output']['save_format']}")
    model.save_model(save_path)
    logging.info(f"Новая модель сохранена в {save_path}")


if __name__ == "__main__":
    main()
