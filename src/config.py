from pydantic import BaseSettings


class Settings(BaseSettings):
    SEED: int = 42
    LOG_LEVEL: str = "INFO"

    LOG_PATH_TRAIN: str = "/app/logs/train.log"
    LOG_PATH_INFERENCE: str = "/app/logs/inference.log"
    INPUT_PATH: str | None = None
    LATEST_DIR: str = "/app/models/latest"
    ARCHIVE_DIR: str = "/app/models/archive"

    DROP_COLUMNS: str | None = None
    TARGET_COLUMN: str = "Class"
    NEGATIVE_SAMPLE: int | None = None
    SHUFFLE: bool = True
    STRATIFY: bool = False
    TEST_SIZE: float = 0.2
    MODEL_PARAMS: str = "{}"
    METRICS: str = "roc_auc,accuracy"

    SAVE_FORMAT: str = "cbm"

    INFERENCE_URL: str | None = None
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()
