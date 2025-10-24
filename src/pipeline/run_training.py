import logging
from pathlib import Path


def setup_logger(log_file: Path = None) -> logging.Logger:
    logger = logging.getLogger("train_yolo11n")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrainingRunner:
    def __init__(
        self,
        trainee_model,
    ):
        self.trainee_model = trainee_model

    def run_train(
        self,
        data_yaml: Path,
        pretrained_weights: Path,
        project_dir: Path,
        save_dir_name: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        device: str = "cuda",
    ):
        logger = setup_logger(project_dir / "train.log")
        logger.info(f"📦 Loading yolo11n pretrained weights: {pretrained_weights}")

        logger.info("🚀 Starting training...")

        results = self.trainee_model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(project_dir),
            name=save_dir_name,
            verbose=True,
        )

        logger.info("✅ Training complete.")
        logger.info(f"📁 Results saved in: {project_dir / 'yolo11m'}")
