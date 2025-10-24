from pathlib import Path

from models.trainee_models.factory import create_trainee
from pipeline.run_training import TrainingRunner


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train yolo11n with fine-tuning.")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument(
        "--weights", type=str, required=True, help="Pretrained yolo11n weights (.pt)"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model config path"
    )
    parser.add_argument(
        "--save_dir_name", type=str, required=True, help="Model config path"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # load a model to be trained
    trainee_model = create_trainee(args.model_name, weights_path=args.weights)
    runner = TrainingRunner(trainee_model=trainee_model)

    runner.run_train(
        data_yaml=Path(args.data),
        pretrained_weights=Path(args.weights),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project_dir=Path(args.output),
        save_dir_name=args.save_dir_name,
    )


if __name__ == "__main__":
    main()
