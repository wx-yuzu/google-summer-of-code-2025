import argparse
from pathlib import Path

from annotation.annotation_exporter import AnnotationExporter
from annotation.image_importer import ImageImporter
from annotation.postprocess import PostProcessor
from annotation.preprocess import PreProcessor
from ir.dataset_ir import DatasetIR, DatasetIRConfig
from models.annotation_models.factory import create_annotator
from pipeline.run_annotation import AnnotationRunner


def main():
    parser = argparse.ArgumentParser(description="Annotate images")
    parser.add_argument("--model_name", "-m", required=True, type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--class_names", nargs="*", type=str)
    parser.add_argument("--ontology_path", type=str)
    parser.add_argument("--export_format", type=str)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    weights_path = Path(args.weights)

    # (1) create an image dataset loader
    importer = ImageImporter(image_dir=input_dir)

    # (2) load an annotation model
    annotator = create_annotator(
        args.model_name,
        weights_path=weights_path,
        class_names=args.class_names,
        ontology_yaml_path=args.ontology_path,
    )

    # (3) preprocess images
    preprocessor = PreProcessor(image_size=annotator.input_image_size)

    # (4) annotation postprocessor (e.g. filter unconfident annotations)
    postprocessor = PostProcessor()

    # (5) handle class_names
    print(args.class_names)
    if args.class_names:
        class_names = args.class_names
    else:
        class_names = list(annotator.model.names.values())

    cfg = DatasetIRConfig(
        categories=class_names,  # args.class_names,
        default_subset="train",
        yolo_format="yolo_ultralytics",
        box_mode="XYXY_ABS",
    )

    exporter = AnnotationExporter(
        output_dir=output_dir,
        ir=DatasetIR(cfg),
        fmt=args.export_format,
    )

    runner = AnnotationRunner(
        importer=importer,
        exporter=exporter,
        annotator=annotator,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    runner.run()


if __name__ == "__main__":
    main()
