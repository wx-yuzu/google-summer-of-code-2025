import logging

from annotation.annotation_exporter import AnnotationExporter
from annotation.image_importer import ImageImporter
from annotation.postprocess import PostProcessor
from annotation.preprocess import PreProcessor
from ir.model_output_ir import NormalizedDetections
from models.annotation_models.base import AnnotationModel

logger = logging.getLogger(__name__)


class AnnotationRunner:
    def __init__(
        self,
        importer: ImageImporter,
        exporter: AnnotationExporter,
        annotator: AnnotationModel,
        preprocessor: PreProcessor,
        postprocessor: PostProcessor,
    ):
        self.importer = importer
        self.exporter = exporter
        self.annotator = annotator
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def run(self):
        # 先に IR を用意して Exporter に渡す
        images = self.importer.read_all_images()
        accepted_paths = []
        detections: list[NormalizedDetections] = []
        raw_outputs = []

        for path, image in zip(self.importer.image_paths, images):
            try:
                processed = self.preprocessor.preprocess_image(image)
                raw_output = self.annotator.annotate(processed)
                raw_outputs.append(raw_output)

                # [TODO] これはモデルごとに実装した方がいいのでは
                # PostProcessor は NormalizedDetections を返す実装に統一
                det: NormalizedDetections = self.postprocessor.normalized_postprocess(
                    raw_output,
                    conf_threshold=getattr(self, "conf_threshold", 0.0),
                    iou_threshold=getattr(self, "iou_threshold", 0.0),
                    target_hw=(
                        image.height,
                        image.width,
                    ),  # (H, W): 念のためのスケール補正
                )

                if det.xyxy.shape[0] == 0:
                    logger.info(f"No kept detections after thresholds: {path.name}")
                    continue

                detections.append(det)
                accepted_paths.append(path)
                logger.info(f"Annotated {path.name} ({det.xyxy.shape[0]} objs)")

            except Exception as e:
                logger.error(f"Failed to process {path.name}: {e}")

        # save_raw_outputs(raw_outputs=raw_outputs, paths=accepted_paths)

        # export to datumaro format
        self.exporter.add_from_normalized_batch(
            image_paths=accepted_paths,
            detections=detections,
            train_ratio=0.8,  # [TODO] set in user configuration
            # extra_attrs={"source": {"model": self.annotator.name, "version": getattr(self.annotator, "version", "")}},
        )

        self.exporter.export(save_media=True)

        logger.info(f"Exported dataset to {self.exporter.output_dir}")
