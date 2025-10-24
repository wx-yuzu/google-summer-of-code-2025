from pathlib import Path
from typing import List

import numpy
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from transformers import AutoModelForCausalLM, AutoProcessor
from ultralytics import YOLOE, YOLOWorld

from ir.model_output_ir import NormalizedDetections
from models.annotation_models.base import (
    AnnotationModel,
    OpenVocabularyAnnotationModel
)
from ontology.detection_ontology import DetectionOntology

ANNOTATOR_REGISTRY = {}


def register_annotator(name):
    def decorator(cls):
        ANNOTATOR_REGISTRY[name] = cls
        return cls

    return decorator


def create_annotator(model_name: str, **kwargs) -> AnnotationModel:
    return ANNOTATOR_REGISTRY[model_name](**kwargs)


@register_annotator("yolo11")
class Yolo11DetectionModel(AnnotationModel):
    def __init__(self, weights_path: Path, **kwargs):
        self.model = YOLOE(weights_path)
        self.input_image_size = (640, 640)

    def annotate(self, image_tensors):
        with torch.no_grad():
            outputs = self.model(image_tensors)[0]
        return NormalizedDetections.from_ultralytics(outputs)


@register_annotator("yolow")
class YoloWorldDetectionModel(OpenVocabularyAnnotationModel):
    def __init__(
        self,
        weights_path: Path,
        ontology_yaml_path: Path = None,
        use_sahi: bool = False,
        **kwargs,
    ):
        self.model = YOLOWorld(weights_path)
        self.input_image_size = (640, 640)
        if ontology_yaml_path:
            self.ontology = DetectionOntology.from_yaml(ontology_yaml_path)
            self.model.set_classes(self.ontology.prompts())
            self.class_names = self.ontology.classes()

    # [TODO] implement renaming in ontology
    def rename_class(self, outputs):
        # As the outputs.names attribute is consistently reused at the same memory address across all inferences.
        # Consequently, you should always make a fresh copy and replace it with a new dict instance for each inference.
        pred_classes = outputs.boxes.cls
        tmp = pred_classes.to(torch.long).clone()

        for k, v in self.ontology.id_migration_map.items():
            hit = tmp == k
            tmp[hit] = v

        outputs.boxes.data = outputs.boxes.data.clone()
        outputs.boxes.data[:, 5] = tmp.to(outputs.boxes.data.dtype)
        outputs.names = self.ontology.id_to_class_name

        return outputs

    def annotate(self, image_tensors):
        with torch.no_grad():
            outputs = self.model.predict(image_tensors)[0]

        if self.ontology:
            outputs = self.rename_class(outputs)

        return NormalizedDetections.from_ultralytics(outputs)


@register_annotator("yoloe")
class YoloeAnnotationModel(AnnotationModel):
    def __init__(self, weights_path: Path, class_names: List[str]):
        self.model = YOLOE(weights_path)
        self.input_image_size = (640, 640)
        self.class_names = class_names
        self.model.set_classes(class_names, self.model.get_text_pe(class_names))

    def annotate(self, image_tensors):
        with torch.no_grad():
            outputs = self.model.predict(image_tensors)[0]
        return NormalizedDetections.from_ultralytics(outputs)


@register_annotator("florence-2-large")
class Florence2(OpenVocabularyAnnotationModel):
    # [TODO] 'cuda'->'cpu'
    def __init__(self, ontology=None, ontology_yaml_path=None, device="cuda"):
        model_id = "microsoft/Florence-2-large"
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype="auto"
            )
            .eval()
            .to(device)
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.device = device
        self.ontology = None

        if ontology_yaml_path:
            self.ontology = DetectionOntology.from_yaml(ontology_yaml_path)
            self.class_prompts = self.ontology.prompts()
            self.class_labels = self.ontology.classes()

        if ontology:
            self.ontology = ontology
            self.class_prompts = self.ontology.prompts()
            self.class_labels = self.ontology.classes()

    def rename_class(self, prompt_labels):
        return []

    def annotate(
        self,
        image,
        task_prompt="<OPEN_VOCABULARY_DETECTION>",
        text_input=None,
        max_new_tokens=1024,
    ):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        if self.ontology is None:
            prompt = task_prompt
        else:
            text_input = "/".join(self.class_prompts)
            prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device, torch.float16
        )
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            pixel_values=inputs["pixel_values"].to(self.device),
            max_new_tokens=max_new_tokens,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )

        anns = parsed_answer[task_prompt]
        bboxes = anns["bboxes"]
        labels = anns["bboxes_labels"]

        # [TODO] Implement this
        if self.ontology:
            labels = self.rename_class(labels)
            class_ids = [self.ontology.get_class(lab) for lab in labels]

        return NormalizedDetections(
            xyxy=numpy.array(bboxes),
            cls=numpy.array(class_ids),
            box_mode="XYXY_ABS",
            orig_img=image,
            input_shape=image.shape,
        )


def test_construct_yolo11():
    # annotator = Yolo11nAnnotationModel(model_path="./src/models/weights/yolo11n.pt")
    annotator = create_annotator(
        model_name="yolo11", weights_path="./weights/yolo11n.pt"
    )


def test_annotate_with_yolo11():
    annotator = create_annotator(
        model_name="yolo11", weights_path="./weights/yolo11n.pt"
    )
    img = Image.open("./src/test/mock_datasets/cat.jpg")
    img = img.resize((640, 640))
    batch = torch.stack([to_tensor(img)])
    annotator.annotate(batch)


def test_construct_yolo_world():
    annotator = YoloWorldDetectionModel(
        weights_path="./weights/yolov8x-worldv2.pt",
        ontology=DetectionOntology(
            prompt_to_class={
                "purple grapes": "ripe grapes",
                "lightgreen grapes": "unripe grapes",
            }
        ),
    )


def test_annotate_with_yolo_world():
    annotator = YoloWorldDetectionModel(
        weights_path="./weights/yolov8x-worldv2.pt",
        ontology_yaml_path=Path("./src/tests/ontology.yaml"),
    )
    img = Image.open("./src/tests/mock_grapes_datasets/green_grapes.jpg")
    img = img.resize((640, 640))
    batch = torch.stack([to_tensor(img)])
    result = annotator.annotate(batch)
    result.save("./grape1.png")
    img = Image.open("./src/tests/mock_grapes_datasets/grapes.jpg")
    img = img.resize((640, 640))
    batch = torch.stack([to_tensor(img)])
    result = annotator.annotate(batch)

    result.save("./grape2.png")


def test_florence2():
    annotator = Florence2(
        ontology=DetectionOntology(
            prompt_to_class={
                "purple grapes": "ripe grapes",
                "lightgreen grapes": "unripe grapes",
            }
        )
    )
    img = Image.open("./src/tests/mock_grapes_datasets/green_grapes.jpg")
    result = annotator.annotate(img)
    print(result)


if __name__ == "__main__":
    test_florence2()
