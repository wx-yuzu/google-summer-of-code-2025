#!/bin/bash
PYTHONPATH=./src python3 ./src/main_annotation.py \
    --model yoloe \
    --ontology_path <path/to/ontology.yaml> \
    --input_dir <path/to/input_image_dir> \
    --output_dir <path/to/output_dir> \
    --weights <path/to/yolo_model_weights> \
    --export_format yolo