#!/bin/bash

PYTHONPATH=./src python3 ./src/main_annotation.py \
    --model yoloe \
    --ontology_path ./src/tests/ontology.yaml \
    --input_dir ./dataset/wgisd_test/images \
    --output_dir ./dump/dump_yoloe_ws_coco_test \
    --weights ./weights/yolov8x-worldv2.pt \
    --export_format coco

# PYTHONPATH=./src python3 ./src/main_annotation.py \
#     --model yolo11 \
#     --input_dir ./dataset/wgisd_all_train/images \
#     --output_dir ./dump/dump_yolo11m_ws_all_train_after_training_last_pt \
#     --weights ./src/runs/train_yolo11n/yolo11m6_from_yolow/weights/last.pt