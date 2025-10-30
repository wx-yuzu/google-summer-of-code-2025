# 📝 Open-Vocabulary Annotation Toolkit

A lightweight toolkit that turns open-vocabulary object detection models into automatic annotators. Run your favorite detectors (YOLOW, YOLOE, Florence-2) on your images with any classes you define in natural language. Export clean labels in YOLO (Ultralytics) or COCO formats. Designed to support knowledge distillation and rapid dataset building.



## ⭐️ Features

- Prompt-anything: detect arbitrary concepts by passing class names as prompts
- Multiple models: YOLOW / YOLOE / Florence-2
- Pluggable ontology: map model-friendly prompts → human-friendly class labels with YAML
- Exporter included: save annotations as YOLO or COCO out of the box
- Batchable shell script: one command to process a whole directory of images



## ⏰ Quick Start

### 1) Install [uv](https://github.com/astral-sh/uv) (Python package manager)

See official instructions for your platform.

### 2) Create a virtual environment and install dependencies

```bash
uv venv .venv
source .venv/bin/activate # (for Mac)
uv pip install -r requirements.txt
```

### 3) Run the annotator

```bash
sh src/annotation.sh
```

That’s it — the script will run inference and export annotations according to your configuration.



## ⚙️ Configuration

All runtime options are in `src/annotation.sh`. Edit it for your environment.

#### Script arguments (at a glance)

| Name            | Example                 | Notes                                   |
|-----------------|-------------------------|-----------------------------------------|
| model           | `yolow`                | `yolow`, `yoloe`, `florence-2` |
| input_dir       | `data/images`           | Images directly under this directory    |
| output_dir      | `runs/labels`           | Created if missing                      |
| weights         | `weights/yolo11.pt`     | For Ultralytics models                  |
| export_format   | `yolo` / `coco`         | Output format                           |
| ontology_path   | `configs/ontology.yaml` | prompt→label mapping           |

---

#### Required parameters

- `model`: one of `yolow`, `yoloe`, `florence-2`
- `input_dir`: directory containing input images (files directly under this directory)
- `output_dir`: directory where annotation results will be written
- `export_format`: `yolo` or `coco`
- `ontology_path`: path to a YAML file that defines a mapping from prompts to your final class labels (details below)

#### Optional parameters
- `weights`: path to model weights (for Ultralytics models: YOLOW / YOLOE)

#### Examples

```bash
--model yoloe \
--ontology_path <path/to/ontology.yaml> \
--input_dir <path/to/input_image_dir> \
--output_dir <path/to/output_dir> \
--weights <path/to/yolo_model_weights> \
--export_format coco
```



## 📕 Ontology (prompt ↔ label mapping)

Open-vocabulary models often respond best to descriptive prompts, while your dataset needs concise, human-friendly labels. The ontology bridges the two.

- Prompts: what the model sees (free-form text)
- Class labels: what you want in your dataset

### Example 1: one-to-one mapping

```yaml
# ontology.yaml
purple grapes: ripe_grapes
lightgreen grapes: unripe_grapes
```

Run with:
```bash
--ontology_path ontology.yaml
```
The toolkit will export labels as `ripe_grapes` and `unripe_grapes`.

### Example 2: many prompts → one label (aggregation)

```yaml
purple grapes: grapes
lightgreen grapes: grapes
```

Multiple prompts can aggregate to the same final class (useful when you probe a variety of prompts).  
Note: A single prompt cannot diverge into multiple final class labels.

### When to use an ontology?

- You want model-friendly phrasing (“lightgreen grapes”) but standardized labels (“unripe_grapes”)
- You’re experimenting with several prompt variants and want them collapsed to one class


<!-- ## Tips and good practices

- Prompt engineering helps: try synonyms, adjectives, or context (“ripe red apple on a tree”)
- Aggregate variants with the ontology to keep your final label set clean
- Version your configs: commit `annotation.sh` and your YAMLs for reproducibility
- Sanity-check outputs: spot-check a few images per class before large jobs

--- -->

## ⚠️ Troubleshooting

- No detections  
  Try broader or simpler prompts, or check you pointed to the correct weights.

- Wrong classes in export  
  Verify `ontology.yaml` keys match your prompts exactly (case/spacing).

- Empty `output_dir`  
  Confirm `input_dir` contains images directly (not nested subfolders).



## 🎓 Citation

If this toolkit helps your research or production work, please consider citing or referencing this repository.



## ❤️ Acknowledgements

This project builds on open-vocabulary detection research and implementations including YOLO variants and Florence-2.
