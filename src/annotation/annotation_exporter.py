from pathlib import Path
from typing import Optional, Sequence

from ir.dataset_ir import DatasetIR
from ir.model_output_ir import NormalizedDetections


# FB: Rename to understand what this class does at a glance
class AnnotationExporter:
    """Export annotations and media from a DatasetIR to target formats.

    Notes
    -----
    - This class orchestrates directory creation and delegates the actual dataset
      serialization to the underlying `DatasetIR.ds.export(...)` (Datumaro).
    - Train/val split assignment can be driven by explicit `subsets` or by a
      simple ratio when using `add_from_normalized_batch`.
    """

    def __init__(self, output_dir: Path, ir: DatasetIR, fmt: str):
        """Initialize the exporter with an output directory, IR, and format.

        Parameters
        ----------
        output_dir : pathlib.Path
            Root directory where export artifacts will be written.
        ir : DatasetIR
            Intermediate representation that wraps a Datumaro dataset.
        fmt : str
            Target export format selector (for example, `"yolo"` or `"coco"`).
        """
        self.output_dir = output_dir
        self.fmt = fmt
        self.ir = ir
        self.class_names = ir.cfg.categories

    def ensure_export_dirs(self) -> None:
        """Create (if missing) the directory structure used during export.

        Side Effects
        ------------
        - Sets instance attributes for commonly used subdirectories:
          `image_dir_path`, `label_dir_path`, `raw_dir_path`,
          `train_image_dir_path`, `val_image_dir_path`,
          `train_label_dir_path`, `val_label_dir_path`.
        - Ensures all of the above paths exist on disk.
        """
        self.image_dir_path = self.output_dir / "images"
        self.label_dir_path = self.output_dir / "labels"
        self.raw_dir_path = self.output_dir / "raw_output"
        self.train_image_dir_path = self.image_dir_path / "train"
        self.val_image_dir_path = self.image_dir_path / "val"
        self.train_label_dir_path = self.label_dir_path / "train"
        self.val_label_dir_path = self.label_dir_path / "val"

        for p in [
            self.output_dir,
            self.image_dir_path,
            self.label_dir_path,
            self.train_image_dir_path,
            self.val_image_dir_path,
            self.train_label_dir_path,
            self.val_label_dir_path,
            self.raw_dir_path,
        ]:
            p.mkdir(parents=True, exist_ok=True)

    # ==========================
    # Public API: Save NormalizedDetections
    # ==========================
    def add_from_normalized_batch(
        self,
        image_paths: Sequence[Path | str],
        detections: Sequence[NormalizedDetections],
        subsets: Optional[Sequence[str]] = None,
        train_ratio: float = 0.8,
        extra_attrs: Optional[dict] = None,
    ) -> None:
        """Append a batch of images and detections into the IR dataset.

        If `subsets` is not provided, the first `int(n * train_ratio)` items are
        assigned to `"train"` and the remainder to `"val"`. This method expects
        detections in pixel-space XYXY with `conf` and `cls` arrays.

        Parameters
        ----------
        image_paths : Sequence[pathlib.Path | str]
            Paths to images to be added.
        detections : Sequence[NormalizedDetections]
            Detection objects aligned with `image_paths`.
        subsets : Sequence[str] | None, optional
            Per-item subset labels (e.g., `"train"` or `"val"`). If `None`,
            a simple ratio-based split is applied.
        train_ratio : float, optional
            Fraction of items to assign to the `"train"` subset when `subsets`
            is `None` (default is 0.8).
        extra_attrs : dict | None, optional
            Additional attributes to attach per item.

        Raises
        ------
        AssertionError
            If `image_paths` and `detections` have different lengths.
        """
        """
        image_paths と detections を IR に追加。
        - subsets を与えない場合は先頭から train_ratio で 'train'/'val' を自動付与
        - NormalizedDetections は px の XYXY/CONF/CLS 前提
        """
        assert len(image_paths) == len(detections)
        n = len(image_paths)

        if subsets is None:
            cut = int(n * train_ratio)
            subsets = ["train"] * cut + ["val"] * (n - cut)

        for img_path, det, subset in zip(image_paths, detections, subsets):
            # Use the file name stem as item_id (ensure uniqueness upstream if needed).
            item_id = Path(img_path).stem
            # Delegate to IR utility (handles XYXY→XYWH and score attributes internally).
            self.ir.add_item_from_normalized(
                ds=self.ir.ds,
                item_id=item_id,
                image=str(img_path),
                det=det,
                subset=subset,
                extra_attrs=extra_attrs,
            )

    # --- export ---

    def export_yolo_ultralytics(self, save_media: bool = True) -> None:
        """Export the dataset in Ultralytics YOLO layout.

        The export writes the standard directory structure (`images/`, `labels/`,
        split by subset) and any supplemental files produced by the Datumaro
        exporter when using the `"yolo_ultralytics"` format.

        Parameters
        ----------
        save_media : bool, optional
            If `True`, copy media files into the export structure (default True).
        """
        """
        Ultralytics YOLO 形式で出力（images/labels/train|val 等、data.yaml も出力される）
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Format name is 'yolo_ultralytics'.
        self.ir.ds.export(
            str(self.output_dir), format="yolo_ultralytics", save_media=save_media
        )
        # Note: When `save_media=True`, images are copied to the expected structure.

    def export_yaml(self) -> None:
        """Write a minimal `data.yaml` for YOLO training.

        The file includes train/val image directories, number of classes, and the
        class name list as recognized by this exporter.
        """
        class_names = self.class_names
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write(
                (
                    f"train: ./images/train\n"
                    f"val: ./images/val\n\n"
                    f"nc: {len(class_names)}\n"
                    f"names: {class_names}\n"
                )
            )

    def export_coco(self, save_media: bool = True) -> None:
        """Export the dataset in COCO format.

        Images are organized per subset directory by default unless the exporter
        is configured otherwise (e.g., with a `merge-images` option).

        Parameters
        ----------
        save_media : bool, optional
            If `True`, copy media files into the export structure (default True).
        """
        """
        COCO 形式で出力。subsetごとに出力され、画像は subset 別ディレクトリに保存（merge-imagesオプション未指定時）。 
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ir.ds.export(str(self.output_dir), format="coco", save_media=save_media)
        # COCO export typically separates images by subset unless configured otherwise.

    def export(self, save_media=True) -> None:
        """Export the dataset using the requested target format.

        This method ensures the output directories exist and dispatches to the
        specific exporter based on `self.fmt` (case-insensitive). For YOLO, it
        also writes a companion `data.yaml`.

        Parameters
        ----------
        save_media : bool, optional
            If `True`, copy media files into the export structure (default True).

        Raises
        ------
        ValueError
            If the requested format is not supported.
        """

        if self.fmt.lower() == "yolo":
            self.ensure_export_dirs()
            self.export_yolo_ultralytics(
                save_media=save_media
            )  # from datumaro to YOLO format
            self.export_yaml()  # YAML if needed (YOLO training dataset)
        elif self.fmt.lower() == "coco":
            self.export_coco(save_media=save_media)
        else:
            raise ValueError(f"unsupported format: {self.fmt}")
