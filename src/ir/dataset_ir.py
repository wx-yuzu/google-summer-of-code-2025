from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import datumaro as dm
import numpy as np
from pydantic import BaseModel, field_validator

from ir.model_output_ir import NormalizedDetections


# ---------- Pydantic config ----------
class DatasetIRConfig(BaseModel):
    """Define global configuration for DatasetIR.

    Attributes
    ----------
    categories : list[str]
        List of class names in label-index order.
    default_subset : str
        Default subset name to use when none is provided. Defaults to ``"train"``.
    yolo_format : str
        Datumaro export format string for Ultralytics YOLO.
    box_mode : Literal["XYXY_ABS", "XYWH_ABS"]
        Expected internal box mode for detections.
    """

    categories: List[str]
    default_subset: str = "train"
    yolo_format: str = "yolo_ultralytics"  # Datumaro's Ultralytics target
    box_mode: Literal["XYXY_ABS", "XYWH_ABS"] = "XYXY_ABS"

    @field_validator("categories")
    @classmethod
    def _validate_categories(cls, v: List[str]) -> List[str]:
        """Validate that categories are non-empty strings and unique.

        Parameters
        ----------
        v : list[str]
            Category names to validate.

        Returns
        -------
        list[str]
            The validated category list.

        Raises
        ------
        ValueError
            If any category is empty/non-string or if duplicates are present.
        """
        if not v or any(not isinstance(n, str) or not n.strip() for n in v):
            raise ValueError("categories must be non-empty strings")
        if len(set(v)) != len(v):
            raise ValueError("categories must be unique")
        return v


class DatasetIR:
    """Wrap Datumaro with a stable intermediate representation (IR).

    Notes
    -----
    - Helper methods add items for classification, detection, and segmentation,
      and provide basic validators and exporters.
    """

    def __init__(self, cfg: DatasetIRConfig):
        """Initialize the IR with label categories and image media type.

        Parameters
        ----------
        cfg : DatasetIRConfig
            Global configuration including categories and defaults.
        """
        self.cfg = cfg
        label_cats = dm.LabelCategories.from_iterable(cfg.categories)
        # v1.11+: media_type is required for manual Dataset construction
        self.ds = dm.Dataset(
            categories={dm.AnnotationType.label: label_cats},
            media_type=dm.Image,  # image dataset
        )

    # --- helpers ---
    @staticmethod
    def _to_media(img: Union[str, np.ndarray]) -> dm.Image:
        """Convert a filepath or ndarray into a Datumaro Image media.

        Parameters
        ----------
        img : str | numpy.ndarray
            Either a file path (string) or an in-memory HWC uint8 array.

        Returns
        -------
        dm.Image
            A Datumaro media object created from the input.

        Raises
        ------
        TypeError
            If ``img`` is neither a string path nor a NumPy array.

        Notes
        -----
        - Use factory methods introduced in v1.11.
        """
        if isinstance(img, str):
            # File path.
            return dm.Image.from_file(path=img)
        if isinstance(img, np.ndarray):
            # In-memory HWC uint8 ndarray.
            return dm.Image.from_numpy(data=img)
        raise TypeError(f"Unsupported image type: {type(img)}")

    def add_classification(
        self,
        item_id: str,
        image: Union[str, np.ndarray],
        label_id: int,
        subset: Optional[str] = None,
        attrs: Optional[dict] = None,
    ) -> None:
        """Add a single classification item to the dataset.

        Parameters
        ----------
        item_id : str
            Item identifier (e.g., file stem).
        image : str | numpy.ndarray
            Image source as a file path or HWC uint8 array.
        label_id : int
            Label index referencing ``self.cfg.categories``.
        subset : str | None, optional
            Subset name (defaults to ``cfg.default_subset`` when ``None``).
        attrs : dict | None, optional
            Optional annotation attributes to attach.
        """
        ann = dm.Label(label=label_id, attributes=attrs or {})
        self.ds.put(
            dm.DatasetItem(
                id=item_id,
                subset=subset or self.cfg.default_subset,
                media=self._to_media(image),  # media=dm.Image(...)
                annotations=[ann],
            )
        )

    def add_detections(
        self,
        item_id: str,
        image: Union[str, np.ndarray],
        boxes: Sequence[Tuple[int, Tuple[float, float, float, float], Optional[dict]]],
        subset: Optional[str] = None,
    ) -> None:
        """Add a detection item with XYWH pixel-space boxes.

        Parameters
        ----------
        item_id : str
            Item identifier.
        image : str | numpy.ndarray
            Image source as a file path or HWC uint8 array.
        boxes : Sequence[Tuple[int, Tuple[float, float, float, float], dict | None]]
            Sequence of ``(label_id, (x, y, w, h), attributes)`` tuples.
        subset : str | None, optional
            Subset name (defaults to ``cfg.default_subset`` when ``None``).

        Notes
        -----
        Boxes with non-positive width/height are still added; downstream validators
        (e.g., ``find_invalid_bboxes``) can flag them later.
        """
        anns = []
        for label_id, (x, y, w, h), attrs in boxes:
            if w <= 0 or h <= 0:
                # Keep behavior simple: still put; the validator can catch later.
                pass
            anns.append(dm.Bbox(x, y, w, h, label=label_id, attributes=attrs or {}))
        self.ds.put(
            dm.DatasetItem(
                id=item_id,
                subset=subset or self.cfg.default_subset,
                media=self._to_media(image),
                annotations=anns,
            )
        )

    def add_masks(
        self,
        item_id: str,
        image: Union[str, np.ndarray],
        masks: Sequence[Tuple[int, np.ndarray, Optional[dict]]],
        subset: Optional[str] = None,
    ) -> None:
        """Add a segmentation item with binary masks.

        Parameters
        ----------
        item_id : str
            Item identifier.
        image : str | numpy.ndarray
            Image source as a file path or HWC uint8 array.
        masks : Sequence[Tuple[int, numpy.ndarray, dict | None]]
            Sequence of ``(label_id, mask_array, attributes)`` tuples.
        subset : str | None, optional
            Subset name (defaults to ``cfg.default_subset`` when ``None``).
        """
        anns = []
        for label_id, mask_arr, attrs in masks:
            anns.append(dm.Mask(image=mask_arr, label=label_id, attributes=attrs or {}))
        self.ds.put(
            dm.DatasetItem(
                id=item_id,
                subset=subset or self.cfg.default_subset,
                media=self._to_media(image),
                annotations=anns,
            )
        )

    # --- validators (optional) ---

    def find_empty_items(self) -> List[str]:
        """Return item IDs that have zero annotations."""
        return [it.id for it in self.ds if not it.annotations]

    def find_invalid_bboxes(self) -> List[str]:
        """Return item IDs that contain invalid bounding boxes.

        Notes
        -----
        A bbox is considered invalid here if ``w <= 0`` or ``h <= 0``.
        """
        bad = []
        for it in self.ds:
            for ann in it.annotations:
                if isinstance(ann, dm.Bbox) and (ann.w <= 0 or ann.h <= 0):
                    bad.append(it.id)
                    break
        return bad

    # --- export / import ---

    def export_yolo(self, out_dir: str, save_media: bool = True) -> None:
        """Export the dataset in the configured Ultralytics YOLO format.

        Parameters
        ----------
        out_dir : str
            Destination directory.
        save_media : bool, optional
            If ``True``, copy media files to the export structure.
        """
        self.ds.export(out_dir, format=self.cfg.yolo_format, save_media=save_media)

    def export_coco(self, out_dir: str, save_media: bool = True) -> None:
        """Export the dataset in COCO format.

        Parameters
        ----------
        out_dir : str
            Destination directory.
        save_media : bool, optional
            If ``True``, copy media files to the export structure.
        """
        self.ds.export(out_dir, format="coco", save_media=save_media)

    @classmethod
    def import_from(cls, src: str, fmt: str, cfg: DatasetIRConfig) -> "DatasetIR":
        """Import a Datumaro dataset and wrap it in DatasetIR.

        Parameters
        ----------
        src : str
            Source dataset directory.
        fmt : str
            Datumaro import format.
        cfg : DatasetIRConfig
            Configuration whose defaults (e.g., subset, YOLO format) are reused.

        Returns
        -------
        DatasetIR
            An IR instance populated with items and categories from the source.

        Notes
        -----
        - Label definitions are taken from the imported dataset to keep indices
          and names aligned with the source.
        """
        imported = dm.Dataset.import_from(src, fmt)
        # Align label definitions with the imported dataset.
        label_cats = imported.categories().get(dm.AnnotationType.label)
        cats = [c.name for c in label_cats]
        ir = cls(
            DatasetIRConfig(
                categories=cats,
                default_subset=cfg.default_subset,
                yolo_format=cfg.yolo_format,
            )
        )
        for item in imported:
            ir.ds.put(item)
        return ir

    def _assert_label(self, label_id: int):
        """Assert that a label index is within the valid category range.

        Parameters
        ----------
        label_id : int
            Label index to validate.

        Raises
        ------
        ValueError
            If the label index is out of range.
        """
        n = len(self.cfg.categories)
        if not (0 <= label_id < n):
            raise ValueError(f"label_id {label_id} out of range [0, {n})")

    @staticmethod
    def _xyxy_to_xywh(x1, y1, x2, y2):
        """Convert XYXY coordinates to XYWH (all floats)."""
        return float(x1), float(y1), float(x2 - x1), float(y2 - y1)

    def add_detections_xyxy_pixels(
        self,
        item_id: str,
        image: Union[str, np.ndarray],
        xyxy: np.ndarray,  # shape (N,4), pixel coords
        classes: np.ndarray,  # shape (N,), int
        scores: Optional[np.ndarray] = None,  # shape (N,), float
        subset: Optional[str] = None,
        extra_attrs: Optional[Dict[str, object]] = None,
    ) -> None:
        """Add detections given as XYXY pixel coordinates.

        The IR stores boxes internally as XYWH pixels. If ``scores`` are provided,
        they are recorded under ``attributes['score']``.

        Parameters
        ----------
        item_id : str
            Item identifier.
        image : str | numpy.ndarray
            Image source as a file path or HWC uint8 array.
        xyxy : numpy.ndarray
            Array of shape ``(N, 4)`` in pixel coordinates ``[x1, y1, x2, y2]``.
        classes : numpy.ndarray
            Array of shape ``(N,)`` with integer label indices.
        scores : numpy.ndarray | None, optional
            Array of shape ``(N,)`` with detection confidences.
        subset : str | None, optional
            Subset name (defaults to ``cfg.default_subset`` when ``None``).
        extra_attrs : dict[str, object] | None, optional
            Additional attributes to merge into each annotation.

        Raises
        ------
        TypeError
            If input array shapes are invalid or inconsistent.
        ValueError
            If a label index is out of range.

        Notes
        -----
        - IR unifies boxes as XYWH in pixels. Scores are stored under attributes['score'].
        """
        if not isinstance(xyxy, np.ndarray) or xyxy.ndim != 2 or xyxy.shape[1] != 4:
            raise TypeError("xyxy must be (N,4) ndarray")
        if (
            not isinstance(classes, np.ndarray)
            or classes.ndim != 1
            or classes.shape[0] != xyxy.shape[0]
        ):
            raise TypeError("classes must be (N,) ndarray matching xyxy")
        if scores is not None and (
            not isinstance(scores, np.ndarray) or scores.shape[0] != xyxy.shape[0]
        ):
            raise TypeError("scores must be (N,) ndarray matching xyxy")

        anns = []
        for i in range(xyxy.shape[0]):
            label_id = int(classes[i])
            self._assert_label(label_id)
            x1, y1, x2, y2 = map(float, xyxy[i])
            x, y, w, h = self._xyxy_to_xywh(x1, y1, x2, y2)
            attrs = dict(extra_attrs or {})
            if scores is not None:
                attrs["score"] = float(scores[i])
            anns.append(dm.Bbox(x, y, w, h, label=label_id, attributes=attrs))

        self.ds.put(
            dm.DatasetItem(
                id=item_id,
                subset=subset or self.cfg.default_subset,
                media=self._to_media(image),
                annotations=anns,
            )
        )

    def _xyxy_to_xywh(self, box: np.ndarray) -> tuple[float, float, float, float]:
        """Convert a single XYXY box array to XYWH tuple of floats."""
        x1, y1, x2, y2 = map(float, box)
        return x1, y1, (x2 - x1), (y2 - y1)

    def build_dataset(
        self, categories: List[str], default_subset: str = "train"
    ) -> dm.Dataset:
        """Create an empty Datumaro dataset with category and media definitions.

        Parameters
        ----------
        categories : list[str]
            Class name list to register as label categories.
        default_subset : str, optional
            Default subset name to store on the dataset object, by default ``"train"``.

        Returns
        -------
        dm.Dataset
            A new dataset with label categories and image media type set.

        Notes
        -----
        - The dataset keeps a convenience attribute ``_default_subset`` for downstream
          callers that rely on a default subset value.
        - Datumaro supports many formats for I/O and validation.
        """
        label_cats = dm.LabelCategories.from_iterable(categories)
        ds = dm.Dataset(
            categories={dm.AnnotationType.label: label_cats},
            media_type=dm.Image,  # Image dataset.
        )
        ds._default_subset = default_subset  # Keep a convenience default subset.
        return ds

    def add_item_from_normalized(
        self,
        ds: dm.Dataset,
        item_id: str,
        image: Union[str, np.ndarray],
        det: NormalizedDetections,
        subset: Optional[str] = None,
        extra_attrs: Optional[Dict[str, object]] = None,
    ) -> None:
        """Insert an item from normalized detections (XYXY px) into a dataset.

        The method converts XYXY pixel boxes to Datumaro ``Bbox`` (XYWH), attaches
        confidence under ``attributes['score']``, and writes the item into ``ds``.

        Parameters
        ----------
        ds : dm.Dataset
            Target Datumaro dataset to update.
        item_id : str
            Item identifier to assign.
        image : str | numpy.ndarray
            Image source as a file path or HWC uint8 array.
        det : NormalizedDetections
            Detection container with ``xyxy``, ``conf``, ``cls``, and ``box_mode``.
        subset : str | None, optional
            Subset name (falls back to ``ds._default_subset`` or ``"train"``).
        extra_attrs : dict[str, object] | None, optional
            Extra attributes merged into each bbox annotation.
        """
        """
        Ingest L1 → L2. Convert XYXY pixels to Datumaro Bbox (XYWH) and store score
        under attributes. When exporting to COCO, attributes['score'] is preserved.
        """
        anns: List[dm.Annotation] = []

        # XYXY -> XYWH (Datumaro Bbox uses XYWH).
        if det.box_mode == "XYWH_ABS":
            xywh = det.xyxy.astype(np.float32)
        else:
            xywh = np.vstack([self._xyxy_to_xywh(b) for b in det.xyxy]).astype(
                np.float32
            )

        for i in range(xywh.shape[0]):
            x, y, w, h = map(float, xywh[i])
            label_id = int(det.cls[i])
            attrs = dict(extra_attrs or {})
            attrs["score"] = float(det.conf[i])
            anns.append(dm.Bbox(x, y, w, h, label=label_id, attributes=attrs))

        ds.put(
            dm.DatasetItem(
                id=item_id,
                subset=subset or getattr(ds, "_default_subset", "train"),
                media=self._to_media(image),
                annotations=anns,
            )
        )
