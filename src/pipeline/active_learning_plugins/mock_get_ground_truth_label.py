from pathlib import Path

import datumaro as dm
from datumaro.components.media import Image as DmImage
from PIL import Image as PILImage


class GroundTruthLabelHandler:
    """Provide helpers to construct Datumaro items from ground-truth (GT) labels.

    Notes
    -----
    - In this class GT is to use mock human correction.
    """

    def get_ground_truth(
        self,
        it,
        root_image_dir: Path,
        root_label_dir: Path,
        image_filename: str,
        label_filename: str,
        import_format="yolo",
    ):
        """Load a single GT sample and return it as a Datumaro item.

        The function locates the image/label files under the provided roots and,
        currently for ``import_format="yolo"`` only, converts them into a
        ``dm.DatasetItem`` that mirrors the given item's id/subset.

        Parameters
        ----------
        it : dm.DatasetItem
            Reference item whose ``id`` and ``subset`` will be reused.
        root_image_dir : pathlib.Path
            Directory containing GT images.
        root_label_dir : pathlib.Path
            Directory containing GT labels (YOLO text files).
        image_filename : str
            Image file name (for example, ``"000123.jpg"``).
        label_filename : str
            Label file name (for example, ``"000123.txt"``).
        import_format : str, optional
            Expected GT format. Only ``"yolo"`` is supported at the moment.

        Returns
        -------
        dm.DatasetItem
            A new DatasetItem with media and annotations taken from the GT files.

        Raises
        ------
        NotImplementedError
            If ``import_format`` is not ``"yolo"``.
        """
        # Find GT paths and import them into Datumaro format based on `import_format`.
        if import_format != "yolo":
            raise NotImplementedError(
                "GT format must be in yolo format. (This error should be called only when ground truth labels are hidden on purpose.)"
            )

        image_path = root_image_dir / image_filename
        label_path = root_label_dir / label_filename

        return self.load_single_yolo_to_dm_item(
            it,
            img_path=image_path,
            label_path=label_path,
        )

    def load_single_yolo_to_dm_item(
        self, it, img_path: Path, label_path: Path
    ) -> dm.DatasetItem:
        """Convert a single YOLO label file and image into a Datumaro item.

        The YOLO label file is expected to contain one object per line in the form:
        ``<class> <cx> <cy> <w> <h>``, where coordinates are *normalized* to [0, 1]
        relative to image width/height. This function denormalizes them to pixel space
        and constructs ``dm.Bbox`` annotations.

        Signature
        ---------
        def load_single_yolo_to_dm_item(
            self,
            it: dm.DatasetItem,
            img_path: pathlib.Path,
            label_path: pathlib.Path,
        ) -> dm.DatasetItem

        Parameters
        ----------
        it : dm.DatasetItem
            Reference item providing ``id`` and ``subset`` for the returned item.
        img_path : pathlib.Path
            Path to the image file.
        label_path : pathlib.Path
            Path to the YOLO label text file.

        Returns
        -------
        dm.DatasetItem
            A DatasetItem with media set to ``img_path`` and YOLO-derived bboxes.

        Notes
        -----
        The attribute ``{"human_corrected": True}`` is attached to the returned item
        to indicate that annotations are considered ground truth / curated.
        """

        # Read image size in pixels.
        W, H = PILImage.open(img_path).size

        # Parse YOLO labels and convert to dm.Bbox (denormalize to pixel coordinates).
        bboxes = []
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                cls, cx, cy, w, h = line.split()
                cls = int(cls)
                cx, cy, w, h = map(float, (cx, cy, w, h))
                x = (cx - w / 2.0) * W
                y = (cy - h / 2.0) * H
                bboxes.append(dm.Bbox(x, y, w * W, h * H, label=cls))

        item = dm.DatasetItem(
            id=it.id,
            subset=it.subset,
            media=DmImage.from_file(str(img_path)),
            annotations=bboxes,
            attributes={"human_corrected": True},
        )

        return item  # Return the Datumaro-formatted item.
