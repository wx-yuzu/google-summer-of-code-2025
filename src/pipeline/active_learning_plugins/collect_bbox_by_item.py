from pathlib import Path
from typing import List, Union

import datumaro as dm

PathLike = Union[str, Path]


class DatumaroBBoxIndexBuilder:
    """Provide utilities for working with Datumaro datasets.

    Responsibilities
    ----------------
    - Import a dataset from a designated directory and format.
    - Compute a stable comparison key (``path.stem``) for items.
    - Build a mapping from ``(stem, subset)`` to lists of BBox annotations.

    Notes
    -----
    This class is stateless (no internal mutable state). The order of BBoxes per
    image is preserved as in the source dataset.
    """

    def datumaro_import(
        self,
        annotation_dir: str,  # strでないとdatumaroの都合で落ちる
        annotation_format: str,
    ) -> dm.Dataset:
        """Import a Datumaro dataset from a directory and format.

        Parameters
        ----------
        annotation_dir : str
            Directory containing annotations and metadata (string path required by Datumaro).
        annotation_format : str
            Datumaro-readable format name (for example, ``"coco"``).

        Returns
        -------
        dm.Dataset
            The imported Datumaro dataset.
        """
        return dm.Dataset.import_from(annotation_dir, annotation_format)

    # ---- key = path.stem ----
    def stem_of(self, it: dm.DatasetItem) -> str:
        """Derive a comparison key from the underlying file name's stem.

        If the media file path is unavailable (for example, synthetic items), fall back
        to deriving the stem from ``item.id``.

        Parameters
        ----------
        it : dm.DatasetItem
            Target dataset item.

        Returns
        -------
        str
            The file-name stem (without extension).
        """
        p = getattr(it.media, "path", None)
        return Path(p).stem if p else Path(it.id).stem

    def bboxes_by_item(self, ds: dm.Dataset) -> dict[str, List[dm.Bbox]]:
        """Build a mapping from each dataset item to its list of bounding boxes.

        The key is a composite ``(path_stem, subset)`` tuple. The order of bounding
        boxes is preserved exactly as they appear in the source annotations.

        Parameters
        ----------
        ds : dm.Dataset
            Source dataset. Each item is expected to expose ``annotations`` and ``subset``.

        Returns
        -------
        Dict[tuple[str, str], List[dm.Bbox]]
            Mapping from ``(path_stem, subset)`` to the list of ``dm.Bbox`` found on that item.

        Notes
        -----
        The function annotation advertises ``Dict[str, List[dm.Bbox]]``, but the actual
        key used is a tuple ``(stem, subset)``. Adjust external typing expectations if
        you rely on this return type.
        """
        out: dict[tuple[str, str], List[dm.Bbox]] = {}
        for item in ds:
            boxes = [
                ann for ann in item.annotations if ann.type is dm.AnnotationType.bbox
            ]
            out[(self.stem_of(item), item.subset)] = (
                boxes  # keyは必要に応じて item.media.path.stem などに変更可
            )
        return out


def collect_image_path_from_datumaro(
    annotation_dir,
    annotation_format,
):
    """Collect file-system paths of image media from a Datumaro dataset.

    Only items whose media is an instance of ``dm.Image`` are considered. If a media
    object has no backing path (``None``), it is skipped.

    Parameters
    ----------
    annotation_dir : str | os.PathLike
        Directory containing annotations and metadata.
    annotation_format : str
        Datumaro-readable format name (for example, ``"coco"``).

    Returns
    -------
    List[pathlib.Path]
        A list of image paths as ``Path`` objects, in dataset iteration order.
    """

    ds = dm.Dataset.import_from(annotation_dir, annotation_format)
    image_paths = []
    for item in ds:
        if isinstance(
            item.media, dm.Image
        ):  # 画像メディアのみ対象（Video などが混ざる可能性に備える）
            p = item.media.path  # 画像のパス（絶対/相対どちらの場合もある）
            if p is not None:
                image_paths.append(Path(p))

    return image_paths
