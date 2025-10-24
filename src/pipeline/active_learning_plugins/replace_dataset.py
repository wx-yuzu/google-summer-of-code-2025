from pathlib import Path

import datumaro as dm

from pipeline.active_learning_plugins.mock_get_ground_truth_label import (
    GroundTruthLabelHandler
)


def mock_get_labels(it, item_id: str):
    handler = GroundTruthLabelHandler()
    gt_dataset_item = handler.get_ground_truth(
        it,
        root_image_dir=Path("./dataset/wgisd_all_train/images"),
        root_label_dir=Path("./dataset/wgisd_all_train/labels"),
        image_filename=f"{item_id}.jpg",
        label_filename=f"{item_id}.txt",
    )
    return gt_dataset_item


def replace_then_export_dataset(
    output_dir: str, dataset: dm.Dataset, items_to_be_replaced: dict[str, list]
) -> None:
    """Replace or update items in a Datumaro-like dataset and export it.

    This function iterates over ``items_to_be_replaced`` and, for each (item_id, subset),
    rebuilds a ground-truth item using filesystem roots and file names, then inserts
    it into ``dataset`` using ``dataset.put``. Finally, it exports the dataset to the
    specified ``output_dir`` in the Ultralytics YOLO format.

    Signature
    ---------
    def replace_then_export_dataset(
        output_dir: str | os.PathLike,
        dataset: dm.Dataset,
        items_to_be_replaced: dict[tuple[str, str], list[dm.Annotation]],
    ) -> None

    Parameters
    ----------
    output_dir : str | os.PathLike
        Destination directory where the dataset will be exported.
    dataset : dm.Dataset
        Target dataset object that supports ``get(...)``, ``put(...)``, and ``export(...)``.
    items_to_be_replaced : dict[tuple[str, str], Any]
        Iterable mapping of ``(item_id, subset)`` to annotations or metadata used to
        reconstruct ground-truth items. The values are passed implicitly through the loop;
        only the keys (``item_id``, ``subset``) are used here to locate/replace items.

    Returns
    -------
    None
        This function performs in-place updates and writes output to disk.
    """

    for (item_id, subset), old_anns in items_to_be_replaced.items():
        # `dataset.remove` does not behave as intended in this context:
        # - It may implicitly set `subset="default"`.
        # - Since `dataset.put` overwrites by key (id, subset), use `put` instead to replace.
        it = dataset.get(item_id, subset=subset)
        gt_item = mock_get_labels(it, item_id)
        dataset.put(gt_item)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset.export(output_dir, format="yolo_ultralytics", save_media=True)
