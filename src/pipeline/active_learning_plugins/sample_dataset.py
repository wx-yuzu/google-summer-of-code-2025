from pathlib import Path

import datumaro as dm

from pipeline.active_learning_plugins.collect_bbox_by_item import (
    DatumaroBBoxIndexBuilder
)
from pipeline.active_learning_plugins.sampling_strategy.iou_sampling import (
    IouSamplingStrategy
)


def get_missing_datasets(
    image_path_list: list[str | Path],
    annotated_image_path_list: list[str | Path],
) -> list[str | Path]:
    """Compute missing image paths that are not yet annotated.

    The function returns a sorted list of elements that exist in
    ``image_path_list`` but not in ``annotated_image_path_list``.

    Parameters
    ----------
    image_path_list : Iterable
        All candidate image paths.
    annotated_image_path_list : Iterable
        Image paths that already have annotations.

    Returns
    -------
    list
        Sorted list of paths present in ``image_path_list`` but absent from
        ``annotated_image_path_list``.
    """
    return sorted(set(image_path_list) - set(annotated_image_path_list))


def sample_unlabeled_dataset_item(
    missing_image_paths: list[str | Path],
    n_sample: int = 10,
) -> list[str | Path]:
    """Return up to ``n_sample`` unlabeled image paths.

    If the number of ``missing_image_paths`` is smaller than ``n_sample``,
    the entire list is returned; otherwise the first ``n_sample`` elements
    (in current order) are returned.

    Parameters
    ----------
    missing_image_paths : list
        Candidate unlabeled image paths.
    n_sample : int, optional
        Desired number of samples to return, by default 10.

    Returns
    -------
    list
        Up to ``n_sample`` paths drawn from ``missing_image_paths``.
    """
    if len(missing_image_paths) < n_sample:
        return missing_image_paths
    else:
        return missing_image_paths[:n_sample]


def sample_dataset_item(
    src_annotation_path: str,
    src_annotation_format: str,
    target_annotation_path: str,
    target_annotation_format: str,
    n_sample: int = 10,
) -> dict[str, list[dm.Bbox]]:
    """Sample dataset items by IoU between a source and a target dataset.

    This function imports two datasets using Datumaro, builds per-item BBox
    indices for each, and then uses ``IouSamplingStrategy`` to select items
    according to the IoU-based ranking between source and target annotations.

    Signature
    ---------
    def sample_dataset_item(
        src_annotation_path: str | os.PathLike,
        src_annotation_format: str,
        target_annotation_path: str | os.PathLike,
        target_annotation_format: str,
        n_sample: int = 10
    ) -> dict[str, list[dm.Bbox]]  # depending on strategy's return type

    Parameters
    ----------
    src_annotation_path : str | os.PathLike
        Path to the source dataset annotations.
    src_annotation_format : str
        Datumaro-readable format for the source dataset (e.g., ``"coco"``).
    target_annotation_path : str | os.PathLike
        Path to the target dataset annotations.
    target_annotation_format : str
        Datumaro-readable format for the target dataset (e.g., ``"coco"``).
    n_sample : int, optional
        Number of items to sample, by default 10.

    Returns
    -------
    dict
        A mapping produced by ``IouSamplingStrategy.sample_idx`` that represents
        the selected items from the source dataset (the exact key/value types
        depend on the strategy implementation).
    """
    index_builder = DatumaroBBoxIndexBuilder()

    src_ds = index_builder.datumaro_import(src_annotation_path, src_annotation_format)

    tgt_ds = index_builder.datumaro_import(
        target_annotation_path, target_annotation_format
    )

    # Build per-item BBox indices for both datasets.
    src_dict_bbox = index_builder.bboxes_by_item(src_ds)
    tgt_dict_bbox = index_builder.bboxes_by_item(tgt_ds)

    # [FIXME] recieve sampling strategy in arguments
    strat = IouSamplingStrategy()

    # Select items using IoU ranking between source and target annotations.
    sampled_dict_bbox: list[int] = strat.sample_idx(
        src_datumaro_anns=src_dict_bbox,
        tgt_datumaro_anns=tgt_dict_bbox,
        n_sample=n_sample,
        use_class_metrics=False,
        respect_labels=True,
    )
    print(sampled_dict_bbox)

    return sampled_dict_bbox
