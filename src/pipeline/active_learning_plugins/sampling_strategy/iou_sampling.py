import datumaro as dm
import numpy as np

from evaluation.evaluator import IouEvaluator
from pipeline.active_learning_plugins.sampling_strategy.base import (
    BaseSamplingStrategy
)


class IouSamplingStrategy(BaseSamplingStrategy):
    """
    Select data samples by Intersection over Union (IoU).

    Concept:
        - Computes IoU between source and target bounding-box annotations per item.
        - Intended behavior (per description): pick the worst-k (i.e., lowest IoU) samples.
        - Current implementation note: the sorting is descending (largest to smallest) and
          therefore selects the best-k highest IoUs. This may contradict the intention.
          See the comment near `np.argsort(...)` for details.

    Types and semantics:
        - Uses `dm.Bbox` for bounding boxes (assumed to carry xywh + optional label).
        - IoU per item is computed via `IouEvaluator.calc_iou_from_labelled_xywh(...)`
          and extracted from the returned dict under key `'iou'` (scalar-like, `.item()`).

    Returns:
        - The method `sample_idx(...)` returns a `dict[str, list[dm.Bbox]]`:
          a subset mapping from annotation path (str) to list of source `dm.Bbox`es,
          consisting of the selected k items according to the IoU criterion.
    """

    def sample_idx(
        self,
        src_datumaro_anns: dict[str, list[dm.Bbox]],
        tgt_datumaro_anns: dict[str, list[dm.Bbox]],
        n_sample,
        use_class_metrics=False,
        respect_labels=True,
    ) -> dict[str, list[dm.Bbox]]:
        """
        Args:
          src_datumaro_anns (dict[str, list[dm.Bbox]]):
              Mapping from source annotation path to a list of source bounding boxes.
              Each dm.Bbox is expected as (x, y, w, h) and may include a class label.
          tgt_datumaro_anns (dict[str, list[dm.Bbox]]):
              Mapping from target annotation path to a list of target bounding boxes.
              If a source path is missing here, a single zero-area bbox is used as fallback.
          n_sample (int):
              Number of items to select based on IoU ranking.
          use_class_metrics (bool, optional):
              If True, IoU computation may include class-aware metrics (depends on evaluator).
          respect_labels (bool, optional):
              If True, IoU is computed while respecting class labels; otherwise class labels may be ignored.

        Returns:
          dict[str, list[dm.Bbox]]:
              Subset of the source annotations containing the selected items
              (path -> list of dm.Bbox) according to the IoU criterion.
        """

        ev = IouEvaluator()
        ious = []

        # [TODO] include skipped annotations
        # Iterate over all source items; for each, fetch the corresponding target boxes by path.
        for src_ann_path, src_ann_bbox in src_datumaro_anns.items():
            # If target path is missing, default to a single zero-area bbox (IoU->0 vs non-zero area).
            tgt_ann_bbox = tgt_datumaro_anns.get(
                src_ann_path, [dm.Bbox(0, 0, 0, 0, label=0)]
            )

            # Compute IoU between labelled xywh boxes.
            # Expected output: dict with key 'iou' (tensor-like/float), compatible with `.item()`.
            out = ev.calc_iou_from_labelled_xywh(
                src_ann_bbox,
                tgt_ann_bbox,
                use_class_metrics=use_class_metrics,
                respect_labels=respect_labels,
            )
            ious.append(out["iou"].item())  # Store scalar IoU for ranking.

        # Convert to ndarray for efficient argsort-based selection.
        np_ious = np.asarray(ious)

        # Rank items by IoU.
        sample_idxs = np.argsort(np_ious)[::-1][:n_sample].tolist()

        # Recover original (key, value) pairs for the selected indices.
        items = list(src_datumaro_anns.items())
        picked_items = [items[idx] for idx in sample_idxs]
        picked_dict = dict(picked_items)

        return picked_dict
