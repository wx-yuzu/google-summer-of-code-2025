from typing import Any, Dict, List, Tuple

BBox = Tuple[float, float, float, float]
Det = Tuple[BBox, int, float]
# (bbox, class_id, confidence_score) — confidence は TP/FP の判定用で後でソート
GT = Tuple[BBox, int]

from typing import Any, Dict, List, Tuple

import datumaro as dm
import torch
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.functional.detection import (  # pairwise行列/平均に使う（関数API）
    intersection_over_union
)


class IouEvaluator:
    # ---------- 変換: dm.Bbox( XYWH[pixel] ) → torchmetrics用( XYXY ) ----------
    def _bboxes_to_xyxy_and_labels(
        self, boxes: List[dm.Bbox]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        dm.Bbox のリストを torchmetrics 入力に変換する。
        戻り:
        xyxy: (N,4) float32 tensor  [x1,y1,x2,y2]
        labels: (N,) int64 tensor   （そのまま dm.Bbox.label を使用）
        """
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
                (0,), dtype=torch.long
            )

        xyxy = torch.tensor(
            [[b.x, b.y, b.x + b.w, b.y + b.h] for b in boxes],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [int(b.label) if b.label is not None else -1 for b in boxes],
            dtype=torch.long,
        )
        return xyxy, labels

    # ---------- 変換: dm.Bbox( XYWH[pixel] ) → torchmetrics用( XYXY ) ----------
    def _bboxes_to_xywh_and_labels(
        self, boxes: List[dm.Bbox]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        dm.Bbox のリストを torchmetrics 入力に変換する。
        戻り:
        xyxy: (N,4) float32 tensor  [x1,y1,x2,y2]
        labels: (N,) int64 tensor   （そのまま dm.Bbox.label を使用）
        """
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
                (0,), dtype=torch.long
            )

        xyxy = torch.tensor(
            [[b.x, b.y, b.w, b.h] for b in boxes],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [int(b.label) if b.label is not None else -1 for b in boxes],
            dtype=torch.long,
        )
        return xyxy, labels

    # ---------- 画像1枚分: ラベルを加味して IoU を集計（平均 or クラス別） ----------
    def calc_iou_from_labelled_xywh(
        self,
        src_boxes: List[dm.Bbox],
        tgt_boxes: List[dm.Bbox],
        use_class_metrics: bool = False,
        respect_labels: bool = True,
    ) -> Dict[str, Any]:
        """
        ラベルを尊重してIoUを集計する（torchmetricsのMetric API）。
        戻り:
        {'iou': Tensor}  ほか、class_metrics=Trueなら 'iou/cl_{id}': Tensor も含む
        備考:
        - preds/target は「画像ごとに1辞書」のリストで渡すのがtorchmetricsの想定。
        - 既定 box_format='xywh'。dm.Bbox→xywhへ変換済みのテンソルを渡す。

        use_class_metrics=False: return {'iou': tensor(0.1957)}
        use_class_metrics=True: return {'iou': tensor(0.1957), 'iou/cl_0': tensor(0.3913), 'iou/cl_1': tensor(0.)}
        """

        src_xywh, src_labels = self._bboxes_to_xywh_and_labels(src_boxes)
        tgt_xywh, tgt_labels = self._bboxes_to_xywh_and_labels(tgt_boxes)

        metric = IntersectionOverUnion(
            box_format="xywh",
            iou_threshold=None,
            class_metrics=use_class_metrics,
            respect_labels=respect_labels,
        )  # :contentReference[oaicite:2]{index=2}
        preds = [{"boxes": src_xywh, "labels": src_labels}]
        target = [{"boxes": tgt_xywh, "labels": tgt_labels}]
        return metric(
            preds, target
        )  # 例: {'iou': tensor(0.78)} / {'iou': ..., 'iou/cl_0': ..., ...}


def test_calc_iou():
    a = [dm.Bbox(10, 10, 20, 20, label=0), dm.Bbox(40, 40, 10, 10, label=1)]
    b = [dm.Bbox(15, 15, 20, 20, label=0), dm.Bbox(200, 200, 5, 5, label=1)]

    ev = IouEvaluator()
    out = ev.calc_iou_from_labelled_xywh(
        a, b, use_class_metrics=False, respect_labels=True
    )
    print("labeled IoU summary:\n", out)
