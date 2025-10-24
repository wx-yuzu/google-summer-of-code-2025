from typing import List, Optional

import torch
from torchvision.ops import nms

from ir.model_output_ir import NormalizedDetections


class PostProcessor:
    """Apply confidence filtering and NMS, with optional normalization/scaling."""

    # [TODO] Remove unintended/invalid bounding boxes (e.g., zero-area or out-of-bounds).
    def postprocess_yolo_result(
        self,
        result: torch.Tensor,
        conf_threshold: float = 0.0,
        iou_threshold: float = 0.0,
    ) -> List[torch.Tensor]:
        """Filter detections by confidence and suppress overlaps via NMS.

        Parameters
        ----------
        result : torch.Tensor
            Ultralytics-style detection result object (expects `.boxes.xyxy`,
            `.boxes.conf`, and `.boxes.cls` attributes).
        conf_threshold : float, optional
            Minimum confidence to keep a detection (default is 0.0).
        iou_threshold : float, optional
            IoU threshold for non-maximum suppression (default is 0.0).

        Returns
        -------
        list[torch.Tensor]
            A list-like triple ``[boxes, confs, classes]`` after filtering and NMS:
            - ``boxes``: ``Tensor`` of shape ``(K, 4)`` in absolute XYXY pixels.
            - ``confs``: ``Tensor`` of shape ``(K,)`` with per-box confidences.
            - ``classes``: ``Tensor`` of shape ``(K,)`` with class indices.

        Notes
        -----
        - This function assumes the input ``result`` comes from a detector that exposes
          the Ultralytics ``.boxes`` API.
        - Returned tensors are subsets of the original, selected by the NMS keep indices.
        """
        print(f"There is {len(result)} objects detected.")

        # Extract absolute-coordinate boxes, confidences, and class indices.
        boxes = result.boxes.xyxy  # (N, 4) absolute pixel coords
        confs = result.boxes.conf  # (N,)
        classes = result.boxes.cls  # (N,)

        # Filter by confidence threshold.
        mask = confs > conf_threshold
        boxes, confs, classes = boxes[mask], confs[mask], classes[mask]

        # Apply non-maximum suppression to reduce overlapping detections.
        keep = nms(boxes, confs, iou_threshold)
        return boxes[keep], confs[keep], classes[keep]

    def normalized_postprocess(
        self,
        det: NormalizedDetections,  # Ultralytics Results (single image)
        conf_threshold: float = 0.0,
        iou_threshold: float = 0.0,
        target_hw: Optional[
            tuple[int, int]
        ] = None,  # (H, W) size of the target/original image
    ) -> NormalizedDetections:
        """Filter, suppress, and optionally rescale normalized detections.

        The input is expected to be (or convertible to) ``NormalizedDetections``,
        where ``xyxy``, ``conf``, and ``cls`` are NumPy arrays on CPU. This method
        applies confidence filtering, NMS, and optional coordinate rescaling to
        match a target height/width.

        Parameters
        ----------
        det : NormalizedDetections | Any
            A detection structure compatible with ``NormalizedDetections``
            (fields: ``xyxy``, ``conf``, ``cls``, and ``input_shape``).
        conf_threshold : float, optional
            Minimum confidence to keep a detection (default is 0.0).
        iou_threshold : float, optional
            IoU threshold for NMS; if ``0.0`` or no boxes, NMS is skipped (default is 0.0).
        target_hw : tuple[int, int] | None, optional
            Target image size ``(H, W)`` to which coordinates may be scaled. If ``None``,
            coordinates are left as-is.

        Returns
        -------
        NormalizedDetections
            The filtered/suppressed (and possibly rescaled) detections. Arrays remain
            on CPU as NumPy arrays.

        Notes
        -----
        - Scaling uses ``det.input_shape`` as the source size and ``target_hw`` as
          the destination size; it is a safety measure in case shapes differ.
        """
        # 1) Confidence filtering.
        if conf_threshold > 0.0 and det.conf.size > 0:
            m = det.conf >= conf_threshold
            # With Pydantic v2 `validate_assignment`, assignments are validated on set.
            det.xyxy = det.xyxy[m]
            det.conf = det.conf[m]
            det.cls = det.cls[m]

        # 2) Non-maximum suppression.
        if iou_threshold > 0.0 and det.xyxy.shape[0] > 0:
            keep = (
                nms(
                    torch.from_numpy(det.xyxy),
                    torch.from_numpy(det.conf),
                    iou_threshold,
                )
                .cpu()
                .numpy()
            )
            det.xyxy = det.xyxy[keep]
            det.conf = det.conf[keep]
            det.cls = det.cls[keep]

        # 3) Rescale.
        if target_hw is not None and det.xyxy.shape[0] > 0:
            oh, ow = det.input_shape[:2]
            th, tw = target_hw
            if (oh, ow) != (th, tw) and ow > 0 and oh > 0:
                sx = tw / float(ow)
                sy = th / float(oh)
                det.xyxy[:, [0, 2]] *= sx
                det.xyxy[:, [1, 3]] *= sy

        return det
