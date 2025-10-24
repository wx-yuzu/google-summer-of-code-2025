from __future__ import annotations

from typing import Any, Literal, Sequence, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class NormalizedDetections(BaseModel):
    """Unified per-image detection IR in original-pixel XYXY coordinates.

    Attributes
    ----------
    xyxy : numpy.ndarray
        Array of shape ``(N, 4)`` with ``[x1, y1, x2, y2]`` in absolute pixels (float32).
    conf : numpy.ndarray
        Array of shape ``(N,)`` with confidence scores (float32).
    cls : numpy.ndarray
        Array of shape ``(N,)`` with class indices (int64).
    box_mode : Literal["XYXY_ABS", "XYWH_ABS"]
        Representation of boxes stored in this structure. Defaults to ``"XYXY_ABS"``.
    input_shape : numpy.ndarray | Sequence[int | float]
        Original model input shape or image shape, typically ``(H, W, ...)``.
    orig_img : numpy.ndarray
        Original image array associated with the detections.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow ndarray without forcing schema generation.
        validate_assignment=True,
    )

    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray
    box_mode: Literal["XYXY_ABS", "XYWH_ABS"] = "XYXY_ABS"  # Default to XYXY.
    input_shape: Union[np.ndarray, Sequence[Union[int, float]]]
    orig_img: np.ndarray

    # --------- Coercion (pre-processing) ----------
    @field_validator("xyxy", mode="before")
    @classmethod
    def _coerce_xyxy(cls, v: Any) -> np.ndarray:
        """Coerce ``xyxy`` into a float32 ndarray of shape ``(N, 4)``.

        If a flat array of 4 elements is provided, reshape to ``(1, 4)``.

        Raises
        ------
        ValueError
            If the provided array has neither 4 elements nor a second dimension of 4.
        """
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim == 1:
            # When a single box is provided as 4 elements.
            if arr.size != 4:
                raise ValueError("xyxy must have 4 elements or shape (N,4)")
            arr = arr.reshape(1, 4)
        return arr

    @field_validator("conf", mode="before")
    @classmethod
    def _coerce_conf(cls, v: Any) -> np.ndarray:
        """Coerce ``conf`` into a float32 ndarray."""
        return np.asarray(v, dtype=np.float32)

    @field_validator("cls", mode="before")
    @classmethod
    def _coerce_cls(cls, v: Any) -> np.ndarray:
        """Coerce ``cls`` into an int64 ndarray."""
        return np.asarray(v, dtype=np.int64)

    # --------- Shape validation (post-processing) ----------
    @model_validator(mode="after")
    def _check_shapes(self) -> "NormalizedDetections":
        """Validate internal array shapes and mutual consistency.

        Ensures that ``xyxy`` is ``(N, 4)`` and that ``conf`` and ``cls`` are
        both ``(N,)`` for the same ``N``.
        """
        if self.xyxy.ndim != 2 or self.xyxy.shape[1] != 4:
            raise ValueError("xyxy must be (N,4)")
        n = self.xyxy.shape[0]
        if self.conf.shape != (n,):
            raise ValueError(f"conf must be (N,), got {self.conf.shape}")
        if self.cls.shape != (n,):
            raise ValueError(f"cls must be (N,), got {self.cls.shape}")
        return self

    # --------- Construction from Ultralytics Results ----------
    @classmethod
    def from_ultralytics(cls, result) -> "NormalizedDetections":
        """Create a ``NormalizedDetections`` instance from an Ultralytics result.

        The ``result.boxes`` object is expected to expose ``xyxy``, ``conf``, and
        ``cls`` attributes. Tensors are moved to CPU and converted to NumPy, while
        non-tensor inputs pass through validators for normalization.

        Parameters
        ----------
        result : Any
            Ultralytics single-image inference output with ``boxes``, ``orig_shape``,
            and ``orig_img`` fields.

        Returns
        -------
        NormalizedDetections
            A validated detection container in absolute XYXY coordinates.
        """
        # Access through getattr to avoid hard dependencies on torch types.
        xyxy = getattr(result.boxes, "xyxy")
        conf = getattr(result.boxes, "conf")
        classes = getattr(result.boxes, "cls")

        # Accept torch/numpy/list; validators will finalize dtype/shape.
        try:
            xyxy = xyxy.detach().cpu().numpy()
            conf = conf.detach().cpu().numpy()
            classes = classes.detach().cpu().numpy()
        except AttributeError:
            pass

        ret = cls(
            xyxy=xyxy,
            conf=conf,
            cls=classes,
            box_mode="XYXY_ABS",
            input_shape=result.orig_shape,
            orig_img=result.orig_img,
        )
        return ret

    # [TODO] Implement from_coco if needed.
    # def from_coco(...)
