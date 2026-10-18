# -*- coding: utf-8 -*-
"""ONNX Runtime YOLO segmentation detector."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort

# Real class order from nsfw-anime-medium-x1280.pt:
# 0 anus, 1 nipple, 2 penis, 3 vagina, 4 female face, 5 male face, 6 pubic hair
MODEL_CLASS_NAMES = [
    "anus",
    "nipple",
    "penis",
    "vagina",
    "female_face",
    "male_face",
    "pubic_hair",
]

# Classes we actually use for censoring.
ALLOWED_CLASS_KEYS = ["anus", "nipple", "penis", "vagina", "pubic_hair"]

# Canonical class id -> business key (used by ALLOWED_CLASS_KEYS / UI).
CANONICAL_KEYS = {
    "anus": "anus",
    "nipple": "nipple",
    "nipple_f": "nipple",   # legacy model naming (Automatic_coding)
    "penis": "penis",
    "vagina": "vagina",
    "pussy": "vagina",      # legacy model naming (Automatic_coding)
    "female_face": "female_face",
    "male_face": "male_face",
    "pubic_hair": "pubic_hair",
}


def _read_model_names(model_path: str) -> Optional[List[str]]:
    """Read the class-name metadata embedded by Ultralytics exports.

    Returns None when the metadata is missing (callers fall back to the
    canonical class list)."""
    try:
        import onnx

        m = onnx.load(model_path, load_external_data=False)
        for prop in m.metadata_props:
            if prop.key == "names":
                # {0: 'anus', 1: 'nipple', ...} (python-dict literal text)
                import ast
                names = ast.literal_eval(prop.value)
                ordered = [names[str(i)] if str(i) in names else names[i]
                           for i in sorted(
                               int(k) if str(k).isdigit() else k
                               for k in names)]
                return ordered
    except Exception:
        pass
    return None


@dataclass
class Detection:
    bbox: tuple  # (x1, y1, x2, y2) in original image coordinates
    mask: np.ndarray  # bool array, same size as original image
    class_id: int
    class_name: str  # internal key, e.g. "vagina"
    confidence: float


def _letterbox(
    img: np.ndarray,
    new_shape=(1280, 1280),
    color=(114, 114, 114),
):
    """Resize and pad image to new_shape while keeping aspect ratio."""
    shape = img.shape[:2]  # H, W
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # W, H
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, r, (dw, dh)


def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# Detection resolutions offered in UI (letterbox target for dynamic-shape models)
DETECT_SIZES = [640, 768, 832, 1024, 1280]


def pick_providers(use_gpu: bool = True) -> List[str]:
    """Choose ONNX Runtime providers: DirectML GPU with CPU fallback, or CPU only.

    DirectML works on any DirectX 12 GPU without installing CUDA/cuDNN, which
    makes it the right accelerator for this app. The provider list is only
    advisory - ONNXDetector retries with CPU-only if session creation fails."""
    if use_gpu:
        try:
            import onnxruntime as ort
            if "DmlExecutionProvider" in ort.get_available_providers():
                return ["DmlExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
    return ["CPUExecutionProvider"]


class ONNXDetector:
    """YOLO instance segmentation via ONNX Runtime."""

    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model_path = os.path.abspath(model_path)
        self.providers = providers or ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # Use all available CPU cores for intra-op parallelism
        sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
        sess_options.inter_op_num_threads = 1

        try:
            self.session = ort.InferenceSession(
                self.model_path, sess_options=sess_options, providers=self.providers
            )
        except Exception:
            # Provider unavailable in this environment (e.g. DirectML DLL
            # missing in a packaged build): fall back to CPU so the app still
            # works.
            self.providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(
                self.model_path, sess_options=sess_options, providers=self.providers
            )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.output_shapes = [o.shape for o in self.session.get_outputs()]

        # Determine fixed input size if possible
        shape = self.input_shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_h, self.input_w = shape[2], shape[3]
        else:
            self.input_h, self.input_w = 1280, 1280

        # Model class names: prefer the metadata embedded by the exporter,
        # fall back to the canonical 7-class list.
        meta_names = _read_model_names(self.model_path)
        self.model_class_names = meta_names or list(MODEL_CLASS_NAMES)
        # Business keys (canonical ids) in model class order.
        self.model_class_keys = [
            CANONICAL_KEYS.get(name, name) for name in self.model_class_names
        ]
        self.has_segmentation = any(
            len(out.shape) == 4 and out.shape[0] == 1
            for out in self.session.get_outputs()
        )

    def detect(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.7,
        allowed_classes: Optional[List[str]] = None,
        input_size: Optional[int] = None,
    ) -> List[Detection]:
        """Run segmentation inference on a BGR image.

        ``input_size`` sets the letterbox target for models with dynamic
        input shape (e.g. 640/832/1024/1280). Smaller sizes are much faster
        at the cost of accuracy on small regions. Fixed-shape models always
        use their native size.
        """
        if allowed_classes is None:
            allowed_classes = list(ALLOWED_CLASS_KEYS)

        # Fixed-shape model: must letterbox to its native input size.
        if (
            len(self.input_shape) == 4
            and isinstance(self.input_shape[2], int)
            and isinstance(self.input_shape[3], int)
        ):
            ih, iw = self.input_h, self.input_w
        else:
            size = int(input_size) if input_size else 1280
            ih = iw = size

        h0, w0 = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img, ratio, (dw, dh) = _letterbox(
            img_rgb, (ih, iw)
        )

        blob = img.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]  # NCHW

        outputs = self.session.run(self.output_names, {self.input_name: blob})

        # Identify prediction and prototype outputs
        pred = None
        proto = None
        for out, shape in zip(outputs, self.output_shapes):
            if out.ndim == 3 and out.shape[0] == 1:
                pred = out[0]
            elif out.ndim == 4 and out.shape[0] == 1:
                proto = out[0]

        if pred is None:
            raise RuntimeError("Unexpected ONNX output format: no prediction tensor")

        num_classes = len(self.model_class_keys)
        # pred shape: (4 + nc + nm, num_anchors)
        if pred.shape[0] < 4 + num_classes:
            # If class count is larger than expected, infer from actual rows
            num_classes = pred.shape[0] - 4
            if proto is not None:
                num_classes -= proto.shape[0]
        nm = pred.shape[0] - 4 - num_classes
        if nm < 0:
            nm = 0

        # Decode boxes/scores
        boxes_xywh = pred[:4].T  # (N, 4)
        scores = pred[4:4 + num_classes].T  # (N, nc)
        mask_coeffs = pred[4 + num_classes:].T  # (N, nm) if nm>0

        boxes_xyxy = _xywh2xyxy(boxes_xywh)

        detections: List[Detection] = []
        for cls_id, cls_name in enumerate(self.model_class_keys[:num_classes]):
            if cls_name not in allowed_classes:
                continue

            cls_scores = scores[:, cls_id]
            keep = np.where(cls_scores >= conf_threshold)[0]
            if len(keep) == 0:
                continue

            cand_boxes = boxes_xyxy[keep]
            cand_scores = cls_scores[keep]
            # NMS per class
            nms_indices = cv2.dnn.NMSBoxes(
                cand_boxes.tolist(),
                cand_scores.tolist(),
                score_threshold=conf_threshold,
                nms_threshold=iou_threshold,
            )
            if nms_indices is None or len(nms_indices) == 0:
                continue
            if isinstance(nms_indices, np.ndarray):
                nms_indices = nms_indices.flatten()

            for i in nms_indices:
                idx = keep[i]
                x1, y1, x2, y2 = cand_boxes[i]
                score = float(cand_scores[i])

                # Map box from letterboxed space to original image space
                x1o = (x1 - dw) / ratio
                y1o = (y1 - dh) / ratio
                x2o = (x2 - dw) / ratio
                y2o = (y2 - dh) / ratio
                x1o = max(0, min(w0 - 1, x1o))
                y1o = max(0, min(h0 - 1, y1o))
                x2o = max(0, min(w0 - 1, x2o))
                y2o = max(0, min(h0 - 1, y2o))
                bbox = (int(round(x1o)), int(round(y1o)), int(round(x2o)), int(round(y2o)))

                mask = np.zeros((h0, w0), dtype=np.bool_)
                if proto is not None and nm > 0:
                    coeffs = mask_coeffs[idx]  # (nm,)
                    proto_h, proto_w = proto.shape[1], proto.shape[2]
                    proto_flat = proto.reshape(nm, -1)
                    mask_small = _sigmoid(coeffs @ proto_flat).reshape(proto_h, proto_w)

                    # Box in letterboxed coordinates (clipped to input size)
                    lb_x1 = int(np.clip(round(x1), 0, iw - 1))
                    lb_y1 = int(np.clip(round(y1), 0, ih - 1))
                    lb_x2 = int(np.clip(round(x2), lb_x1 + 1, iw))
                    lb_y2 = int(np.clip(round(y2), lb_y1 + 1, ih))
                    bw = max(1, lb_x2 - lb_x1)
                    bh = max(1, lb_y2 - lb_y1)

                    # Crop the full-resolution mask at the box location, then resize
                    scale_x = proto_w / iw
                    scale_y = proto_h / ih
                    p_x1 = max(0, int(round(lb_x1 * scale_x)))
                    p_y1 = max(0, int(round(lb_y1 * scale_y)))
                    p_x2 = min(proto_w, int(round(lb_x2 * scale_x)))
                    p_y2 = min(proto_h, int(round(lb_y2 * scale_y)))

                    if p_x2 > p_x1 and p_y2 > p_y1:
                        mask_crop = mask_small[p_y1:p_y2, p_x1:p_x2]
                        mask_box = cv2.resize(
                            mask_crop, (bw, bh), interpolation=cv2.INTER_LINEAR
                        )
                        mask_box = mask_box > 0.5

                        full_lb = np.zeros((ih, iw), dtype=np.uint8)
                        full_lb[lb_y1:lb_y2, lb_x1:lb_x2] = mask_box.astype(np.uint8) * 255

                        # Crop content area and resize to original size
                        new_h = int(round(h0 * ratio))
                        new_w = int(round(w0 * ratio))
                        top = int(round(dh - 0.1))
                        left = int(round(dw - 0.1))
                        crop = full_lb[top:top + new_h, left:left + new_w]
                        mask_orig = cv2.resize(
                            crop, (w0, h0), interpolation=cv2.INTER_NEAREST
                        )
                        mask = mask_orig > 127
                else:
                    # Detection-only model (no segmentation prototype):
                    # use a filled rectangle over the box as the mask.
                    mask = np.zeros((h0, w0), dtype=np.bool_)
                    x1c, y1c = max(0, int(round(x1o))), max(0, int(round(y1o)))
                    x2c, y2c = min(w0 - 1, int(round(x2o))), min(h0 - 1, int(round(y2o)))
                    if x2c > x1c and y2c > y1c:
                        mask[y1c:y2c, x1c:x2c] = True

                detections.append(
                    Detection(
                        bbox=bbox,
                        mask=mask,
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=score,
                    )
                )

        return detections


def get_available_models(models_dir: str) -> List[str]:
    """Return ONNX files in models directory."""
    if not os.path.isdir(models_dir):
        return []
    return sorted(
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.lower().endswith(".onnx")
    )
