"""Pretrained perception adapters. No clustering, Kalman filtering or assignment solver.

VoxelNeXt uses its authors' nuScenes checkpoint. SAM 2.1 is executed through the
pinned Ultralytics implementation. Neither checkpoint is trained on these bags.
"""

from pathlib import Path
import logging
import numpy as np


class VoxelNeXtDetector:
    def __init__(self, cfg):
        import torch
        from easydict import EasyDict
        from pcdet.config import cfg_from_yaml_file
        from pcdet.datasets.dataset import DatasetTemplate
        from pcdet.models import build_network

        self.torch = torch
        root = Path("/opt/voxelnext/tools")
        config = EasyDict()
        # Upstream resolves base YAMLs relative to tools/. Resolve that reference
        # explicitly, rather than changing the process working directory.
        import yaml

        file = root / "cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"
        raw = yaml.safe_load(file.read_text())
        raw["DATA_CONFIG"]["_BASE_CONFIG_"] = str(
            root / raw["DATA_CONFIG"]["_BASE_CONFIG_"]
        )
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as temp:
            yaml.safe_dump(raw, temp)
            temp.flush()
            cfg_from_yaml_file(temp.name, config)
        self.names = config.CLASS_NAMES
        self.config = config
        self.dataset = DatasetTemplate(
            config.DATA_CONFIG,
            self.names,
            training=False,
            root_path=Path("/tmp"),
            logger=logging.getLogger("voxelnext"),
        )
        self.model = build_network(config.MODEL, len(self.names), self.dataset)
        self.model.load_params_from_file(
            cfg["weights"], logging.getLogger("voxelnext"), to_cpu=True
        )
        self.model.cuda().eval()
        self.threshold = cfg.get("confidence", 0.25)
        self.intensity_scale = cfg.get("intensity_scale", 255.0)
        self.sample = None
        self.model.dense_head.register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        self.sample = inputs[0]["encoded_spconv_tensor"]

    def detect(self, raw):
        from pcdet.models import load_data_to_gpu

        torch = self.torch
        if raw.ndim != 2 or raw.shape[1] != 4:
            raise ValueError(
                "VoxelNeXt requires XYZ and recorded intensity; refusing fabricated intensity"
            )
        points = np.asarray(raw, dtype=np.float32)
        points = points[np.isfinite(points).all(1)].copy()
        points[:, 3] = np.clip(points[:, 3] / self.intensity_scale, 0, 1)
        # Checkpoint expects XYZ, intensity, time-lag. These bags supply one sweep:
        # zero time-lag denotes current points, not invented historical sweeps.
        points = np.column_stack((points, np.zeros(len(points), np.float32)))
        data = self.dataset.prepare_data({"points": points, "frame_id": 0})
        batch = self.dataset.collate_batch([data])
        load_data_to_gpu(batch)
        with torch.inference_mode():
            predictions, _ = self.model(batch)
            p = predictions[0]
            keep = p["pred_scores"] >= self.threshold
            boxes = p["pred_boxes"][keep].cpu().numpy()
            scores = p["pred_scores"][keep].cpu().numpy()
            labels = p["pred_labels"][keep].cpu().numpy().astype(int) - 1
            # Sparse backbone BEV features nearest to each regressed box center.
            sparse = self.sample
            xy = sparse.indices[:, [2, 1]].float()
            xy = xy * 8 * torch.tensor([0.075, 0.075], device="cuda") + torch.tensor(
                [-54.0, -54.0], device="cuda"
            )
            if len(boxes):
                nearest = torch.cdist(
                    torch.as_tensor(boxes[:, :2], device="cuda"), xy
                ).argmin(1)
                features = sparse.features[nearest].float().cpu().numpy()
            else:
                features = np.empty((0, 128), np.float32)
        self.sample = None
        self.model.dense_head.forward_ret_dict.clear()
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "features": features,
            "names": self.names,
        }


class SAMMemoryTracker:
    """Prompt birth with YOLO; propagate identities using pretrained SAM memory.

    Detection-to-mask overlap is used only for birth suppression. No Hungarian
    assignment or hand-coded motion model maintains identity. Slots are never
    reused within a sequence. Capacity is explicit rather than silently recycled.
    """

    def __init__(self, cfg):
        import torch
        from ultralytics.models.sam.predict import SAM2DynamicInteractivePredictor

        self.torch = torch

        class StreamingPredictor(SAM2DynamicInteractivePredictor):
            def track_step(self, obj_idx=None, *args, **kwargs):
                result = super().track_step(obj_idx, *args, **kwargs)
                if obj_idx is None:
                    self.last_output = result
                return result

        self.capacity = cfg.get("max_objects", 8)
        self.memory_frames = cfg.get("memory_frames", 3)
        self.predictor = StreamingPredictor(
            overrides={
                "model": cfg["weights"],
                "task": "segment",
                "mode": "predict",
                "imgsz": 1024,
                "device": 0,
                "quantize": 16,
                "verbose": False,
                "conf": 0.01,
                "save": False,
            },
            max_obj_num=self.capacity,
        )
        self.labels = {}
        self.confidences = {}
        self.next_slot = 0
        self.last_masks = {}
        self.capacity_rejections = 0

    def reset(self):
        self.predictor.memory_bank.clear()
        self.predictor.obj_idx_set.clear()
        self.labels.clear()
        self.confidences.clear()
        self.last_masks.clear()
        self.next_slot = 0
        self.capacity_rejections = 0

    def update(self, image, detections):
        torch = self.torch
        prompts, slots = [], []
        height, width = image.shape[:2]
        for box in sorted(detections, key=lambda b: -b["confidence"]):
            if box["label"] not in (
                "car",
                "truck",
                "bus",
                "person",
                "motorcycle",
                "bicycle",
            ):
                continue
            x1, y1, x2, y2 = np.asarray(box["xyxy"], int)
            x1, x2 = np.clip([x1, x2], 0, width)
            y1, y2 = np.clip([y1, y2], 0, height)
            if x2 <= x1 or y2 <= y1:
                continue
            already = False
            for slot, mask in self.last_masks.items():
                if self.labels[slot] != box["label"]:
                    continue
                overlap = mask[y1:y2, x1:x2].sum()
                union = mask.sum() + (x2 - x1) * (y2 - y1) - overlap
                if overlap / max(union, 1) > 0.15:
                    already = True
                    break
            # Also suppress duplicate births in the same frame.
            for previous in prompts:
                from .geometry import iou

                if iou(np.asarray(previous), np.asarray(box["xyxy"])) > 0.5:
                    already = True
            if already:
                continue
            if self.next_slot == self.capacity:
                self.capacity_rejections += 1
                continue
            slot = self.next_slot
            self.next_slot += 1
            self.labels[slot] = box["label"]
            self.confidences[slot] = box["confidence"]
            prompts.append(box["xyxy"])
            slots.append(slot)
        if not self.labels:
            return []
        kwargs = (
            {"bboxes": prompts, "obj_ids": slots, "update_memory": True}
            if slots
            else {}
        )
        with torch.inference_mode():
            self.predictor(source=image, **kwargs)
            output = self.predictor.last_output
            keys = sorted(self.predictor.obj_idx_set)
            low = output["pred_masks"]
            from ultralytics.utils.ops import scale_masks

            masks = scale_masks(low, (height, width), padding=False)[:, 0] > 0
            scores = output["object_score_logits"].sigmoid().flatten()
            features, pos = self.predictor.model._encode_new_memory(
                current_vision_feats=self.predictor.vision_feats,
                feat_sizes=self.predictor.feat_sizes,
                pred_masks_high_res=torch.nn.functional.interpolate(
                    low, size=self.predictor.imgsz, mode="bilinear", align_corners=False
                ),
                object_score_logits=output["object_score_logits"],
                is_mask_from_pts=False,
            )
            self.predictor.memory_bank.append(
                dict(output, maskmem_features=features, maskmem_pos_enc=pos)
            )
            # FIFO is bounded; no full-video images or computation graph are retained.
            self.predictor.memory_bank = self.predictor.memory_bank[
                -self.memory_frames :
            ]
            masks = masks.cpu().numpy()
            scores = scores.cpu().numpy()
            pointers = output["obj_ptr"].float().cpu().numpy()
        tracks = []
        self.last_masks = {}
        for slot in keys:
            mask = masks[slot]
            if scores[slot] < 0.5 or mask.sum() < 20:
                continue
            y, x = np.where(mask)
            bbox = np.array([x.min(), y.min(), x.max() + 1, y.max() + 1], np.float32)
            self.last_masks[slot] = mask
            tracks.append(
                {
                    "id": slot + 1,
                    "mask": mask,
                    "pointer": pointers[slot],
                    "bbox": bbox,
                    "label": self.labels[slot],
                    "confidence": self.confidences[slot],
                    "presence": float(scores[slot]),
                }
            )
        return tracks
