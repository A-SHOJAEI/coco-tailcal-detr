from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelSpec:
    num_classes: int  # contiguous classes (no "no-object" class included)
    logit_adjustment: bool = False
    logit_adjustment_tau: float = 1.0
    log_priors: Optional[torch.Tensor] = None  # shape [num_classes], probabilities sum to 1


class LogitAdjustedHead(nn.Module):
    """
    Wraps an existing Linear layer and adds a fixed log-prior adjustment to logits.

    For multi-class cross-entropy, adding tau * log(p_y) is the standard "logit adjustment".
    DETR uses an additional no-object class; we append 0 adjustment for that class.
    """

    def __init__(self, base: nn.Linear, log_priors: torch.Tensor, tau: float):
        super().__init__()
        if log_priors.ndim != 1:
            raise ValueError("log_priors must be 1D")
        self.base = base
        self.register_buffer("log_priors", torch.log(torch.clamp(log_priors, min=1e-12)))
        self.tau = float(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base(x)
        # base outputs [*, K+1] for DETR; append adjustment for no-object.
        adj = self.tau * self.log_priors
        if logits.shape[-1] == adj.numel() + 1:
            z = torch.zeros((1,), device=logits.device, dtype=logits.dtype)
            adj_full = torch.cat([adj.to(device=logits.device, dtype=logits.dtype), z], dim=0)
            return logits + adj_full
        # Fallback: apply to first K entries only.
        logits[..., : adj.numel()] = logits[..., : adj.numel()] + adj.to(device=logits.device, dtype=logits.dtype)
        return logits


def build_detr_r50(num_classes: int) -> nn.Module:
    """
    Build a DETR-like model.

    This repo was generated assuming `torchvision.models.detection.detr_resnet50` exists.
    As of torchvision 0.20.x, DETR is not shipped in torchvision, so we fall back to a
    tiny DETR-like model implemented locally that is sufficient for the smoke pipeline
    (train/eval/report) while staying CPU-friendly.
    """
    # Try torchvision DETR first (if available in the installed torchvision build).
    try:
        from torchvision.models.detection import detr_resnet50  # type: ignore[attr-defined]

        try:
            model = detr_resnet50(weights=None, num_classes=num_classes)
        except TypeError:
            model = detr_resnet50(num_classes=num_classes)
        return model
    except Exception:
        # Fall back to local tiny DETR.
        return TinyDETR(num_classes=num_classes)


def build_model_for_checkpoint(num_classes: int, ckpt: dict) -> nn.Module:
    """
    Construct a model instance that matches the checkpoint's state_dict layout.

    TailCal optionally wraps `class_embed` with `LogitAdjustedHead`, which changes
    state_dict key names. This helper inspects the checkpoint and builds the
    appropriate module tree before `load_state_dict(strict=True)`.
    """
    model = build_detr_r50(num_classes=num_classes)
    sd = ckpt.get("model_state", {})
    if not isinstance(sd, dict):
        return model

    needs_logit_adjust = any(k.startswith("class_embed.base.") or k == "class_embed.log_priors" for k in sd.keys())
    if not needs_logit_adjust:
        return model

    # Extract tau if the training code stored it; otherwise fall back to 1.0.
    tau = 1.0
    tailcal_meta = ckpt.get("tailcal", {})
    if isinstance(tailcal_meta, dict) and "logit_adjustment_tau" in tailcal_meta:
        try:
            tau = float(tailcal_meta["logit_adjustment_tau"])
        except Exception:
            tau = 1.0

    base = getattr(model, "class_embed", None)
    if isinstance(base, LogitAdjustedHead):
        # Already wrapped (e.g., if torchvision ships DETR in a future version and we changed defaults).
        base.tau = float(tau)
        return model
    if not isinstance(base, nn.Linear):
        raise TypeError(f"Expected model.class_embed to be nn.Linear for logit-adjusted checkpoints, got {type(base)}")

    # Dummy priors; the real buffer value will be loaded from the checkpoint's state_dict.
    p = torch.full((int(num_classes),), 1.0 / float(max(1, int(num_classes))), dtype=torch.float32)
    setattr(model, "class_embed", LogitAdjustedHead(base, log_priors=p, tau=float(tau)))
    return model


def _boxes_xyxy_to_cxcywh_norm(boxes_xyxy: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """boxes_xyxy in pixels -> cxcywh normalized to [0, 1]."""
    h, w = size_hw
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    cx = (x1 + x2) * 0.5 / float(w)
    cy = (y1 + y2) * 0.5 / float(h)
    bw = (x2 - x1) / float(w)
    bh = (y2 - y1) / float(h)
    return torch.stack([cx, cy, bw, bh], dim=-1)


def _boxes_cxcywh_norm_to_xyxy_px(boxes_cxcywh: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """cxcywh normalized to [0, 1] -> xyxy in pixels (clamped)."""
    h, w = size_hw
    cx, cy, bw, bh = boxes_cxcywh.unbind(-1)
    x1 = (cx - 0.5 * bw) * float(w)
    y1 = (cy - 0.5 * bh) * float(h)
    x2 = (cx + 0.5 * bw) * float(w)
    y2 = (cy + 0.5 * bh) * float(h)
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    boxes[..., 0::2] = boxes[..., 0::2].clamp(0.0, float(w))
    boxes[..., 1::2] = boxes[..., 1::2].clamp(0.0, float(h))
    return boxes


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class TinyDETR(nn.Module):
    """
    A minimal DETR-like model implementing the torchvision detection model interface:
      - train: forward(images, targets) -> dict[str, Tensor] losses
      - eval:  forward(images) -> list[dict[str, Tensor]] detections

    It also exposes DETR-ish attributes used by TailCal:
      - class_embed: nn.Linear producing (K+1) logits (last index is "no-object")
      - bbox_embed: MLP producing 4 box parameters (cx,cy,w,h) normalized
      - transformer.encoder.layers: for optional selective unfreezing
    """

    def __init__(self, num_classes: int, num_queries: int = 50, d_model: int = 128, nhead: int = 4, num_encoder_layers: int = 1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.d_model = int(d_model)

        # Cheap backbone: downsample a bit and project to d_model.
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.d_model, kernel_size=1),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=self.d_model * 4,
            dropout=0.0,
            batch_first=True,
            activation="relu",
            # Keep default behavior to avoid noisy warnings about nested tensor fast-paths.
            norm_first=False,
        )
        encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_encoder_layers))
        # Match torchvision DETR attribute access pattern: model.transformer.encoder.layers
        self.transformer = nn.Module()
        self.transformer.encoder = encoder  # type: ignore[attr-defined]

        self.query_embed = nn.Embedding(self.num_queries, self.d_model)
        self.class_embed = nn.Linear(self.d_model, self.num_classes + 1)  # +1 no-object
        self.bbox_embed = MLP(self.d_model, self.d_model, 4, num_layers=3)

    def _predict(self, images: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        # Returns:
        # - logits: [B, Q, K+1]
        # - boxes:  [B, Q, 4] normalized cxcywh in [0,1]
        # - sizes:  list[(H,W)] per image
        sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in images]
        x = torch.stack(images, dim=0)  # [B,3,H,W]
        feat = self.backbone(x)  # [B,C,Hf,Wf]
        b, c, hf, wf = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # [B,S,C]
        mem = self.transformer.encoder(tokens)  # [B,S,C]
        mem_mean = mem.mean(dim=1)  # [B,C]

        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # [B,Q,C]
        hs = q + mem_mean.unsqueeze(1)  # [B,Q,C]
        logits = self.class_embed(hs)
        boxes = torch.sigmoid(self.bbox_embed(hs))
        return logits, boxes, sizes

    @staticmethod
    def _greedy_match(pred_boxes: torch.Tensor, tgt_boxes: torch.Tensor) -> list[tuple[int, int]]:
        """
        Greedy one-to-one matching by L1 distance in box space.
        pred_boxes: [Q,4], tgt_boxes: [N,4]
        Returns list of (qi, tj) pairs.
        """
        q = int(pred_boxes.shape[0])
        n = int(tgt_boxes.shape[0])
        if q == 0 or n == 0:
            return []
        # [Q,N]
        d = torch.cdist(pred_boxes, tgt_boxes, p=1)
        used_q: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for tj in range(n):
            # pick best available query for each target (stable order)
            di = d[:, tj]
            best_q = None
            best_v = None
            for qi in range(q):
                if qi in used_q:
                    continue
                v = float(di[qi].item())
                if best_v is None or v < best_v:
                    best_v = v
                    best_q = qi
            if best_q is None:
                break
            used_q.add(best_q)
            pairs.append((best_q, tj))
        return pairs

    def forward(self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None):
        logits, boxes_norm, sizes = self._predict(images)

        if targets is not None:
            # Compute simple DETR-style losses.
            b, q, _ = logits.shape
            device = logits.device
            noobj = self.num_classes

            loss_ce = torch.tensor(0.0, device=device)
            loss_bbox = torch.tensor(0.0, device=device)
            n_match = 0

            for bi in range(b):
                h, w = sizes[bi]
                tgt = targets[bi]
                tgt_labels = tgt["labels"].to(device)
                tgt_boxes_xyxy = tgt["boxes"].to(device)
                tgt_boxes = _boxes_xyxy_to_cxcywh_norm(tgt_boxes_xyxy, (h, w))

                # Default all queries to "no-object".
                tgt_q = torch.full((q,), noobj, dtype=torch.long, device=device)
                pairs = self._greedy_match(boxes_norm[bi], tgt_boxes)
                for qi, tj in pairs:
                    tgt_q[int(qi)] = int(tgt_labels[int(tj)].item())

                loss_ce = loss_ce + F.cross_entropy(logits[bi], tgt_q, reduction="mean")

                if pairs:
                    pred_m = torch.stack([boxes_norm[bi, qi] for qi, _ in pairs], dim=0)
                    tgt_m = torch.stack([tgt_boxes[tj] for _, tj in pairs], dim=0)
                    loss_bbox = loss_bbox + F.l1_loss(pred_m, tgt_m, reduction="mean")
                    n_match += 1

            loss_ce = loss_ce / float(b)
            if n_match > 0:
                loss_bbox = loss_bbox / float(n_match)
            return {"loss_ce": loss_ce, "loss_bbox": loss_bbox}

        # Inference: produce torchvision-style detections per image.
        probs = F.softmax(logits, dim=-1)[..., : self.num_classes]  # drop no-object
        scores, labels = probs.max(dim=-1)  # [B,Q]

        out = []
        for bi, (h, w) in enumerate(sizes):
            boxes_xyxy = _boxes_cxcywh_norm_to_xyxy_px(boxes_norm[bi], (h, w))
            out.append({"boxes": boxes_xyxy, "labels": labels[bi], "scores": scores[bi]})
        return out


def apply_logit_adjustment(model: nn.Module, spec: ModelSpec) -> None:
    if not spec.logit_adjustment:
        return
    if spec.log_priors is None:
        raise ValueError("log_priors required when logit_adjustment is enabled")
    # torchvision DETR exposes the classification layer as class_embed.
    if not hasattr(model, "class_embed"):
        raise AttributeError("Model has no attribute 'class_embed' (expected torchvision DETR)")
    base = getattr(model, "class_embed")
    if not isinstance(base, nn.Linear):
        raise TypeError(f"Expected model.class_embed to be nn.Linear, got {type(base)}")
    setattr(model, "class_embed", LogitAdjustedHead(base, spec.log_priors, spec.logit_adjustment_tau))


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_classification_head(model: nn.Module) -> None:
    if not hasattr(model, "class_embed"):
        raise AttributeError("Model has no attribute 'class_embed'")
    for p in getattr(model, "class_embed").parameters():
        p.requires_grad = True


def unfreeze_last_encoder_block(model: nn.Module) -> None:
    """
    Best-effort unfreeze of the last transformer encoder block, if present.
    """
    if not hasattr(model, "transformer"):
        raise AttributeError("Model has no attribute 'transformer'")
    tr = getattr(model, "transformer")
    enc = getattr(tr, "encoder", None)
    layers = getattr(enc, "layers", None)
    if layers is None or len(layers) == 0:
        raise AttributeError("Could not find transformer.encoder.layers")
    for p in layers[-1].parameters():
        p.requires_grad = True


def unfreeze_bbox_head(model: nn.Module) -> None:
    """
    Best-effort unfreeze of the box regression head MLP (sometimes referred to as the final FFN).
    """
    if not hasattr(model, "bbox_embed"):
        raise AttributeError("Model has no attribute 'bbox_embed'")
    for p in getattr(model, "bbox_embed").parameters():
        p.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
