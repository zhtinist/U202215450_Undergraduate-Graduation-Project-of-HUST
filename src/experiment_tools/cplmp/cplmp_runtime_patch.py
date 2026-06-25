"""
Runtime monkey-patch for CPLMP (Cross-Page Logic Memory Pool) on UDOP2T5.

This module replaces ``UDOP2T5.encode_images_wocr`` with a wrapper that, when
``CPLMP_ENABLE`` is truthy, applies the causal CPLMP flow aligned with
``template-of-thesis-main/body/method.tex``: per page, read only from prior
memory ``M_{t-1}``, then residual on ``X_t``; memory row ``\\bar{m}_t`` is
aggregated from the current ``X_t`` (Eq.~3-1) and appended after that page's
forward update (values computed before read in code, append after—causally
equivalent to read-then-append). When disabled, the original implementation is
used unchanged.

Environment variables (defaults match thesis main experiment where noted):
  CPLMP_ENABLE            0/1 — master switch
  CPLMP_WINDOW            int, default 8  (W)
  CPLMP_TOPK              int, default 16 (K)
  CPLMP_TEMP              float, default 0.72 (tau)
  CPLMP_ALPHA             float, default 0.06 (alpha)
  CPLMP_ALIGN_SCALE       0/1 — RMS scale align on readout, default 1
  CPLMP_CTX_LN            0/1 — LayerNorm on readout vectors, default 0
  CPLMP_USE_TANH          0/1 — tanh on readout, default 0
  CPLMP_SKIP_MEMORY       0/1 — M6: do not append to memory (read uses prior only)
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

_CPLMP_PATCHED = False
_ORIGINAL_ENCODE: Optional[callable] = None


def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _topk_mean_write_vector(
    X: torch.Tensor, valid: torch.Tensor, k: int
) -> torch.Tensor:
    """Eq. (cplmp-write): X [L,d], valid [L] bool → [1,d]."""
    idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return torch.zeros(1, X.shape[-1], device=X.device, dtype=X.dtype)
    rows = X.index_select(0, idx)
    n = rows.shape[0]
    if n <= k:
        return rows.mean(dim=0, keepdim=True)
    norms = rows.norm(dim=-1)
    top_idx = torch.topk(norms, min(k, n), largest=True).indices
    return rows.index_select(0, top_idx).mean(dim=0, keepdim=True)


def _cross_attn_readout(
    X: torch.Tensor,
    M: torch.Tensor,
    valid: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Scaled dot-product read; invalid token rows → zero context."""
    d = X.shape[-1]
    out = torch.zeros_like(X)
    if M.shape[0] == 0 or not bool(valid.any()):
        return out
    idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    Xv = X.index_select(0, idx)
    logits = (Xv @ M.t()) / math.sqrt(float(d))
    logits = logits / max(tau, 1e-6)
    w = torch.softmax(logits, dim=-1)
    Cv = w @ M
    out.index_copy_(0, idx, Cv)
    return out


def _rms_rows(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(t.pow(2).mean(dim=-1, keepdim=True).clamp(min=eps))


def _apply_readout_post(
    X: torch.Tensor,
    C: torch.Tensor,
    valid: torch.Tensor,
    align_scale: bool,
    ctx_ln: bool,
    use_tanh: bool,
) -> torch.Tensor:
    Z = C
    if align_scale:
        rms_x = _rms_rows(X)
        rms_c = _rms_rows(Z)
        Z = Z * (rms_x / rms_c)
    if ctx_ln:
        Z = F.layer_norm(Z, (Z.shape[-1],))
    if use_tanh:
        Z = torch.tanh(Z)
    Z = Z * valid.unsqueeze(-1).to(Z.dtype)
    return Z


def _cplmp_on_stacked_pages(
    image_features: torch.Tensor,
    image_attns: torch.Tensor,
    window: int,
    topk: int,
    tau: float,
    alpha: float,
    align_scale: bool,
    ctx_ln: bool,
    use_tanh: bool,
    skip_memory_update: bool,
    attn_out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    image_features: [T, L, d], image_attns: [T, L] (long/float mask).
    Returns flattened (≤1024, d) and matching 1-D attention, same as baseline.
    """
    T, L, d = image_features.shape
    memory: List[torch.Tensor] = []

    page_feats: List[torch.Tensor] = []
    page_attns: List[torch.Tensor] = []

    for t in range(T):
        X = image_features[t].contiguous()
        attn_row = image_attns[t]
        valid = attn_row.ne(0) if attn_row.dtype != torch.bool else attn_row

        M_prev = (
            torch.cat(memory, dim=0)
            if memory
            else X.new_zeros((0, d), dtype=X.dtype, device=X.device)
        )

        bar_m = _topk_mean_write_vector(X, valid, topk)

        X_out = X
        if M_prev.shape[0] > 0 and alpha != 0.0:
            C = _cross_attn_readout(X, M_prev, valid, tau)
            Z = _apply_readout_post(
                X, C, valid, align_scale, ctx_ln, use_tanh
            )
            upd = alpha * Z
            mask_f = valid.unsqueeze(-1).to(upd.dtype)
            X_out = X + upd * mask_f

        page_feats.append(X_out)
        page_attns.append(attn_row.contiguous())

        if not skip_memory_update:
            memory.append(bar_m)
            if len(memory) > window:
                memory = memory[-window:]

    mean_feature = torch.cat(page_feats, dim=0)
    image_attn = torch.cat(page_attns, dim=0)
    mean_feature = mean_feature[:1024, ...]
    image_attn = image_attn[:1024].to(dtype=attn_out_dtype)
    return mean_feature, image_attn


def _encode_images_wocr_cplmp(
    self,
    images,
    ocr_ids,
    seg_data,
    visual_seg_data,
    question,
):
    assert _ORIGINAL_ENCODE is not None
    if not _env_truthy("CPLMP_ENABLE", default=False):
        return _ORIGINAL_ENCODE(
            self, images, ocr_ids, seg_data, visual_seg_data, question
        )

    window = max(1, _env_int("CPLMP_WINDOW", 8))
    topk = max(1, _env_int("CPLMP_TOPK", 16))
    tau = _env_float("CPLMP_TEMP", 0.72)
    alpha = _env_float("CPLMP_ALPHA", 0.06)
    align_scale = _env_truthy("CPLMP_ALIGN_SCALE", default=True)
    ctx_ln = _env_truthy("CPLMP_CTX_LN", default=False)
    use_tanh = _env_truthy("CPLMP_USE_TANH", default=False)
    skip_memory = _env_truthy("CPLMP_SKIP_MEMORY", default=False)

    pad_token = self.vision_tower.image_processor.pad_token_id

    def _as_list(x):
        if isinstance(x, torch.Tensor):
            return [x]
        return x

    images_l = _as_list(images)
    ocrs_l = _as_list(ocr_ids)
    seg_l = _as_list(seg_data)
    vis_l = _as_list(visual_seg_data)
    if question is None:
        question_l = [None] * len(images_l)
    else:
        question_l = _as_list(question)

    image_features_pool = []
    image_attn_pool = []

    for image, ocr_id, seg_data_, visual_seg_data_, question_ in zip(
        images_l, ocrs_l, seg_l, vis_l, question_l
    ):
        attn_mask = ocr_id.ne(pad_token).to(dtype=image.dtype, device=image.device)
        features = self.vision_tower(
            input_ids=ocr_id,
            attention_mask=attn_mask,
            image=image,
            seg_data=seg_data_,
            visual_seg_data=visual_seg_data_,
            question=question_,
        )
        image_features = features.last_hidden_state
        image_attns = features.attention_mask

        if image.dim() == 4 and image.shape[0] > 1:
            T = image.shape[0]
            if image_features.dim() == 3 and image_features.shape[0] == T:
                mf, ia = _cplmp_on_stacked_pages(
                    image_features,
                    image_attns,
                    window,
                    topk,
                    tau,
                    alpha,
                    align_scale,
                    ctx_ln,
                    use_tanh,
                    skip_memory,
                    image_attns.dtype,
                )
            else:
                mf = image_features.reshape(-1, image_features.shape[-1])[:1024, ...]
                ia = image_attns.reshape(-1)[:1024]
        else:
            mf = image_features.reshape(-1, image_features.shape[-1])[:1024, ...]
            ia = image_attns.reshape(-1)[:1024]

        image_features_pool.append(mf)
        image_attn_pool.append(ia)

    if any(x.shape != image_features_pool[0].shape for x in image_features_pool):
        max_token_len = max(x.shape[0] for x in image_features_pool)
        new_embed = []
        new_attn = []
        for feature in image_features_pool:
            pad_rows = max_token_len - feature.shape[0]
            cur = torch.cat(
                (
                    feature,
                    torch.zeros(
                        (pad_rows, feature.shape[1]),
                        dtype=feature.dtype,
                        device=feature.device,
                    ),
                ),
                dim=0,
            )
            new_embed.append(cur)
        for feature in image_attn_pool:
            pad_rows = max_token_len - feature.shape[0]
            cur = torch.cat(
                (
                    feature,
                    torch.zeros(
                        (pad_rows,),
                        dtype=feature.dtype,
                        device=feature.device,
                    ),
                ),
                dim=0,
            )
            new_attn.append(cur)
        pool_image_features = torch.stack(new_embed, dim=0)
        pool_image_attn = torch.stack(new_attn, dim=0)
    else:
        pool_image_features = torch.stack(image_features_pool, dim=0)
        pool_image_attn = torch.stack(image_attn_pool, dim=0)

    return pool_image_features, pool_image_attn


def apply_cplmp_patch() -> None:
    """Monkey-patch ``UDOP2T5.encode_images_wocr`` once (idempotent)."""
    global _CPLMP_PATCHED, _ORIGINAL_ENCODE
    if _CPLMP_PATCHED:
        return
    from model.language_model.llava_phi import UDOP2T5

    _ORIGINAL_ENCODE = UDOP2T5.encode_images_wocr
    UDOP2T5.encode_images_wocr = _encode_images_wocr_cplmp
    _CPLMP_PATCHED = True
