#!/usr/bin/env python3
"""
Evaluate UDOP+T5 checkpoint on a JSON manifest (e.g. 10pct subset).
Outputs ROUGE-1/L and METEOR into a JSON file.

Multi-GPU: set --num-workers N and --cuda-devices 0,1,2,3 (length must match N).
Each worker loads the full model on one GPU and processes a disjoint index shard.

Paths are resolved from this file so the script is portable across machines.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
LEC = SRC / "LecSlides_370K"
TRAIN_FILE = LEC / "train" / "train.py"

for p in (SRC, LEC, LEC / "train"):
    sp = str(p)
    if p.exists() and sp not in sys.path:
        sys.path.insert(0, sp)


def _load_train_module():
    spec = importlib.util.spec_from_file_location("lec_train", TRAIN_FILE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patch_runtime() -> None:
    import transformers

    try:
        from baseline_entry import apply_lecslides_compat_patches  # type: ignore

        apply_lecslides_compat_patches()
    except Exception:
        pass

    legacy = "/home/emzhang/data/t5-large"
    t5_path = os.environ.get("LECSLIDES_T5_TOKENIZER_PATH") or os.environ.get(
        "LEC_T5_TOKENIZER"
    )
    _orig = transformers.AutoTokenizer.from_pretrained

    def _wrapped(path, *args, **kwargs):
        if isinstance(path, str) and path == legacy and t5_path:
            path = t5_path
        return _orig(path, *args, **kwargs)

    transformers.AutoTokenizer.from_pretrained = _wrapped  # type: ignore[method-assign]


def _set_eval_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _chunk_indices(indices: List[int], n_workers: int) -> List[List[int]]:
    n = len(indices)
    base = n // n_workers
    rem = n % n_workers
    out: List[List[int]] = []
    start = 0
    for w in range(n_workers):
        sz = base + (1 if w < rem else 0)
        out.append(indices[start : start + sz])
        start += sz
    return out


def _import_udop2t5():
    try:
        from llava.model.language_model.llava_phi import UDOP2T5
    except ImportError:
        from model.language_model.llava_phi import UDOP2T5
    return UDOP2T5


def _eval_worker_shard(payload: Dict[str, Any]) -> Dict[str, Any]:
    gpu_id = int(payload["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if payload.get("lec_t5_tokenizer_path"):
        os.environ["LECSLIDES_T5_TOKENIZER_PATH"] = str(payload["lec_t5_tokenizer_path"])

    import torch

    base_seed = int(payload.get("eval_seed", 42))
    _set_eval_seed(base_seed + gpu_id * 97981)
    _patch_runtime()
    tr = _load_train_module()
    from dataclasses import dataclass as _dc
    from transformers import AutoTokenizer

    UDOP2T5 = _import_udop2t5()

    @_dc
    class DA:
        data_path: str = ""
        lazy_preprocess: bool = False
        is_multimodal: bool = True
        image_folder: Optional[str] = None
        image_aspect_ratio: str = "square"
        ocr_order: str = "my"
        image_processor: object | None = None

    checkpoint = Path(payload["checkpoint"])
    data_json = Path(payload["data_json"])
    indices: List[int] = list(payload["indices"])
    max_new_tokens = int(payload["max_new_tokens"])
    num_beams = int(payload["num_beams"])
    length_penalty = float(payload["length_penalty"])
    no_repeat_ngram_size = int(payload["no_repeat_ngram_size"])

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
    )
    model = UDOP2T5.from_pretrained(str(checkpoint), torch_dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    model.config.use_cache = True
    vt = model.vision_tower
    if hasattr(vt, "to"):
        vt.to(device=device, dtype=torch.float32)

    data_args = DA(
        data_path=str(data_json),
        image_processor=vt.image_processor,
    )
    ds = tr.Slideshare_UDOP_Dataset_For_T5(
        tokenizer=tokenizer, data_path=str(data_json), data_args=data_args
    )
    collate = tr.DataCollatorForSupervisedDatasetUDOPSlideVQA(
        tokenizer=tokenizer, image_processor=data_args.image_processor
    )

    dev = device

    def to_dev(x, dev_arg=None):
        d = dev_arg if dev_arg is not None else dev
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(d)
        if isinstance(x, list):
            return [to_dev(t, d) for t in x]
        return x

    preds: List[str] = []
    refs: List[str] = []
    out_indices: List[int] = []
    for j, i in enumerate(indices):
        row = collate([ds[i]])
        sample = ds.list_data_dict[i]
        ref = sample.get("summary") or sample.get("description") or ""
        try:
            kwargs = dict(
                input_ids=row["input_ids"].to(device),
                attention_mask=row["attention_mask"].to(device),
                images=to_dev(row.get("images"), device),
                ocrs=to_dev(row.get("ocrs"), device),
                seg_data=to_dev(row.get("seg_data"), device),
                visual_seg_data=to_dev(row.get("visual_seg_data"), device),
                question=None,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=False,
            )
            with torch.inference_mode():
                out = model.generate(**kwargs)
            pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        except Exception as e:
            pred = f"[ERROR:{e}]"
        preds.append(pred)
        refs.append(ref)
        out_indices.append(i)
        if (j + 1) % 10 == 0:
            print(
                f"[gpu{gpu_id}] done {j + 1}/{len(indices)}",
                flush=True,
            )
    return {"indices": out_indices, "preds": preds, "refs": refs}


def _run_single_gpu(
    args: argparse.Namespace,
    tr: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    ds: Any,
    collate: Any,
    n: int,
):
    preds: List[str] = []
    refs: List[str] = []
    import torch

    dev = device

    def to_dev(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(dev)
        if isinstance(x, list):
            return [to_dev(t) for t in x]
        return x

    for i in range(n):
        row = collate([ds[i]])
        sample = ds.list_data_dict[i]
        ref = sample.get("summary") or sample.get("description") or ""
        try:
            kwargs = dict(
                input_ids=row["input_ids"].to(device),
                attention_mask=row["attention_mask"].to(device),
                images=to_dev(row.get("images")),
                ocrs=to_dev(row.get("ocrs")),
                seg_data=to_dev(row.get("seg_data")),
                visual_seg_data=to_dev(row.get("visual_seg_data")),
                question=None,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                do_sample=False,
            )
            with torch.inference_mode():
                out = model.generate(**kwargs)
            pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        except Exception as e:
            pred = f"[ERROR:{e}]"
        preds.append(pred)
        refs.append(ref)
        if (i + 1) % 10 == 0:
            print(f"done {i + 1}/{n}", flush=True)
    return preds, refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-json", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="并行进程数；>1 时每进程独占一块 GPU（见 --cuda-devices）",
    )
    ap.add_argument(
        "--cuda-devices",
        type=str,
        default=None,
        help="逗号分隔物理 GPU 编号，如 0,1,2,3；数量须等于 num-workers",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="评测可复现用随机种子（多进程时各 GPU 在 seed 上偏移）",
    )
    args = ap.parse_args()

    import torch

    _set_eval_seed(args.seed)
    _patch_runtime()
    tr = _load_train_module()
    from dataclasses import dataclass as _dc
    from transformers import AutoTokenizer

    UDOP2T5 = _import_udop2t5()

    @_dc
    class DA:
        data_path: str = ""
        lazy_preprocess: bool = False
        is_multimodal: bool = True
        image_folder: Optional[str] = None
        image_aspect_ratio: str = "square"
        ocr_order: str = "my"
        image_processor: object | None = None

    lec_tok = os.environ.get("LECSLIDES_T5_TOKENIZER_PATH") or os.environ.get(
        "LEC_T5_TOKENIZER"
    )
    full_len = len(json.loads(Path(args.data_json).read_text(encoding="utf-8")))
    n = full_len if args.max_samples <= 0 else min(full_len, args.max_samples)
    idx_list = list(range(n))

    used_gpus: Optional[str] = None
    preds: List[str]
    refs: List[str]

    if args.num_workers > 1:
        if not torch.cuda.is_available():
            print("num-workers>1 需要 CUDA", flush=True)
            raise SystemExit(1)
        if args.cuda_devices:
            gpus = [int(x.strip()) for x in args.cuda_devices.split(",") if x.strip()]
        else:
            gpus = list(range(min(args.num_workers, torch.cuda.device_count())))
        if len(gpus) != args.num_workers:
            print(
                f"error: --num-workers={args.num_workers} 与 GPU 列表 {gpus} 长度不一致，请设置 --cuda-devices",
                flush=True,
            )
            raise SystemExit(1)
        chunks = _chunk_indices(idx_list, args.num_workers)
        payload_base = {
            "checkpoint": str(args.checkpoint.resolve()),
            "data_json": str(args.data_json.resolve()),
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "lec_t5_tokenizer_path": lec_tok or "",
            "eval_seed": args.seed,
        }
        pos = {idx: k for k, idx in enumerate(idx_list)}
        preds = [""] * n
        refs = [""] * n
        ctx = mp.get_context("spawn")
        print(
            f"parallel eval: {args.num_workers} workers on GPUs {gpus}, total samples={n}",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as ex:
            futs = []
            for w in range(args.num_workers):
                chunk = chunks[w]
                if not chunk:
                    continue
                p = {**payload_base, "gpu_id": gpus[w], "indices": chunk}
                futs.append(ex.submit(_eval_worker_shard, p))
            for fut in as_completed(futs):
                r = fut.result()
                for idx, pred, ref in zip(r["indices"], r["preds"], r["refs"]):
                    k = pos[idx]
                    preds[k] = pred
                    refs[k] = ref
        used_gpus = ",".join(str(x) for x in gpus)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.checkpoint),
            model_max_length=1024,
            padding_side="right",
            use_fast=False,
        )
        model = UDOP2T5.from_pretrained(str(args.checkpoint), torch_dtype=torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        model.config.use_cache = True
        vt = model.vision_tower
        if hasattr(vt, "to"):
            vt.to(device=device, dtype=torch.float32)
        data_args = DA(
            data_path=str(args.data_json),
            image_processor=vt.image_processor,
        )
        ds = tr.Slideshare_UDOP_Dataset_For_T5(
            tokenizer=tokenizer, data_path=str(args.data_json), data_args=data_args
        )
        collate = tr.DataCollatorForSupervisedDatasetUDOPSlideVQA(
            tokenizer=tokenizer, image_processor=data_args.image_processor
        )
        preds, refs = _run_single_gpu(args, tr, tokenizer, model, device, ds, collate, n)
        used_gpus = None

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1_list: List[float] = []
    rl_list: List[float] = []
    for p, r in zip(preds, refs):
        if not r.strip():
            continue
        sc = scorer.score(r, p)
        r1_list.append(sc["rouge1"].fmeasure)
        rl_list.append(sc["rougeL"].fmeasure)
    rouge = {
        "rouge1_f": sum(r1_list) / max(1, len(r1_list)),
        "rougeL_f": sum(rl_list) / max(1, len(rl_list)),
    }

    meteor_mean: Any = None
    try:
        from nltk.translate.meteor_score import meteor_score

        ms: List[float] = []
        for p, r in zip(preds, refs):
            if not r.strip():
                continue
            rt = r.lower().split()
            pt = p.lower().split()
            if not rt or not pt:
                ms.append(0.0)
            else:
                ms.append(float(meteor_score([rt], pt)))
        meteor_mean = sum(ms) / max(1, len(ms))
    except Exception as e:
        meteor_mean = f"error:{e}"

    out = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data_json": str(args.data_json.resolve()),
        "num_samples": n,
        "num_workers": args.num_workers,
        "decode": {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "do_sample": False,
        },
        "rouge": rouge,
        "meteor_mean": meteor_mean,
        "note": "10pct baseline eval.",
        "seed": args.seed,
    }
    if used_gpus is not None:
        out["cuda_devices"] = used_gpus

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote", args.output_json)


if __name__ == "__main__":
    main()
