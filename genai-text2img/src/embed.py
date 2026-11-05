"""Text -> embedding baseline pipeline.

Two embedding views are produced:
  1. CLIP joint space (ViT-B/32): pooled text + image vectors, used for a
     sanity-check retrieval baseline (does the caption find its own image?).
  2. Stable Diffusion conditioning: the per-token text-encoder output
     (77 x 768) that the diffusion U-Net is actually conditioned on.

Usage:
    python -m src.embed --pairs data/processed/pairs.csv --out data/embeddings
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLIP_ID = "openai/clip-vit-base-patch32"
SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def load_clip(model_id: str = CLIP_ID, device: str | None = None):
    from transformers import CLIPModel, CLIPProcessor

    device = device or get_device()
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor, device


def encode_texts(texts, model, processor, device, batch_size: int = 128) -> np.ndarray:
    import torch

    out = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
        out.append(feats.float().cpu().numpy())
    return _l2norm(np.concatenate(out))


def encode_images(images, model, processor, device, batch_size: int = 64) -> np.ndarray:
    """`images` is a list of PIL images or of file paths."""
    import torch
    from PIL import Image

    out = []
    for i in range(0, len(images), batch_size):
        batch = [Image.open(p).convert("RGB") if not hasattr(p, "size") else p.convert("RGB")
                 for p in images[i:i + batch_size]]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        out.append(feats.float().cpu().numpy())
    return _l2norm(np.concatenate(out))


def retrieval_metrics(text_emb: np.ndarray, img_emb: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Paired retrieval: row i of text_emb belongs to row i of img_emb.

    Returns Recall@k for text->image and image->text, plus the median rank.
    """
    sim = text_emb @ img_emb.T                      # [N_text, N_img]
    n = sim.shape[0]
    diag = np.diag(sim)[:, None]
    rank_t2i = (sim > diag).sum(axis=1)             # 0 == correct item ranked first
    rank_i2t = (sim.T > diag).sum(axis=1)
    res = {"n": int(n)}
    for k in ks:
        res[f"t2i_R@{k}"] = round(float((rank_t2i < k).mean()), 4)
        res[f"i2t_R@{k}"] = round(float((rank_i2t < k).mean()), 4)
    res["t2i_median_rank"] = int(np.median(rank_t2i) + 1)
    res["i2t_median_rank"] = int(np.median(rank_i2t) + 1)
    return res


def sd_prompt_embeddings(prompts, sd_id: str = SD_ID, device: str | None = None) -> np.ndarray:
    """Per-token conditioning tensors used by Stable Diffusion: [N, 77, 768]."""
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    device = device or get_device()
    tok = CLIPTokenizer.from_pretrained(sd_id, subfolder="tokenizer")
    enc = CLIPTextModel.from_pretrained(sd_id, subfolder="text_encoder").to(device).eval()
    ids = tok(list(prompts), padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        emb = enc(ids)[0]
    return emb.float().cpu().numpy()


def run(pairs_csv: str, out_dir: str, data_root: str | None = None, batch_size: int = 128) -> dict:
    pairs = pd.read_csv(pairs_csv)
    root = Path(data_root) if data_root else Path(pairs_csv).parent
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, proc, device = load_clip(device=None)
    print(f"device={device}  pairs={len(pairs)}")
    txt = encode_texts(pairs["caption"].tolist(), model, proc, device, batch_size)
    img = encode_images([root / p for p in pairs["image_path"]], model, proc, device, max(batch_size // 2, 8))
    np.save(out / "clip_text.npy", txt)
    np.save(out / "clip_image.npy", img)

    metrics = {"all": retrieval_metrics(txt, img)}
    test_mask = (pairs["split"] == "test").to_numpy()
    if test_mask.sum() > 10:
        metrics["test_split"] = retrieval_metrics(txt[test_mask], img[test_mask])
    metrics["clip_model"] = CLIP_ID
    metrics["embedding_dim"] = int(txt.shape[1])
    (out / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/processed/pairs.csv")
    ap.add_argument("--out", default="data/embeddings")
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args()
    run(a.pairs, a.out, batch_size=a.batch_size)
