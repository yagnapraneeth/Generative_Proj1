"""Dataset selection + cleaning for the text->image project.

Default source: Flickr30k (streamed from the Hugging Face Hub, so the full
~4 GB is never downloaded). Produces a clean, de-duplicated set of
image-caption pairs (default 2,000 images -> 2,000 pairs; range 1k-5k).

Usage:
    python -m src.data_prep --source hf --n-images 2000 --out data/processed
    python -m src.data_prep --source local --local-csv my.csv --out data/processed
        (local CSV needs columns: image_path,caption ; several rows per image OK)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
from PIL import Image

HF_REPO = "Mozilla/flickr30k-transformed-captions"  # parquet mirror of Flickr30k (works with any `datasets` version)
CAPTION_KEYS = ("original_alt_text", "caption", "captions", "alt_text")  # first match wins
MIN_WORDS, MAX_WORDS = 5, 40
TARGET_WORDS = 12          # when picking one caption per image, prefer length near this
IMG_SIZE = 256             # stored size (square, center-cropped)
MIN_SIDE = 256             # discard images smaller than this

_URL_RE = re.compile(r"(https?://|www\.)", re.I)


# --------------------------------------------------------------------------- #
# caption cleaning
# --------------------------------------------------------------------------- #
def clean_caption(text: str) -> str | None:
    """Return a normalised caption, or None if it should be discarded."""
    if not isinstance(text, str):
        return None
    t = unicodedata.normalize("NFKC", text)
    t = "".join(ch for ch in t if unicodedata.category(ch)[0] != "C" or ch in " \t")
    t = re.sub(r"\s+", " ", t).strip().strip("\"'`")
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)          # "dog ." -> "dog."
    t = re.sub(r"([.!?])\1+", r"\1", t)              # "..", "!!" -> single
    if not t or _URL_RE.search(t):
        return None
    letters = sum(ch.isalpha() for ch in t)
    if letters / max(len(t), 1) < 0.6:               # mostly digits/symbols
        return None
    n_words = len(t.split())
    if not (MIN_WORDS <= n_words <= MAX_WORDS):
        return None
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?":
        t += "."
    return t


def pick_captions(captions: Iterable[str], k: int = 1) -> list[str]:
    """Clean all captions of one image and keep the k closest to TARGET_WORDS."""
    cleaned, seen = [], set()
    for c in captions:
        cc = clean_caption(c)
        if cc and cc.lower() not in seen:
            seen.add(cc.lower())
            cleaned.append(cc)
    cleaned.sort(key=lambda s: (abs(len(s.split()) - TARGET_WORDS), s))
    return cleaned[:k]


# --------------------------------------------------------------------------- #
# image handling
# --------------------------------------------------------------------------- #
def prepare_image(img: Image.Image, size: int = IMG_SIZE) -> Image.Image | None:
    """RGB, reject tiny images, center-crop to square, resize."""
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) < MIN_SIDE:
        return None
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)


def image_fingerprint(img: Image.Image) -> str:
    """Cheap near-duplicate detector: hash of an 8x8 grayscale thumbnail."""
    small = img.convert("L").resize((8, 8), Image.BILINEAR)
    return hashlib.md5(small.tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def iter_hf(repo: str = HF_REPO, split: str = "test", seed: int = 42) -> Iterator[dict]:
    from datasets import load_dataset  # lazy import

    ds = load_dataset(repo, split=split, streaming=True).shuffle(seed=seed, buffer_size=1000)
    for i, row in enumerate(ds):
        caps = None
        for key in CAPTION_KEYS:
            if key in row and row[key]:
                caps = row[key]
                break
        if caps is None:
            continue
        if isinstance(caps, str):
            caps = [caps]
        yield {"img_id": str(row.get("img_id", row.get("filename", i))), "image": row["image"], "captions": list(caps)}


def iter_local(csv_path: str) -> Iterator[dict]:
    df = pd.read_csv(csv_path)
    base = Path(csv_path).parent
    for path, grp in df.groupby("image_path", sort=False):
        p = Path(path)
        p = p if p.is_absolute() else base / p
        if not p.exists():
            continue
        yield {"img_id": p.stem, "image": Image.open(p), "captions": grp["caption"].astype(str).tolist()}


# --------------------------------------------------------------------------- #
# main pipeline
# --------------------------------------------------------------------------- #
def build_dataset(
    rows: Iterable[dict],
    out_dir: str | Path,
    n_images: int = 2000,
    captions_per_image: int = 1,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> pd.DataFrame:
    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    stats = {"scanned": 0, "kept_images": 0, "drop_small_or_bad_image": 0,
             "drop_no_valid_caption": 0, "drop_duplicate_image": 0}
    seen_fp, seen_caps, records = set(), set(), []

    for row in rows:
        if stats["kept_images"] >= n_images:
            break
        stats["scanned"] += 1
        try:
            img = prepare_image(row["image"])
        except Exception:
            img = None
        if img is None:
            stats["drop_small_or_bad_image"] += 1
            continue
        fp = image_fingerprint(img)
        if fp in seen_fp:
            stats["drop_duplicate_image"] += 1
            continue
        caps = pick_captions(row["captions"], k=captions_per_image * 3)
        caps = [c for c in caps if c.lower() not in seen_caps][:captions_per_image]
        if not caps:
            stats["drop_no_valid_caption"] += 1
            continue
        seen_fp.add(fp)
        seen_caps.update(c.lower() for c in caps)
        fname = f"{stats['kept_images']:05d}.jpg"
        img.save(img_dir / fname, quality=92)
        for c in caps:
            records.append({"img_id": row["img_id"], "image_path": f"images/{fname}", "caption": c})
        stats["kept_images"] += 1

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No valid image-caption pairs were produced; check the source / filters.")

    # split by IMAGE (no image appears in two splits)
    ids = sorted(df["image_path"].unique())
    random.Random(seed).shuffle(ids)
    n_val, n_test = int(len(ids) * val_frac), int(len(ids) * test_frac)
    split_of = {p: "test" for p in ids[:n_test]}
    split_of.update({p: "val" for p in ids[n_test:n_test + n_val]})
    split_of.update({p: "train" for p in ids[n_test + n_val:]})
    df["split"] = df["image_path"].map(split_of)
    df["n_words"] = df["caption"].str.split().str.len()

    df.to_csv(out / "pairs.csv", index=False)
    stats.update({
        "n_pairs": int(len(df)),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().items()},
        "caption_words_mean": round(float(df["n_words"].mean()), 2),
        "caption_words_min": int(df["n_words"].min()),
        "caption_words_max": int(df["n_words"].max()),
        "image_size": IMG_SIZE,
    })
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return df


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hf", "local"], default="hf")
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--local-csv")
    ap.add_argument("--n-images", type=int, default=2000)
    ap.add_argument("--captions-per-image", type=int, default=1)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    rows = iter_hf(a.hf_repo, seed=a.seed) if a.source == "hf" else iter_local(a.local_csv)
    df = build_dataset(rows, a.out, a.n_images, a.captions_per_image, a.seed)
    print(f"Saved {len(df)} pairs -> {Path(a.out) / 'pairs.csv'}")
    print((Path(a.out) / "stats.json").read_text())
    return df


if __name__ == "__main__":
    main()
