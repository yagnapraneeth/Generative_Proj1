"""Offline tests: no GPU, no network, no torch needed.  Run: python -m pytest -q  (or python tests/test_pipeline.py)"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_prep, embed  # noqa: E402


def test_clean_caption():
    assert data_prep.clean_caption("a man  riding a horse .") == "A man riding a horse."
    assert data_prep.clean_caption("Two dogs run!!") is None            # too short (<5 words)
    assert data_prep.clean_caption("see http://x.com for a very long caption here") is None
    assert data_prep.clean_caption("12345 67890 11111 22222 33333 44444") is None   # not alphabetic
    assert data_prep.clean_caption(None) is None
    assert data_prep.clean_caption("a " * 60) is None                   # too long


def test_pick_captions_dedup_and_target_len():
    caps = ["a man rides a bike", "A man rides a bike.", "a man in a blue jacket rides a red bike down the road", "x"]
    got = data_prep.pick_captions(caps, k=2)
    assert len(got) == 2 and len({g.lower() for g in got}) == 2


def test_prepare_image_rejects_small_and_crops():
    assert data_prep.prepare_image(Image.new("RGB", (100, 300))) is None
    out = data_prep.prepare_image(Image.new("RGB", (640, 400), "red"))
    assert out.size == (data_prep.IMG_SIZE, data_prep.IMG_SIZE)


def _fake_rows(n):
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
        yield {"img_id": f"img{i}", "image": Image.fromarray(arr),
               "captions": [f"a person number {i} walks along the quiet street .", "bad", f"  Someone  stands near object {i} outside today "]}


def test_build_dataset_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        rows = list(_fake_rows(40))
        rows.insert(3, {"img_id": "tiny", "image": Image.new("RGB", (50, 50)), "captions": ["a tiny image caption here ok"]})
        rows.insert(5, dict(rows[0]))                                     # exact duplicate image
        df = data_prep.build_dataset(rows, d, n_images=30, captions_per_image=1)
        assert len(df) == 30 and df["image_path"].nunique() == 30
        assert set(df["split"]) == {"train", "val", "test"}
        # no image leaks across splits
        assert (df.groupby("image_path")["split"].nunique() == 1).all()
        assert all((Path(d) / p).exists() for p in df["image_path"])
        stats = json.loads((Path(d) / "stats.json").read_text())
        assert stats["drop_small_or_bad_image"] >= 1 and stats["drop_duplicate_image"] >= 1
        assert df["n_words"].between(data_prep.MIN_WORDS, data_prep.MAX_WORDS).all()


def test_local_source_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        recs = []
        for i, r in enumerate(_fake_rows(5)):
            r["image"].save(d / f"p{i}.jpg")
            for c in r["captions"]:
                recs.append({"image_path": f"p{i}.jpg", "caption": c})
        pd.DataFrame(recs).to_csv(d / "in.csv", index=False)
        df = data_prep.build_dataset(data_prep.iter_local(str(d / "in.csv")), d / "out", n_images=5)
        assert len(df) == 5


def test_retrieval_metrics_perfect_and_random():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 16)); x /= np.linalg.norm(x, axis=1, keepdims=True)
    m = embed.retrieval_metrics(x, x)
    assert m["t2i_R@1"] == 1.0 and m["i2t_R@1"] == 1.0 and m["t2i_median_rank"] == 1
    y = rng.normal(size=(50, 16)); y /= np.linalg.norm(y, axis=1, keepdims=True)
    assert embed.retrieval_metrics(x, y)["t2i_R@1"] < 0.3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
