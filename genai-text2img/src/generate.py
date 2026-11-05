"""Prompt -> image test runs with Stable Diffusion v1.5 (Diffusers).

Generates the 5 required test images, a contact-sheet grid, and a
results.json with prompt, seed, sampler settings and a CLIP score
(100 * cosine similarity between prompt and image, CLIP ViT-B/32).

Usage:
    python -m src.generate --out outputs/test_runs
    python -m src.generate --steps 30 --size 512 --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CLIP_ID = "openai/clip-vit-base-patch32"

# 5 prompts in the style of the Flickr30k domain (everyday scenes, people, animals)
TEST_PROMPTS = [
    "A man riding a bicycle down a busy city street at sunset.",
    "Two children playing with a golden retriever in a grassy park.",
    "A group of friends laughing around a campfire at night.",
    "A woman in a red dress walking through a crowded outdoor market.",
    "A dog leaping to catch a frisbee on a sandy beach.",
]


def pick_device_and_defaults(size: int | None, steps: int | None):
    import torch

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
        size, steps = size or 512, steps or 30
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
        size, steps = size or 512, steps or 25
    else:
        device, dtype = "cpu", torch.float32
        size, steps = size or 384, steps or 20   # keep CPU runs tractable
    return device, dtype, size, steps


def clip_scores(prompts, images, device):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    scores = []
    for p, im in zip(prompts, images):
        inp = proc(text=[p], images=im, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            out = model(**inp)
        t = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        v = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        scores.append(round(float(100 * (t * v).sum().clamp(min=0)), 2))
    del model
    return scores


def make_grid(images, prompts, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap

    fig, axes = plt.subplots(1, len(images), figsize=(3.6 * len(images), 4.4))
    for ax, im, p in zip(axes, images, prompts):
        ax.imshow(im)
        ax.set_title("\n".join(textwrap.wrap(p, 30)), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(prompts=None, out_dir="outputs/test_runs", model_id=SD_ID, seed=42, steps=None, size=None,
        guidance=7.5, negative_prompt="blurry, low quality, deformed, distorted, text, watermark"):
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    prompts = prompts or TEST_PROMPTS
    device, dtype, size, steps = pick_device_and_defaults(size, steps)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} dtype={dtype} size={size} steps={steps}")

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    if device == "cuda":
        pipe.enable_attention_slicing()

    images, records = [], []
    for i, prompt in enumerate(prompts, 1):
        s = seed + i - 1
        gen = torch.Generator(device="cpu").manual_seed(s)   # CPU generator -> reproducible across devices
        t0 = time.time()
        res = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps,
                   guidance_scale=guidance, height=size, width=size, generator=gen)
        img = res.images[0]
        flagged = bool(res.nsfw_content_detected[0]) if getattr(res, "nsfw_content_detected", None) else False
        fname = f"run_{i}.png"
        img.save(out / fname)
        images.append(img)
        records.append({"run": i, "prompt": prompt, "seed": s, "steps": steps, "guidance_scale": guidance,
                        "size": size, "file": fname, "safety_checker_flagged": flagged,
                        "seconds": round(time.time() - t0, 1)})
        print(f"[{i}/{len(prompts)}] {fname} ({records[-1]['seconds']}s) <- {prompt}")

    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    for rec, sc in zip(records, clip_scores(prompts, images, device)):
        rec["clip_score"] = sc
    meta = {"model": model_id, "scheduler": "DPMSolverMultistep", "device": device,
            "mean_clip_score": round(sum(r["clip_score"] for r in records) / len(records), 2), "runs": records}
    (out / "results.json").write_text(json.dumps(meta, indent=2))
    make_grid(images, prompts, out / "grid.png")
    print(f"Saved {len(images)} images + grid.png + results.json -> {out}")
    return images, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/test_runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--size", type=int)
    ap.add_argument("--guidance", type=float, default=7.5)
    a = ap.parse_args()
    run(out_dir=a.out, seed=a.seed, steps=a.steps, size=a.size, guidance=a.guidance)
