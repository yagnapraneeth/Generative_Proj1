# GenAI Text → Image: Data, Embeddings & Baseline

Week-1 deliverable: dataset selection and cleaning, environment setup (PyTorch + Diffusers + CLIP), a baseline text→embedding notebook, and 5 prompt→image test runs.

## Pipeline
| Step | Module | Output |
|---|---|---|
| 1. Choose dataset | Flickr30k (streamed from the HF Hub) | – |
| 2. Clean & preprocess 2,000 pairs | `src/data_prep.py` | `data/processed/{images/,pairs.csv,stats.json}` |
| 3. Environment | `requirements.txt` | PyTorch, Diffusers, Transformers (CLIP) |
| 4. Baseline text→embedding | `src/embed.py` | CLIP embeddings + retrieval Recall@k, SD 77×768 conditioning |
| 5. Five test runs | `src/generate.py` | `outputs/test_runs/run_1..5.png`, `grid.png`, `results.json` |

Cleaning rules: Unicode/whitespace normalisation, punctuation fixes, 5–40 word captions, URL/symbol-heavy captions dropped, exact-duplicate captions and near-duplicate images removed, images < 256 px dropped, center-crop to 256×256, 80/10/10 train/val/test split **by image** (no leakage).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data_prep --n-images 2000 --out data/processed
python -m src.embed --pairs data/processed/pairs.csv --out data/embeddings
python -m src.generate --out outputs/test_runs
# or run everything interactively:
jupyter notebook notebooks/01_baseline_text_to_image.ipynb
```
No GPU? Use Google Colab (free T4): open the notebook, set `REPO_URL` in the first cell, run all.

Offline sanity tests (no GPU/network): `python tests/test_pipeline.py`

## Test prompts (seeds 42–46, DPM-Solver++, guidance 7.5)
1. A man riding a bicycle down a busy city street at sunset.
2. Two children playing with a golden retriever in a grassy park.
3. A group of friends laughing around a campfire at night.
4. A woman in a red dress walking through a crowded outdoor market.
5. A dog leaping to catch a frisbee on a sandy beach.

Results (images + CLIP scores) are written to `outputs/test_runs/` after you run `src/generate.py`.

## Repo layout
```
src/            data_prep.py  embed.py  generate.py
notebooks/      01_baseline_text_to_image.ipynb
tests/          test_pipeline.py
docs/           proposal.docx
outputs/        test_runs/
```

## Notes
- Flickr30k is for non-commercial research use; Stable Diffusion v1.5 is under the CreativeML Open RAIL-M license. The safety checker is left enabled.
- Model weights (~4 GB) download on first run and are cached by Hugging Face.
