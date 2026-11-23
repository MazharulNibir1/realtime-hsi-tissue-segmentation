# Gradio demo (local)

Server-side version of the demo: Gradio UI + ONNX Runtime on CPU. It runs on your own machine only
(`127.0.0.1`, no public link). The **public demo** is the browser version in [`../web`](../web), which
runs the same models in the visitor's browser.

```bash
make app                  # from the repo root; creates .venv-app from app/requirements.txt
# open http://127.0.0.1:7860
```

or, inside `app/` with any Python 3.10–3.12 environment:

```bash
pip install -r requirements.txt
python app.py             # --port 7861 for another port
```

## What it shows

- 5 sample frames from pigs **never used in training** (kidney included as the model's known failure
  case), or your own `.npy` cube (H×W×100, 500–1000 nm, max 512 px per side).
- Toggle: full 100-band spectrum vs simulated 16-band snapshot sensor.
- Pseudo-RGB, predicted-tissue overlay with legend (annotated region outlined in white), mean class
  confidence, overlay opacity and confidence threshold.
- Click any pixel: its spectrum vs the training-set class means, and its class probabilities.
- Measured latency per frame and a 50-frame benchmark.

**Model:** per-pixel network with 29k–40k parameters (1×1 convolutions), the fastest of the three
architectures trained. Test macro F1 is 0.751 (100 bands) and 0.713 (16 bands); the spatial patch
CNN was more accurate (0.802 / 0.786) but 4–5× slower on CPU (59 ms vs 13 ms per 240×320 frame).
See [`../results`](../results).

## Files

| Path | Content |
|---|---|
| `app.py` | Gradio UI |
| `inference.py` | NumPy + ONNX Runtime inference; port of the training-side input transform (parity-tested in `tests/test_app.py`) |
| `models/<full\|snapshot16>/` | `model.onnx`, `transform.json` (band matrix, normalisation statistics), `model.json` |
| `samples/` | 120×160×100 float16 cubes + label maps + metadata (CC BY 4.0, see `samples/LICENSE.md`) |

**Research demo only, not for clinical use.** Data: HeiPorSPECTRAL (CC BY 4.0), Studier-Fischer et
al., Scientific Data 10, 414 (2023), doi:10.1038/s41597-023-02315-8. Code: MIT.
