---
title: Hyperspectral Tissue Classification
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Surgical HSI organ classification, 100 vs 16 bands (ONNX)
---

# Hyperspectral tissue classification: full spectrum vs simulated 16-band sensor

Interactive demo for per-pixel organ classification on porcine surgical hyperspectral images
(HeiPorSPECTRAL: 20 organs, 500–1000 nm). It compares the full 100-band spectrum with a simulated
16-band snapshot mosaic sensor.

- **Model:** 29k–40k-parameter per-pixel network (1×1 convolutions), ONNX Runtime on CPU. It is the
  fastest of the three architectures trained; test macro F1 is 0.751 (100 bands) and 0.713 (16 bands).
  A spatial patch CNN was more accurate (0.802 / 0.786) but 4–5× slower on CPU (59 ms vs 13 ms per 240×320 frame).
- **Samples:** 5 frames from pigs that were never used for training. Kidney is included as the
  model's known failure case (mostly confused with liver).
- **Interaction:** click any pixel to compare its spectrum with the training-set class means. The
  run-benchmark button measures latency on this server.

Training code, evaluation, and the in-browser (ONNX Runtime Web) version:
https://github.com/MazharulNibir1/realtime-hsi-tissue-segmentation

## Files

| Path | Content |
|---|---|
| `app.py` | Gradio UI |
| `inference.py` | NumPy + ONNX Runtime inference; port of the training-side input transform (parity-tested) |
| `models/<full\|snapshot16>/` | `model.onnx`, `transform.json` (band matrix, normalisation statistics), `model.json` |
| `samples/` | 120×160×100 float16 cubes + label maps + metadata (CC BY 4.0, see `samples/LICENSE.md`) |

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

**Research demo only, not for clinical use.** Data: HeiPorSPECTRAL (CC BY 4.0), Studier-Fischer et
al., Scientific Data 10, 414 (2023), doi:10.1038/s41597-023-02315-8. Code: MIT.
