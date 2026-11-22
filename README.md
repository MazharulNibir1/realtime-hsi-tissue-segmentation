# Hyperspectral Tissue Classification (real-time oriented)

Work in progress. Tissue classification on surgical hyperspectral imaging data, comparing the
full spectrum against a simulated 16-band snapshot mosaic sensor, with two live demos
(Gradio on Hugging Face Spaces, and in-browser inference with ONNX Runtime Web).

Research demo only — not for clinical use.

## Setup

```bash
make setup   # creates .venv (Python 3.11), installs package + dev tools
make test
```
