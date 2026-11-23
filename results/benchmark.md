# Export and latency benchmark

Machine: Apple M3 Pro, macOS-26.6.2-arm64-arm-64bit; PyTorch 2.15.0, ONNX Runtime 1.32.0. Batch 1, full frame, 10 warm-up runs excluded, median of timed runs. Model latency excludes preprocessing (shown separately). F1 columns: test pigs P095, P096.

## Precision variants (ONNX Runtime CPU, 240 × 320)

| Model | Bands | Variant | Params | File | Test macro F1 (exported seed) | Pixel agreement vs fp32 | Max abs logit diff vs PyTorch | ORT CPU |
|---|---|---|---|---|---|---|---|---|
| Patch CNN | full | fp32 | 171k | 0.69 MB | 0.796 | 1.0000 | 1.6e-05 | 59.0 ms (17 fps) |
| Patch CNN | full | fp16 | 171k | 0.35 MB | 0.796 | 0.9993 | 6.7e-02 | 59.0 ms (17 fps) |
| Patch CNN | full | int8 | 171k | 0.19 MB | 0.792 | 0.9755 | 1.9e+00 | 30.7 ms (33 fps) |
| Patch CNN | snapshot16 | fp32 | 169k | 0.68 MB | 0.796 | 1.0000 | 1.5e-05 | 57.6 ms (17 fps) |
| Patch CNN | snapshot16 | fp16 | 169k | 0.34 MB | 0.796 | 0.9991 | 6.9e-02 | 57.7 ms (17 fps) |
| Patch CNN | snapshot16 | int8 | 169k | 0.18 MB | 0.792 | 0.9389 | 5.5e+00 | 29.6 ms (34 fps) |
| Pixel MLP | full | fp32 | 40k | 0.16 MB | 0.745 | 1.0000 | 3.1e-05 | 12.9 ms (77 fps) |
| Pixel MLP | full | fp16 | 40k | 0.08 MB | 0.745 | 0.9994 | 3.2e-02 | 13.3 ms (75 fps) |
| Pixel MLP | full | int8 | 40k | 0.05 MB | 0.736 | 0.9137 | 9.1e+00 | 13.1 ms (77 fps) |
| Pixel MLP | snapshot16 | fp32 | 29k | 0.11 MB | 0.722 | 1.0000 | 2.6e-05 | 10.4 ms (96 fps) |
| Pixel MLP | snapshot16 | fp16 | 29k | 0.06 MB | 0.722 | 0.9992 | 6.4e-02 | 10.4 ms (96 fps) |
| Pixel MLP | snapshot16 | int8 | 29k | 0.04 MB | 0.679 | 0.8486 | 9.8e+00 | 11.4 ms (88 fps) |
| U-Net | full | fp32 | 458k | 1.84 MB | 0.789 | 1.0000 | 1.2e-05 | 18.3 ms (55 fps) |
| U-Net | full | fp16 | 458k | 0.93 MB | 0.789 | 0.9991 | 3.9e-02 | 18.7 ms (54 fps) |
| U-Net | full | int8 | 458k | 0.50 MB | 0.781 | 0.9640 | 1.6e+00 | 12.1 ms (83 fps) |
| U-Net | snapshot16 | fp32 | 456k | 1.83 MB | 0.762 | 1.0000 | 2.1e-05 | 17.3 ms (58 fps) |
| U-Net | snapshot16 | fp16 | 456k | 0.92 MB | 0.762 | 0.9988 | 6.8e-02 | 17.5 ms (57 fps) |
| U-Net | snapshot16 | int8 | 456k | 0.49 MB | 0.769 | 0.9187 | 3.2e+00 | 11.2 ms (89 fps) |

## fp32 latency by runtime

| Model | Bands | Test macro F1 (3 seeds) | Preprocess (NumPy) | PyTorch CPU 240×320 | ORT CPU 240×320 | ORT CPU 480×640 | PyTorch MPS 240×320 | PyTorch MPS 480×640 |
|---|---|---|---|---|---|---|---|---|
| Patch CNN | full | 0.802 ± 0.008 | 13.1 ms (76 fps) | 97.5 ms (10 fps) | 59.0 ms (17 fps) | 239.4 ms (4 fps) | 8.5 ms (117 fps) | 33.5 ms (30 fps) |
| Patch CNN | snapshot16 | 0.786 ± 0.011 | 6.8 ms (147 fps) | 94.4 ms (11 fps) | 57.6 ms (17 fps) | 227.6 ms (4 fps) | 8.0 ms (126 fps) | 31.0 ms (32 fps) |
| Pixel MLP | full | 0.751 ± 0.006 | 14.8 ms (68 fps) | 25.2 ms (40 fps) | 12.9 ms (77 fps) | 52.5 ms (19 fps) | 5.4 ms (187 fps) | 19.9 ms (50 fps) |
| Pixel MLP | snapshot16 | 0.713 ± 0.012 | 6.6 ms (151 fps) | 22.8 ms (44 fps) | 10.4 ms (96 fps) | 42.3 ms (24 fps) | 5.0 ms (201 fps) | 19.1 ms (52 fps) |
| U-Net | full | 0.761 ± 0.024 | 14.7 ms (68 fps) | 46.5 ms (21 fps) | 18.3 ms (55 fps) | 74.6 ms (13 fps) | 3.8 ms (261 fps) | 14.7 ms (68 fps) |
| U-Net | snapshot16 | 0.762 ± 0.009 | 6.6 ms (152 fps) | 44.3 ms (23 fps) | 17.3 ms (58 fps) | 70.7 ms (14 fps) | 3.6 ms (278 fps) | 13.9 ms (72 fps) |

## Demo models

Rule: speed first, because both demos run on CPU-class backends (local Gradio app on ONNX Runtime CPU, browser demo on ONNX Runtime Web WASM). Among fp32 models with file < 10 MB, the lowest ONNX Runtime CPU latency at 240 × 320 wins; a fp16/int8 variant replaces fp32 only if it is smaller, agrees with fp32 on ≥ 99.5% of test pixels and is ≥ 10% faster (beyond timing noise).

- **full:** Pixel MLP (fp32) → `exports/pixel_mlp_full/model_fp32.onnx`
- **snapshot16:** Pixel MLP (fp32) → `exports/pixel_mlp_snapshot16/model_fp32.onnx`

## Notes

- **Parity:** fp32 ONNX vs PyTorch max absolute logit difference is ≤ 3.1e-5 on a real test frame for all
  six models; unit tests also cover dynamic input sizes.
- **fp16** (`keep_io_types`): same test macro F1 (±0.0002) and ≥ 99.88% pixel agreement with fp32, at half
  the file size. It is **not faster on ONNX Runtime CPU** (it computes in fp32 with casts), so the demo
  rule keeps fp32 (a variant must be ≥ 10% faster to replace it). On WebGPU fp16 may be faster; that is measured in the browser demo, not here.
- **int8** (dynamic quantisation): 1.5–1.9× faster on ONNX Runtime CPU for the conv nets, but only
  84.9–97.6% pixel agreement with fp32 and lower F1 (largest drop: 16-band pixel MLP, 0.722 → 0.679). It
  needs `ConvInteger` / `DynamicQuantizeLinear`, which ONNX Runtime Web's WebGPU backend is not expected
  to support. Static (calibrated) quantisation might do better; it was not tried.
- **Demo choice: speed over accuracy.** The pixel MLP is the fastest model on ONNX Runtime CPU (12.9 ms
  full, 10.4 ms 16-band at 240 × 320) and the smallest download (0.16 / 0.11 MB), so both demos use it.
  The cost is accuracy: test macro F1 is 0.751 (full) and 0.713 (16-band), against 0.802 / 0.786 for the
  patch CNN and 0.761 / 0.762 for the U-Net. Its errors are scattered per pixel (no spatial context),
  so overlays look noisier than with the spatial models. On a GPU (PyTorch MPS) the U-Net is faster than
  the MLP (3.6–3.8 ms vs 5.0–5.4 ms), so the fastest choice depends on the target hardware.
- **Real-time context:** model-only latency on the M3 Pro GPU (PyTorch MPS) at 480 × 640 is 13.9–14.7 ms
  for the U-Net (~70 fps) and 19.1–19.9 ms for the pixel MLP (~50 fps). In this Python pipeline NumPy preprocessing
  (6.6 ms for 16 bands, 14.7 ms for 100 bands) costs as much as or more than the model at 240 × 320. A real system would
  fuse it into the model or run it on the GPU. 480 × 640 inputs are 2× nearest-upsampled 240 × 320 frames
  (timing only).
- The 16-band models are only marginally faster than the full-spectrum ones (the first layer is a small
  part of the compute). The 16-band benefit is in the sensor (snapshot capture at video rate), not
  in model latency.
