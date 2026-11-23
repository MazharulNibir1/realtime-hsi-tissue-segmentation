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
