# Browser demo (ONNX Runtime Web)

The public demo: per-pixel organ classification on surgical hyperspectral frames, running entirely in
the visitor's browser. Vite + vanilla TypeScript, no UI framework, no server-side code.

```bash
npm ci
npm run dev        # http://localhost:5173 (sends the COOP/COEP headers, see below)
npm test           # parity test: TypeScript preprocessing + onnxruntime-web vs Python
npm run build      # typecheck + production build into dist/
npm run preview    # serve dist/ with the same headers as production
```

From the repo root: `make web-install web-test web-build`. Deployment: [`docs/DEPLOY_WEB.md`](../docs/DEPLOY_WEB.md).

`npm run build` first runs `scripts/check-assets.mjs`, which fails if the models or samples are Git LFS
pointer files (a clone without LFS). After deploying, `scripts/verify-deploy.sh <url>` checks the headers,
the `.wasm` content type and compression, and that real model and sample files are served.

## How inference runs

1. **Runtime.** If the browser offers a WebGPU adapter, the full `onnxruntime-web` build loads and the
   session uses the WebGPU execution provider (falling back to WASM if session creation fails);
   otherwise the smaller WASM-only build (`onnxruntime-web/wasm`) loads. Force one with
   `?backend=wasm` or `?backend=webgpu`. Vite emits the `.wasm` binaries as content-hashed assets.
2. **Models** (`public/models/<full|snapshot16>/`): `model.onnx` (pixel MLP, fp32), `transform.json`
   (band matrix W, normalisation statistics), `model.json` (provenance, size, test F1). Each model
   is downloaded the first time it is selected.
3. **Samples** (`public/samples/`): 120×160×100 float16 cubes, uint8 label maps, JSON metadata,
   identical to the Gradio app's. CC BY 4.0, see `public/samples/LICENSE.md`.
4. **Preprocessing** (`src/core.ts`, `SpectralTransform`) is a port of
   `hsi.data.normalize.SpectralTransform`. For each pixel x (100 L1-normalised channels):
   - `y = W x` (identity for the full spectrum; the 16 × 100 box filter is applied as sparse rows);
   - `y = y / max(Σ|y|, 1e-8)`;
   - `y = (y − mean) / std` per band, with the training-set statistics from `transform.json`.

   The output is written in CHW order as the model's `[1, bands, H, W]` float32 input.

## Parity test

`scripts/make_web_reference.py` (repo root) runs the Python pipeline on two samples and both models and
writes `tests/fixtures/reference.json`. `tests/parity.test.ts` runs the TypeScript preprocessing and
`onnxruntime-web` (WASM backend, in Node) on the same files and requires:

| Check | Tolerance | Measured (2026-11-25) |
|---|---|---|
| Transformed input at 64 pixels | < 1e-5 | ≤ 1.2e-6 |
| Logits at 64 pixels | < 1e-3 | ≤ 6.4e-5 |
| Argmax agreement, whole frame | ≥ 99.9% | 99.995–100% |
| Annotated-pixel accuracy | equal to 3 decimals | equal |

Regenerate the reference after changing models or samples:
`.venv/bin/python scripts/make_web_reference.py`.

## Headers: cross-origin isolation

Multithreaded WASM needs `SharedArrayBuffer`, which browsers only enable on cross-origin isolated
pages. `vercel.json` (production) and `vite.config.ts` (dev and preview) send:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

With `require-corp`, every subresource must be same-origin or send CORP/CORS headers, so the fonts
are self-hosted (`@fontsource/public-sans`) and nothing is loaded from a third party. Without
isolation the demo still works, with a single WASM thread; the runtime panel says which applies.

`vercel.json` also sets `application/wasm` for `.wasm`, one-year immutable caching for the hashed
`/assets/*`, and one-hour caching with revalidation for models and samples (fixed file names).

## Measured in headless Chrome (Apple M3 Pro, 120×160 frame, 16-band model)

| Backend | 50-frame median | fps | Runtime download |
|---|---|---|---|
| WebGPU | 10 ms (preprocess 4.1, model 3.2, draw 2.7) | 99 | 29.5 MB uncompressed |
| WASM, 4 threads | 13 ms (preprocess 3.6, model 6.3, draw 2.9) | 78 | 15.1 MB uncompressed |

Other devices will differ; the page measures and shows its own numbers. Research demo only, not
for clinical use.
