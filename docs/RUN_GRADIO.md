# Running the Gradio demo

The Gradio app (`app/`) is the server-side variant of the demo and runs **locally only**: it listens
on `127.0.0.1` and never creates a public share link. The public, always-on demo is the browser
version (`web/`, ONNX Runtime Web), deployed as a static site; see [DEPLOY_WEB.md](DEPLOY_WEB.md).

## Run

```bash
make app                  # first run creates .venv-app from app/requirements.txt
# open http://127.0.0.1:7860
```

`make app` uses a separate virtualenv built only from `app/requirements.txt` (gradio 6.32.0,
onnxruntime 1.32.0, numpy 2.4.6, matplotlib 3.11.2), so the app's dependencies are tested in
isolation from the training environment. Another port:

```bash
cd app && ../.venv-app/bin/python app.py --port 7861
```

## Check it works

1. Pick each sample; the overlay and legend update. The kidney sample is the known failure case.
2. Switch *Full spectrum* / *Simulated 16-band*; the latency line shows the band count.
3. Click a pixel; the spectrum plot and pixel probabilities update.
4. Press **Run 50-frame benchmark**; on an Apple M3 Pro the median was 9.8 ms per 120×160 frame
   (about 100 fps), preprocessing included.

## Update after retraining

```bash
make export demo-models samples   # re-export ONNX, copy demo models, regenerate samples
make test                          # includes app parity tests (tests/test_app.py)
make app
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Address already in use` | another app on 7860: `--port 7861` |
| Model fails to load / sample has wrong size | LFS files missing: `git lfs install && git lfs pull` |
| `ModuleNotFoundError` | delete `.venv-app` and run `make app` again |
