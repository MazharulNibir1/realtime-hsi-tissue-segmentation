"""Gradio demo: hyperspectral tissue classification, full spectrum vs simulated 16-band sensor.

Server-side ONNX Runtime (CPU). Research demo only - not for clinical use.
"""

from __future__ import annotations

import time

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from inference import (  # noqa: E402
    IGNORE,
    load_index,
    load_models,
    load_sample,
    load_user_cube,
    overlay,
    pseudo_rgb,
    upscale,
)

INDEX = load_index()
MODELS = load_models()
CLASSES = INDEX["class_names"]
COLOURS = INDEX["class_colours"]
WAVELENGTHS = np.asarray(INDEX["wavelengths_nm"], dtype=np.float32)
SCALE = 3  # display upscaling of the 120 x 160 samples
MODEL_CHOICES = {MODELS[k].info["label"]: k for k in ("full", "snapshot16") if k in MODELS}
SAMPLE_CHOICES = {f"{s['organ']} ({s['subject']}, test pig)": s["id"] for s in INDEX["samples"]}


def pretty(name: str) -> str:
    return name.replace("kidney_with_Gerotas_fascia", "kidney + Gerota's fascia").replace("_", " ")


def legend_html(pred: np.ndarray, gt_organ: str | None) -> str:
    counts = np.bincount(pred.ravel(), minlength=len(CLASSES))
    order = [i for i in np.argsort(-counts) if counts[i] > 0]
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
        f'<span style="width:14px;height:14px;border-radius:3px;background:{COLOURS[i]};'
        f'border:1px solid rgba(0,0,0,.25)"></span>'
        f"<span>{pretty(CLASSES[i])}</span>"
        f'<span style="margin-left:auto;opacity:.7">{100 * counts[i] / pred.size:.1f}%</span></div>'
        for i in order
    )
    note = (
        f"<p style='margin:8px 0 0;opacity:.8'>White outline: annotated region "
        f"(ground truth: <b>{pretty(gt_organ)}</b>). Only one organ per frame is annotated; "
        f"predictions elsewhere are unverified.</p>"
        if gt_organ
        else "<p style='margin:8px 0 0;opacity:.8'>No annotation for uploaded cubes.</p>"
    )
    return (
        f"<div style='font-size:14px'><b>Predicted classes (share of frame)</b>{rows}{note}</div>"
    )


def confidences(probs: np.ndarray, labels: np.ndarray | None) -> dict:
    region = labels != IGNORE if labels is not None and (labels != IGNORE).any() else None
    p = probs[region] if region is not None else probs.reshape(-1, probs.shape[-1])
    mean = p.mean(0)
    return {pretty(CLASSES[i]): float(mean[i]) for i in np.argsort(-mean)[:5]}


def latency_md(t: dict, shape, model_key: str, bench: dict | None = None) -> str:
    total = t["preprocess_ms"] + t["inference_ms"]
    s = (
        f"**Latency** (ONNX Runtime, CPU, this server): preprocessing {t['preprocess_ms']:.1f} ms + "
        f"inference {t['inference_ms']:.1f} ms = **{total:.1f} ms** (≈ {1000 / total:.0f} fps) "
        f"for a {shape[0]}×{shape[1]}×{shape[2]} cube → {MODELS[model_key].transform.n_bands} model bands."
    )
    if bench:
        s += (
            f"\n\n**Benchmark, {bench['n']} frames** (warm-up excluded): median {bench['median']:.1f} ms, "
            f"p90 {bench['p90']:.1f} ms → **{1000 / bench['median']:.0f} fps**."
        )
    return s


def spectrum_figure(cube, probs, r, c, model_key, labels):
    m = MODELS[model_key]
    tr = m.transform
    pix = tr.project_l1(cube[r, c][None])[0]
    x = WAVELENGTHS if model_key == "full" else tr.band_centres
    stats = INDEX["class_mean_spectra"][model_key]
    mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
    top = np.argsort(-probs[r, c])[:2]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for rank, k in enumerate(top):
        ax.fill_between(x, mean[k] - std[k], mean[k] + std[k], color=COLOURS[k], alpha=0.15, lw=0)
        ax.plot(x, mean[k], color=COLOURS[k], lw=2, ls="-" if rank == 0 else "--",
                label=f"{pretty(CLASSES[k])} mean ± sd (p={probs[r, c, k]:.2f})")  # fmt: skip
    ax.plot(x, pix, color="#0b0b0b", lw=1.6, marker="o" if len(x) <= 16 else None, ms=3.5,
            label=f"clicked pixel (row {r}, col {c})")  # fmt: skip
    gt = labels[r, c] if labels is not None else IGNORE
    title = f"Pixel spectrum vs class means ({tr.n_bands} bands, L1-normalised)"
    if gt != IGNORE:
        title += f": ground truth {pretty(CLASSES[gt])}"
    ax.set_title(title, loc="left", fontsize=9.5, color="#0b0b0b")
    ax.set_xlabel("wavelength (nm)", color="#52514e")
    ax.grid(True, color="#e1e0d9", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#898781", labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="best")
    fig.tight_layout()
    return fig


def run(sample_label, model_label, alpha, min_conf, upload):
    model_key = MODEL_CHOICES[model_label]
    if upload is not None:
        cube = load_user_cube(np.load(upload, allow_pickle=False))
        labels, meta, recipe = None, {"organ": None}, None
    else:
        cube, labels, meta = load_sample(SAMPLE_CHOICES[sample_label])
        recipe = meta["pseudo_rgb"]
    probs, t = MODELS[model_key].predict(cube)
    pred = probs.argmax(-1)
    rgb = pseudo_rgb(cube, recipe, WAVELENGTHS)
    gt = labels != IGNORE if labels is not None else None
    conf_map = probs.max(-1)
    over = overlay(rgb, pred, COLOURS, alpha, gt, conf_map, min_conf)
    state = {
        "cube": cube,
        "labels": labels,
        "probs": probs,
        "model": model_key,
        "rgb": rgb,
        "pred": pred,
        "gt": gt,
        "conf": conf_map,
    }
    acc = ""
    if gt is not None and gt.any():
        acc = f" Pixel accuracy inside the annotated region: **{(pred[gt] == labels[gt]).mean():.1%}**."
    r, c = (
        np.argwhere(gt)[len(np.argwhere(gt)) // 2]
        if gt is not None and gt.any()
        else (cube.shape[0] // 2, cube.shape[1] // 2)
    )
    fig = spectrum_figure(cube, probs, int(r), int(c), model_key, labels)
    return (
        upscale(rgb, SCALE),
        upscale(over, SCALE),
        legend_html(pred, meta.get("organ")),
        confidences(probs, labels),
        latency_md(t, cube.shape, model_key) + acc,
        fig,
        state,
    )


def on_click(state, alpha, evt: gr.SelectData):
    if not state:
        return gr.update(), gr.update()
    x, y = evt.index
    r = int(np.clip(y // SCALE, 0, state["cube"].shape[0] - 1))
    c = int(np.clip(x // SCALE, 0, state["cube"].shape[1] - 1))
    fig = spectrum_figure(state["cube"], state["probs"], r, c, state["model"], state["labels"])
    p = state["probs"][r, c]
    pixel_conf = {pretty(CLASSES[i]): float(p[i]) for i in np.argsort(-p)[:5]}
    return fig, pixel_conf


def restyle(state, alpha, min_conf):
    if not state:
        return gr.update()
    over = overlay(
        state["rgb"], state["pred"], COLOURS, alpha, state["gt"], state["conf"], min_conf
    )
    return upscale(over, SCALE)


def benchmark(state, n=50):
    if not state:
        return gr.update()
    m = MODELS[state["model"]]
    for _ in range(5):
        m.predict(state["cube"])
    ms, last = [], None
    for _ in range(n):
        t0 = time.perf_counter()
        _, last = m.predict(state["cube"])
        ms.append((time.perf_counter() - t0) * 1e3)
    return latency_md(last, state["cube"].shape, state["model"],
                      {"n": n, "median": float(np.median(ms)), "p90": float(np.percentile(ms, 90))})  # fmt: skip


INTRO = """
# Hyperspectral tissue classification: full spectrum vs 16-band snapshot sensor

Per-pixel organ classification on porcine surgical hyperspectral images (HeiPorSPECTRAL, 20 organs,
500–1000 nm). Compare the **full 100-band spectrum** with a **simulated 16-band snapshot mosaic sensor**
(the kind of camera that captures video-rate hyperspectral frames). The model is a small (29k–40k parameters) per-pixel
network running in ONNX Runtime on this server's CPU. It was chosen as the fastest of three architectures;
spatial models were more accurate (test macro F1 0.80 vs 0.75).

Pick a sample (all from pigs **never seen in training**), then **click any pixel** to compare its spectrum
with the class means.
"""

FOOTER = f"""
---
**Research demo only, not for clinical use.** Not validated on humans or in real surgery.
Data: HeiPorSPECTRAL (CC BY 4.0), {INDEX['citation']} Changes: {INDEX['changes']}
"""

with gr.Blocks(title="Hyperspectral tissue classification") as demo:
    gr.Markdown(INTRO)
    state = gr.State(None)
    with gr.Row():
        sample = gr.Dropdown(
            list(SAMPLE_CHOICES), value=next(iter(SAMPLE_CHOICES)), label="Sample (test pigs)"
        )
        model = gr.Radio(list(MODEL_CHOICES), value=next(iter(MODEL_CHOICES)), label="Model input")
        alpha = gr.Slider(0, 1, value=0.5, step=0.05, label="Overlay opacity")
        min_conf = gr.Slider(
            0, 0.95, value=0.0, step=0.05, label="Hide pixels with confidence below"
        )
    with gr.Accordion("Or load your own cube (.npy, shape H×W×100, 500–1000 nm)", open=False):
        upload = gr.File(
            file_types=[".npy"], type="filepath", label="Upload .npy (max 512 px per side)"
        )
        gr.Markdown("Values may be reflectance or already L1-normalised; each pixel is L1-normalised first. "
                    "Clear the file to return to the samples.")  # fmt: skip
    with gr.Row():
        rgb_img = gr.Image(
            label="Pseudo-RGB (650 / 570 / 510 nm), click a pixel", interactive=False
        )
        over_img = gr.Image(label="Predicted tissue overlay, click a pixel", interactive=False)
    with gr.Row():
        with gr.Column(scale=1):
            legend = gr.HTML()
        with gr.Column(scale=1):
            conf = gr.Label(num_top_classes=5, label="Mean class confidence (annotated region)")
            pixel_conf = gr.Label(num_top_classes=5, label="Clicked pixel: class probabilities")
    spectrum = gr.Plot(label="Spectrum of the clicked pixel")
    with gr.Row():
        latency = gr.Markdown()
        bench_btn = gr.Button("Run 50-frame benchmark", variant="secondary")
    gr.Markdown(FOOTER)

    inputs = [sample, model, alpha, min_conf, upload]
    outputs = [rgb_img, over_img, legend, conf, latency, spectrum, state]
    for comp in (sample, model, upload):
        comp.change(run, inputs, outputs)
    alpha.release(restyle, [state, alpha, min_conf], over_img)
    min_conf.release(restyle, [state, alpha, min_conf], over_img)
    rgb_img.select(on_click, [state, alpha], [spectrum, pixel_conf])
    over_img.select(on_click, [state, alpha], [spectrum, pixel_conf])
    bench_btn.click(benchmark, state, latency)
    demo.load(run, inputs, outputs)

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Hyperspectral tissue classification demo")
    ap.add_argument("--port", type=int, default=None, help="local port (default: Gradio's 7860)")
    args = ap.parse_args()
    # Local only: listen on localhost, never create a public share link.
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False)
