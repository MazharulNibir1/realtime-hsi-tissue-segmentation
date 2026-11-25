import "@fontsource/public-sans/400.css";
import "@fontsource/public-sans/600.css";
import "./style.css";
import {
  SpectralTransform,
  annotatedAccuracy,
  decode,
  decodeFloat16,
  l1NormaliseCube,
  meanConfidence,
  parseNpy,
  type ModelInfo,
  type Prediction,
  type SampleMeta,
  type TransformSpec,
} from "./core";
import { createSession, loadRuntime, transferredBytes, type LoadedModel, type Runtime } from "./runtime";
import { drawChart, hexToRgb, overlay, paint, percentileRecipe, pixelAt, pseudoRgb, type RgbRecipe, type Series } from "./render";

// ---------------------------------------------------------------- types and state

interface Index {
  samples: { id: string; organ: string; subject: string; demo_model_pixel_accuracy: Record<string, number> }[];
  class_names: string[];
  class_colours: string[];
  wavelengths_nm: number[];
  class_mean_spectra: Record<string, { band_centres_nm: number[]; mean: number[][]; std: number[][] }>;
}

interface Frame {
  id: string;
  title: string;
  cube: Float32Array; // HWC, L1-normalised, 100 bands
  h: number;
  w: number;
  labels: Uint8Array | null;
  organ: string | null;
  recipe: RgbRecipe;
  rgb: Uint8ClampedArray;
}

interface ModelEntry {
  info: ModelInfo;
  transform: SpectralTransform;
  loaded?: LoadedModel;
  loading?: Promise<LoadedModel>;
}

interface Timing { pre: number; infer: number; post: number; total: number }

const BANDS = ["snapshot16", "full"] as const;
type Bands = (typeof BANDS)[number];
const BASE = import.meta.env.BASE_URL;
const NIN = 100;
const SLOW_MS = 250;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const ui = {
  title: $("title"), finding: $("finding"), samples: $("samples"), models: $("models"),
  alpha: $<HTMLInputElement>("alpha"), minconf: $<HTMLInputElement>("minconf"), confOut: $<HTMLOutputElement>("conf-out"),
  upload: $<HTMLInputElement>("upload"), warning: $("warning"), rgb: $<HTMLCanvasElement>("rgb"),
  over: $<HTMLCanvasElement>("over"), chart: $<HTMLCanvasElement>("chart"), chartLegend: $("chart-legend"),
  pixelInfo: $("pixel-info"), legend: $("legend"), confidence: $("confidence"), pixelProbs: $("pixel-probs"),
  runtime: $("runtime"), live: $<HTMLButtonElement>("live"), bench: $<HTMLButtonElement>("bench"),
};

let index: Index;
let rt: Runtime;
let colours: [number, number, number][];
const models = new Map<Bands, ModelEntry>();
let bands: Bands = "snapshot16";
let frame: Frame;
let pred: Prediction | null = null;
let selected: [number, number] | null = null;
let last: Timing | null = null;
let bench: { median: number; p90: number; n: number; backend: string } | null = null;
let liveOn = false;
const liveTimes: number[] = [];
let busy = false;

// ---------------------------------------------------------------- helpers

const fmtMs = (ms: number) => (ms < 10 ? ms.toFixed(1) : ms.toFixed(0));
const fmtKb = (b: number) => (b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.round(b / 1e3)} kB`);
const pretty = (n: string) => n.replace("kidney_with_Gerotas_fascia", "kidney + Gerota's fascia").replaceAll("_", " ");
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
const median = (a: number[]) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
const pct = (a: number[], q: number) => [...a].sort((x, y) => x - y)[Math.min(a.length - 1, Math.floor(q * a.length))];

async function fetchBytes(path: string): Promise<Uint8Array> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}
async function fetchJson<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json() as Promise<T>;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, string> = {}, text?: string) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}

function warn(msg: string | null) {
  ui.warning.hidden = !msg;
  ui.warning.textContent = msg ?? "";
}

// ---------------------------------------------------------------- loading

async function loadSample(id: string): Promise<Frame> {
  const meta = await fetchJson<SampleMeta>(`samples/${id}.json`);
  const [h, w] = meta.shape;
  const [cubeBytes, labels] = await Promise.all([fetchBytes(`samples/${meta.cube_file}`), fetchBytes(`samples/${meta.labels_file}`)]);
  const cube = decodeFloat16(cubeBytes);
  return {
    id, h, w, cube, labels, organ: meta.organ, recipe: meta.pseudo_rgb,
    title: `${cap(pretty(meta.organ))}, pig ${meta.subject}`,
    rgb: pseudoRgb(cube, h * w, meta.pseudo_rgb, NIN),
  };
}

async function ensureModel(b: Bands): Promise<LoadedModel> {
  const m = models.get(b)!;
  if (m.loaded) return m.loaded;
  if (!m.loading) {
    ui.finding.textContent = `Downloading the ${m.info.label.toLowerCase()} model (${fmtKb(m.info.size_bytes)})…`;
    m.loading = fetchBytes(`models/${b}/model.onnx`).then((bytes) => createSession(rt, bytes));
  }
  m.loaded = await m.loading;
  return m.loaded;
}

// ---------------------------------------------------------------- inference

async function infer(): Promise<Timing> {
  const m = models.get(bands)!;
  const loaded = await ensureModel(bands);
  const { h, w, cube } = frame;
  const t0 = performance.now();
  const x = m.transform.apply(cube, h, w);
  const t1 = performance.now();
  const out = await loaded.session.run({ input: new rt.ort.Tensor("float32", x, [1, m.transform.nBands, h, w]) });
  const t2 = performance.now();
  const logits = out.logits.data as Float32Array;
  pred = decode(logits, out.logits.dims[1], h * w);
  drawFrames();
  const t3 = performance.now();
  return { pre: t1 - t0, infer: t2 - t1, post: t3 - t2, total: t3 - t0 };
}

async function runOnce() {
  if (busy) return;
  busy = true;
  try {
    last = await infer();
    bench = null;
    if (!selected) selected = defaultPixel();
    updatePanels();
  } catch (e) {
    warn(`Inference failed: ${(e as Error).message}`);
    console.error(e);
  } finally {
    busy = false;
  }
}

function defaultPixel(): [number, number] {
  const { labels, h, w } = frame;
  if (labels) {
    const idx: number[] = [];
    labels.forEach((v, i) => v !== 255 && idx.push(i));
    if (idx.length) {
      const p = idx[Math.floor(idx.length / 2)];
      return [Math.floor(p / w), p % w];
    }
  }
  return [Math.floor(h / 2), Math.floor(w / 2)];
}

// ---------------------------------------------------------------- drawing

function drawFrames() {
  if (!pred) return;
  const { rgb, w, h, labels } = frame;
  const ov = overlay(rgb, pred, colours, Number(ui.alpha.value), Number(ui.minconf.value), labels, w);
  paint(ui.rgb, rgb, w, h, selected ?? undefined);
  paint(ui.over, ov, w, h, selected ?? undefined);
}

function updatePanels() {
  if (!pred || !last) return;
  const m = models.get(bands)!;
  const loaded = m.loaded!;
  const { h, w, labels } = frame;
  const hw = h * w;
  const k = index.class_names.length;

  // title + finding
  ui.title.textContent = frame.title;
  const parts: string[] = [];
  if (labels) {
    const acc = annotatedAccuracy(pred.pred, labels);
    parts.push(`${(acc.accuracy * 100).toFixed(1)}% of ${acc.n.toLocaleString("en")} annotated pixels classified as ${pretty(frame.organ!)}`);
  } else parts.push("Your cube, no annotation to score against");
  parts.push(`${m.info.n_bands}-band model`);
  parts.push(`${fmtMs(last.total)} ms per frame on ${loaded.backend === "webgpu" ? "WebGPU" : "WebAssembly"}`);
  ui.finding.replaceChildren(...parts.flatMap((p, i) => (i ? [el("span", { class: "sep" }, "·"), document.createTextNode(p)] : [document.createTextNode(p)])));

  // legend: classes present, by share of frame
  const counts = new Array<number>(k).fill(0);
  pred.pred.forEach((c) => counts[c]++);
  const all = counts.map((n, c) => ({ c, n })).filter((r) => r.n > 0).sort((a, b) => b.n - a.n);
  const isGt = (c: number) => frame.organ !== null && index.class_names[c] === frame.organ;
  // top 8 classes (always including the annotated organ); the rest summarised in one row
  const rows = all.filter((r, i) => i < 8 || isGt(r.c));
  const rest = all.filter((r) => !rows.includes(r));
  ui.legend.replaceChildren(
    ...rows.map(({ c, n }) => {
      const tr = el("tr");
      if (isGt(c)) tr.className = "gt";
      const name = el("td");
      name.append(el("span", { class: "swatch", style: `background:${index.class_colours[c]}` }), document.createTextNode(pretty(index.class_names[c])));
      tr.append(name, el("td", { class: "num" }, `${((100 * n) / hw).toFixed(1)}%`));
      return tr;
    }),
    ...(rest.length
      ? [(() => {
          const tr = el("tr", { class: "rest" });
          const restPx = rest.reduce((a, r) => a + r.n, 0);
          tr.append(el("td", {}, `${rest.length} more ${rest.length === 1 ? "class" : "classes"}`),
            el("td", { class: "num" }, `${((100 * restPx) / hw).toFixed(1)}%`));
          return tr;
        })()]
      : []),
  );

  // mean confidence in annotated region
  const mask = labels && labels.some((v) => v !== 255) ? (p: number) => labels[p] !== 255 : undefined;
  bars(ui.confidence, meanConfidence(pred.probs, k, hw, mask).slice(0, 5));
  updatePixel();
  updateRuntime();
  warn(last.total > SLOW_MS ? `This device needs ${fmtMs(last.total)} ms per frame (${(1000 / last.total).toFixed(1)} fps). ` +
    `${rt.backend === "wasm" ? "Without WebGPU, the model runs on the CPU via WebAssembly; a desktop browser with WebGPU is much faster." : ""}` : null);
}

function bars(target: HTMLElement, items: { cls: number; p: number }[]) {
  target.replaceChildren(
    ...items.map((it, i) => {
      const li = el("li");
      if (i === 0) li.className = "top";
      const track = el("span", { class: "track" });
      track.append(el("span", { class: "fill", style: `width:${(it.p * 100).toFixed(1)}%` }));
      li.append(el("span", { class: "name" }, pretty(index.class_names[it.cls])), el("span", { class: "val" }, it.p.toFixed(2)), track);
      return li;
    }),
  );
}

function updatePixel() {
  if (!pred || !selected) return;
  const m = models.get(bands)!;
  const [r, c] = selected;
  const { h, w, labels, cube } = frame;
  const hw = h * w;
  const p = r * w + c;
  const k = index.class_names.length;
  const probs = Array.from({ length: k }, (_, cls) => ({ cls, p: pred!.probs[cls * hw + p] })).sort((a, b) => b.p - a.p);
  bars(ui.pixelProbs, probs.slice(0, 5));

  const truth = labels && labels[p] !== 255 ? pretty(index.class_names[labels[p]]) : null;
  ui.pixelInfo.textContent = `Row ${r}, column ${c}${truth ? ` · annotated as ${truth}` : " · not annotated"}`;

  const spec = index.class_mean_spectra[bands];
  const xs = bands === "full" ? index.wavelengths_nm : spec.band_centres_nm;
  const pixel = m.transform.projectL1(cube, p);
  const scale = (a: ArrayLike<number>) => Array.from(a, (v) => v * 100);
  const top = probs.slice(0, 2);
  const series: Series[] = top.map((t, i) => ({
    x: xs, y: scale(spec.mean[t.cls]), sd: scale(spec.std[t.cls]), colour: index.class_colours[t.cls], dashed: i === 1,
  }));
  series.push({ x: xs, y: scale(pixel), colour: "#ffffff", points: bands !== "full", width: 1.6 });
  drawChart(ui.chart, series, { grid: "#2e2e31", text: "#8a8c92" });
  ui.chartLegend.replaceChildren(
    ...top.map((t, i) => {
      const li = el("li");
      li.append(el("span", { class: `key${i ? " dashed" : ""}`, style: `border-color:${index.class_colours[t.cls]}` }),
        document.createTextNode(`${cap(pretty(index.class_names[t.cls]))} training mean ± sd, p = ${t.p.toFixed(2)}`));
      return li;
    }),
    (() => {
      const li = el("li");
      li.append(el("span", { class: "key", style: "border-color:#ffffff" }), document.createTextNode("This pixel"));
      return li;
    })(),
    el("li", { class: "meta" }, "Wavelength in nm; values are % of the pixel's total (L1-normalised)"),
  );
}

function updateRuntime() {
  if (!last) return;
  const loaded = models.get(bands)!.loaded!;
  const wasm = transferredBytes(".wasm");
  const onnx = transferredBytes(".onnx");
  const rows: [string, string, boolean?][] = [
    [bench ? "Per frame (benchmark median)" : "Last frame", `${fmtMs(last.total)} ms`, true],
    ["Preprocess / model / draw", `${fmtMs(last.pre)} / ${fmtMs(last.infer)} / ${fmtMs(last.post)} ms`],
    ["Frame size", `${frame.w}×${frame.h} px, ${NIN} → ${models.get(bands)!.transform.nBands} bands`],
  ];
  if (liveTimes.length > 5) rows.push(["Live throughput", `${(1000 / median(liveTimes)).toFixed(0)} fps (median of ${liveTimes.length})`]);
  if (bench) rows.push([`${bench.n}-frame benchmark`, `median ${fmtMs(bench.median)} ms, p90 ${fmtMs(bench.p90)} ms → ${(1000 / bench.median).toFixed(0)} fps`]);
  rows.push(
    ["Backend", `${loaded.backend === "webgpu" ? "WebGPU" : "WebAssembly (CPU)"}${loaded.fellBack ? ", WebGPU failed for this model" : ""}`],
    ["CPU threads", `${rt.threads}${rt.isolated ? "" : " (page not cross-origin isolated)"}`],
    ["Downloaded runtime", wasm.decoded ? `${fmtKb(wasm.decoded)}${wasm.transfer ? `, ${fmtKb(wasm.transfer)} over the network` : ", from cache"}` : "-"],
    ["Downloaded models", onnx.decoded ? `${fmtKb(onnx.decoded)}${onnx.transfer ? "" : ", from cache"}` : "-"],
  );
  ui.runtime.replaceChildren(...rows.flatMap(([k, v, big]) => [el("dt", {}, k), el("dd", big ? { class: "big" } : {}, v)]));
}

// ---------------------------------------------------------------- controls

function tab(label: string, sub: string | null, checked: boolean, onClick: () => void) {
  const b = el("button", { class: "tab", role: "radio", type: "button", "aria-checked": String(checked) }, label);
  if (sub) b.append(el("span", { class: "sub" }, sub));
  b.addEventListener("click", onClick);
  return b;
}

function setChecked(group: HTMLElement, i: number) {
  [...group.children].forEach((c, j) => c.setAttribute("aria-checked", String(i === j)));
}

function buildControls() {
  index.samples.forEach((s, i) => {
    ui.samples.append(tab(cap(pretty(s.organ)), s.subject, i === 0, async () => {
      setChecked(ui.samples, i);
      ui.upload.value = "";
      frame = await loadSample(s.id);
      selected = null;
      await runOnce();
    }));
  });
  BANDS.forEach((b, i) => {
    const info = models.get(b)!.info;
    ui.models.append(tab(b === "full" ? "Full spectrum" : "16-band sensor", `${info.n_bands} bands · ${fmtKb(info.size_bytes)}`, b === bands, async () => {
      setChecked(ui.models, i);
      bands = b;
      bench = null;
      liveTimes.length = 0;
      await runOnce();
    }));
  });
  ui.alpha.addEventListener("input", drawFrames);
  ui.minconf.addEventListener("input", () => {
    ui.confOut.textContent = Number(ui.minconf.value).toFixed(2);
    drawFrames();
  });
  for (const cv of [ui.rgb, ui.over])
    cv.addEventListener("click", (ev) => {
      selected = pixelAt(cv, ev);
      drawFrames();
      updatePixel();
    });
  ui.upload.addEventListener("change", onUpload);
  ui.bench.addEventListener("click", runBenchmark);
  ui.live.addEventListener("click", toggleLive);
  window.addEventListener("resize", () => updatePixel());
}

async function onUpload() {
  const file = ui.upload.files?.[0];
  if (!file) return;
  try {
    const { data, shape } = parseNpy(await file.arrayBuffer());
    if (shape.length !== 3 || shape[2] !== NIN) throw new Error(`expected shape (H, W, 100), got (${shape.join(", ")})`);
    if (Math.max(shape[0], shape[1]) > 512) throw new Error("at most 512 pixels per side");
    if (!data.every(Number.isFinite)) throw new Error("the cube contains NaN or infinite values");
    const [h, w] = shape;
    const cube = l1NormaliseCube(data, NIN);
    const recipe = percentileRecipe(cube, h * w, index.wavelengths_nm, NIN);
    frame = { id: "upload", title: file.name, cube, h, w, labels: null, organ: null, recipe, rgb: pseudoRgb(cube, h * w, recipe, NIN) };
    setChecked(ui.samples, -1);
    selected = null;
    warn(null);
    await runOnce();
  } catch (e) {
    warn(`Could not load ${file.name}: ${(e as Error).message}.`);
  }
}

async function runBenchmark() {
  if (liveOn) toggleLive();
  ui.bench.disabled = true;
  const label = ui.bench.textContent;
  try {
    for (let i = 0; i < 5; i++) await infer(); // warm-up, excluded
    const runs: Timing[] = [];
    for (let i = 0; i < 50; i++) {
      runs.push(await infer());
      if (i % 10 === 9) ui.bench.textContent = `${i + 1} / 50`;
    }
    const totals = runs.map((r) => r.total);
    bench = { median: median(totals), p90: pct(totals, 0.9), n: runs.length, backend: models.get(bands)!.loaded!.backend };
    // the readout shows the benchmark's own per-stage medians, not the last single frame
    last = {
      pre: median(runs.map((r) => r.pre)),
      infer: median(runs.map((r) => r.infer)),
      post: median(runs.map((r) => r.post)),
      total: bench.median,
    };
    updatePanels();
  } finally {
    ui.bench.textContent = label;
    ui.bench.disabled = false;
  }
}

function toggleLive() {
  liveOn = !liveOn;
  ui.live.setAttribute("aria-pressed", String(liveOn));
  ui.live.textContent = liveOn ? "Live: on" : "Live: off";
  if (liveOn) {
    liveTimes.length = 0;
    const loop = async () => {
      if (!liveOn) return;
      try {
        last = await infer();
        liveTimes.push(last.total);
        if (liveTimes.length > 60) liveTimes.shift();
        updateRuntime();
        if (liveTimes.length % 10 === 0) updatePanels();
      } catch (e) {
        console.error(e);
        liveOn = false;
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  } else updatePanels();
}

// ---------------------------------------------------------------- boot

async function boot() {
  try {
    index = await fetchJson<Index>("samples/index.json");
    colours = index.class_colours.map(hexToRgb);
    const [infos, specs] = await Promise.all([
      Promise.all(BANDS.map((b) => fetchJson<ModelInfo>(`models/${b}/model.json`))),
      Promise.all(BANDS.map((b) => fetchJson<TransformSpec>(`models/${b}/transform.json`))),
    ]);
    BANDS.forEach((b, i) => models.set(b, { info: infos[i], transform: new SpectralTransform(specs[i]) }));
    buildControls();
    ui.finding.textContent = "Loading ONNX Runtime Web…";
    const forced = new URLSearchParams(location.search).get("backend");
    const force = forced === "wasm" || forced === "webgpu" ? forced : undefined;
    [rt, frame] = await Promise.all([loadRuntime(force), loadSample(index.samples[0].id)]);
    await runOnce();
    ui.bench.disabled = false;
    ui.live.disabled = false;
    (window as unknown as { __hsi: unknown }).__hsi = { ready: true, backend: rt.backend, threads: rt.threads, last };
  } catch (e) {
    ui.title.textContent = "The demo could not start";
    warn(`${(e as Error).message}. Try a current desktop browser (Chrome, Edge, Firefox or Safari).`);
    (window as unknown as { __hsi: unknown }).__hsi = { ready: false, error: String(e) };
    console.error(e);
  }
}

boot();
