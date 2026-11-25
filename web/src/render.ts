/** Canvas rendering: pseudo-RGB, class overlay, crosshair, and the spectrum chart. */
import type { Prediction } from "./core";

export interface RgbRecipe {
  bands: number[];
  lo: number[];
  hi: number[];
  gamma: number;
}

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

/** Recipe for user cubes: per-channel 1st-99th percentile on 650/570/510 nm, gamma 1.2 (as in Python). */
export function percentileRecipe(cube: Float32Array, hw: number, wavelengths: number[], nIn = 100): RgbRecipe {
  const bands = [650, 570, 510].map((t) => {
    let best = 0;
    wavelengths.forEach((w, i) => {
      if (Math.abs(w - t) < Math.abs(wavelengths[best] - t)) best = i;
    });
    return best;
  });
  const q = (arr: Float32Array, f: number) => {
    const s = Float32Array.from(arr).sort();
    const pos = (s.length - 1) * f;
    const lo = Math.floor(pos);
    return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (pos - lo);
  };
  const lo: number[] = [];
  const hi: number[] = [];
  for (const b of bands) {
    const ch = new Float32Array(hw);
    for (let p = 0; p < hw; p++) ch[p] = cube[p * nIn + b];
    lo.push(q(ch, 0.01));
    hi.push(q(ch, 0.99));
  }
  return { bands, lo, hi, gamma: 1.2 };
}

export function pseudoRgb(cube: Float32Array, hw: number, r: RgbRecipe, nIn = 100): Uint8ClampedArray {
  const out = new Uint8ClampedArray(hw * 4);
  for (let p = 0; p < hw; p++) {
    for (let k = 0; k < 3; k++) {
      const v = (cube[p * nIn + r.bands[k]] - r.lo[k]) / Math.max(r.hi[k] - r.lo[k], 1e-12);
      out[p * 4 + k] = Math.round(Math.min(1, Math.max(0, v)) ** (1 / r.gamma) * 255);
    }
    out[p * 4 + 3] = 255;
  }
  return out;
}

export function overlay(
  rgb: Uint8ClampedArray,
  pred: Prediction,
  colours: [number, number, number][],
  alpha: number,
  minConf: number,
  labels: Uint8Array | null,
  w: number,
): Uint8ClampedArray {
  const hw = pred.pred.length;
  const out = new Uint8ClampedArray(rgb);
  for (let p = 0; p < hw; p++) {
    if (pred.conf[p] < minConf) continue;
    const c = colours[pred.pred[p]];
    for (let k = 0; k < 3; k++) out[p * 4 + k] = (1 - alpha) * rgb[p * 4 + k] + alpha * c[k];
  }
  if (labels) {
    // outline the annotated region (4-neighbourhood boundary) in white
    const h = hw / w;
    const inside = (r: number, c: number) => r >= 0 && c >= 0 && r < h && c < w && labels[r * w + c] !== 255;
    for (let r = 0; r < h; r++)
      for (let c = 0; c < w; c++) {
        if (!inside(r, c)) continue;
        if (inside(r - 1, c) && inside(r + 1, c) && inside(r, c - 1) && inside(r, c + 1)) continue;
        const p = (r * w + c) * 4;
        out[p] = out[p + 1] = out[p + 2] = 255;
      }
  }
  return out;
}

export function paint(canvas: HTMLCanvasElement, rgba: Uint8ClampedArray, w: number, h: number, mark?: [number, number]) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  const img = new ImageData(w, h);
  img.data.set(rgba);
  ctx.putImageData(img, 0, 0);
  if (mark) {
    const [r, c] = mark;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(c - 4, r, 3, 1);
    ctx.fillRect(c + 2, r, 3, 1);
    ctx.fillRect(c, r - 4, 1, 3);
    ctx.fillRect(c, r + 2, 1, 3);
  }
}

/** Canvas pixel (row, col) under a pointer event, for a canvas drawn at native resolution. */
export function pixelAt(canvas: HTMLCanvasElement, ev: MouseEvent): [number, number] {
  const rect = canvas.getBoundingClientRect();
  const c = Math.floor(((ev.clientX - rect.left) / rect.width) * canvas.width);
  const r = Math.floor(((ev.clientY - rect.top) / rect.height) * canvas.height);
  return [Math.min(canvas.height - 1, Math.max(0, r)), Math.min(canvas.width - 1, Math.max(0, c))];
}

// ---------------------------------------------------------------- spectrum chart

export interface Series {
  x: number[];
  y: ArrayLike<number>;
  sd?: number[];
  colour: string;
  dashed?: boolean;
  points?: boolean;
  width?: number;
}

export function drawChart(canvas: HTMLCanvasElement, series: Series[], ink: { grid: string; text: string }) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  const pad = { l: 44, r: 16, t: 10, b: 28 };
  const W = cssW - pad.l - pad.r;
  const H = cssH - pad.t - pad.b;
  let ymin = Infinity;
  let ymax = -Infinity;
  for (const s of series)
    for (let i = 0; i < s.y.length; i++) {
      const sd = s.sd ? s.sd[i] : 0;
      ymin = Math.min(ymin, s.y[i] - sd);
      ymax = Math.max(ymax, s.y[i] + sd);
    }
  ymin = Math.max(0, ymin - (ymax - ymin) * 0.08);
  ymax += (ymax - ymin) * 0.08;
  const X = (v: number) => pad.l + ((v - 500) / 500) * W;
  const Y = (v: number) => pad.t + H - ((v - ymin) / (ymax - ymin)) * H;

  ctx.font = '12px "Public Sans", system-ui, sans-serif';
  ctx.fillStyle = ink.text;
  ctx.strokeStyle = ink.grid;
  ctx.lineWidth = 1;
  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {
    const v = ymin + ((ymax - ymin) * i) / yTicks;
    const y = Math.round(Y(v)) + 0.5;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + W, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(v.toFixed(v < 10 ? 2 : 1), pad.l - 8, y);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let nm = 500; nm <= 1000; nm += 100) ctx.fillText(`${nm}`, X(nm), pad.t + H + 8);

  for (const s of series) {
    if (s.sd) {
      ctx.fillStyle = s.colour;
      ctx.globalAlpha = 0.16;
      ctx.beginPath();
      s.x.forEach((xv, i) => (i ? ctx.lineTo(X(xv), Y(s.y[i] + s.sd![i])) : ctx.moveTo(X(xv), Y(s.y[i] + s.sd![i]))));
      for (let i = s.x.length - 1; i >= 0; i--) ctx.lineTo(X(s.x[i]), Y(s.y[i] - s.sd[i]));
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = s.colour;
    ctx.lineWidth = s.width ?? 2;
    ctx.setLineDash(s.dashed ? [6, 4] : []);
    ctx.beginPath();
    s.x.forEach((xv, i) => (i ? ctx.lineTo(X(xv), Y(s.y[i])) : ctx.moveTo(X(xv), Y(s.y[i]))));
    ctx.stroke();
    ctx.setLineDash([]);
    if (s.points) {
      ctx.fillStyle = s.colour;
      s.x.forEach((xv, i) => {
        ctx.beginPath();
        ctx.arc(X(xv), Y(s.y[i]), 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }
}
