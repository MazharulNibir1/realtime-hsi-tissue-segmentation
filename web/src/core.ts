/**
 * Pure, DOM-free core shared by the browser app and the Node parity test.
 *
 * Input transform: an exact port of hsi.data.normalize.SpectralTransform (Python), applied to an
 * L1-normalised 100-band cube stored as H x W x 100 (HWC):
 *
 *   y_b  = sum_c W[b][c] * x_c              band projection (identity for the full spectrum,
 *                                            16 x 100 box filter for the simulated sensor)
 *   y_b  = y_b / max(sum_b |y_b|, l1_eps)    per-pixel L1 re-normalisation
 *   l1_standardize: y_b = (y_b - mean_b) / std_b      (training-set statistics)
 *   l1_snv:         y_b = (y_b - mean(y)) / max(std(y), snv_eps)
 *
 * Output layout is CHW (N = 1), the ONNX model's input. All arithmetic in float64, stored as
 * float32; Python computes in float32, so values agree to ~1e-6 (checked in tests/parity.test.ts).
 */

export interface TransformSpec {
  mode: "l1" | "l1_standardize" | "l1_snv";
  n_bands: number;
  band_centres_nm: number[];
  W: number[][];
  mean: number[] | null;
  std: number[] | null;
  l1_eps: number;
  snv_eps: number;
}

export interface SampleMeta {
  id: string;
  organ: string;
  subject: string;
  source_image: string;
  shape: [number, number, number];
  cube_file: string;
  labels_file: string;
  ignore_index: number;
  label_idx: number;
  wavelengths_nm: number[];
  annotated_pixels: number;
  demo_model_pixel_accuracy: Record<string, number>;
  pseudo_rgb: { bands: number[]; lo: number[]; hi: number[]; gamma: number };
}

export interface ModelInfo {
  bands: string;
  label: string;
  model: string;
  variant: string;
  n_bands: number;
  params: number;
  size_bytes: number;
  test_macro_f1_3seeds: { mean: number; std: number };
}

// ---------------------------------------------------------------- float16

const HALF_LUT = (() => {
  const lut = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    if (e === 0) lut[h] = s * 2 ** -14 * (f / 1024);
    else if (e === 31) lut[h] = f ? NaN : s * Infinity;
    else lut[h] = s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return lut;
})();

/** Decode little-endian IEEE-754 binary16 bytes to float32. */
export function decodeFloat16(buf: ArrayBuffer | Uint8Array): Float32Array {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const n = bytes.length >> 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = HALF_LUT[bytes[2 * i] | (bytes[2 * i + 1] << 8)];
  return out;
}

// ---------------------------------------------------------------- transform

export class SpectralTransform {
  readonly nBands: number;
  readonly nIn: number;
  private readonly identity: boolean;
  // Sparse rows of W: for band b, input channels rowIdx[b] with weights rowW[b]. The 16-band box
  // filter has 6-7 non-zero weights per row, so projection costs ~100 multiply-adds per pixel.
  private readonly rowIdx: Int32Array[];
  private readonly rowW: Float64Array[];

  constructor(readonly spec: TransformSpec) {
    this.nBands = spec.W.length;
    this.nIn = spec.W[0].length;
    let identity = this.nBands === this.nIn;
    this.rowIdx = [];
    this.rowW = [];
    for (let b = 0; b < this.nBands; b++) {
      const idx: number[] = [];
      const wts: number[] = [];
      for (let c = 0; c < this.nIn; c++) {
        const v = spec.W[b][c];
        if (v !== 0) {
          idx.push(c);
          wts.push(v);
        }
        if (identity && v !== (b === c ? 1 : 0)) identity = false;
      }
      this.rowIdx.push(Int32Array.from(idx));
      this.rowW.push(Float64Array.from(wts));
    }
    this.identity = identity;
    if (spec.mode === "l1_standardize" && (!spec.mean || !spec.std))
      throw new Error("l1_standardize transform needs mean and std");
  }

  /** Band projection + L1 of one pixel (the spectrum the chart compares to class means). */
  projectL1(cube: Float32Array, pixel: number, out = new Float64Array(this.nBands)): Float64Array {
    const base = pixel * this.nIn;
    let l1 = 0;
    for (let b = 0; b < this.nBands; b++) {
      let y = 0;
      if (this.identity) y = cube[base + b];
      else {
        const idx = this.rowIdx[b];
        const wts = this.rowW[b];
        for (let j = 0; j < idx.length; j++) y += wts[j] * cube[base + idx[j]];
      }
      out[b] = y;
      l1 += Math.abs(y);
    }
    const d = Math.max(l1, this.spec.l1_eps);
    for (let b = 0; b < this.nBands; b++) out[b] /= d;
    return out;
  }

  /** HWC cube (H*W*100) -> CHW model input (n_bands*H*W). */
  apply(cube: Float32Array, h: number, w: number, out?: Float32Array): Float32Array {
    const hw = h * w;
    if (cube.length !== hw * this.nIn) throw new Error(`cube has ${cube.length} values, expected ${hw * this.nIn}`);
    const dst = out ?? new Float32Array(this.nBands * hw);
    const y = new Float64Array(this.nBands);
    const { mode, mean, std, snv_eps } = this.spec;
    for (let p = 0; p < hw; p++) {
      this.projectL1(cube, p, y);
      if (mode === "l1_standardize") {
        for (let b = 0; b < this.nBands; b++) y[b] = (y[b] - mean![b]) / std![b];
      } else if (mode === "l1_snv") {
        let mu = 0;
        for (let b = 0; b < this.nBands; b++) mu += y[b];
        mu /= this.nBands;
        let v = 0;
        for (let b = 0; b < this.nBands; b++) v += (y[b] - mu) ** 2;
        const sd = Math.max(Math.sqrt(v / this.nBands), snv_eps);
        for (let b = 0; b < this.nBands; b++) y[b] = (y[b] - mu) / sd;
      }
      for (let b = 0; b < this.nBands; b++) dst[b * hw + p] = y[b];
    }
    return dst;
  }
}

/** Per-pixel L1 normalisation of a user cube (reflectance or already-L1 input). */
export function l1NormaliseCube(cube: Float32Array, nBands = 100): Float32Array {
  const out = new Float32Array(cube.length);
  for (let p = 0; p < cube.length; p += nBands) {
    let s = 0;
    for (let c = 0; c < nBands; c++) s += Math.abs(cube[p + c]);
    const d = Math.max(s, 1e-8);
    for (let c = 0; c < nBands; c++) out[p + c] = cube[p + c] / d;
  }
  return out;
}

// ---------------------------------------------------------------- outputs

export interface Prediction {
  pred: Uint8Array; // argmax class per pixel
  conf: Float32Array; // max softmax probability per pixel
  probs: Float32Array; // softmax, CHW (K * H * W)
}

/** Softmax over classes of CHW logits. */
export function decode(logits: Float32Array, k: number, hw: number): Prediction {
  const pred = new Uint8Array(hw);
  const conf = new Float32Array(hw);
  const probs = new Float32Array(k * hw);
  for (let p = 0; p < hw; p++) {
    let mx = -Infinity;
    let arg = 0;
    for (let c = 0; c < k; c++) {
      const v = logits[c * hw + p];
      if (v > mx) {
        mx = v;
        arg = c;
      }
    }
    let s = 0;
    for (let c = 0; c < k; c++) {
      const e = Math.exp(logits[c * hw + p] - mx);
      probs[c * hw + p] = e;
      s += e;
    }
    for (let c = 0; c < k; c++) probs[c * hw + p] /= s;
    pred[p] = arg;
    conf[p] = 1 / s;
  }
  return { pred, conf, probs };
}

/** Pixel accuracy on annotated pixels (labels != ignore). */
export function annotatedAccuracy(pred: Uint8Array, labels: Uint8Array, ignore = 255) {
  let n = 0;
  let ok = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === ignore) continue;
    n++;
    if (pred[i] === labels[i]) ok++;
  }
  return { n, correct: ok, accuracy: n ? ok / n : NaN };
}

/** Mean class probability over a mask (or the whole frame), top-k. */
export function meanConfidence(probs: Float32Array, k: number, hw: number, mask?: (p: number) => boolean) {
  const sums = new Float64Array(k);
  let n = 0;
  for (let p = 0; p < hw; p++) {
    if (mask && !mask(p)) continue;
    n++;
    for (let c = 0; c < k; c++) sums[c] += probs[c * hw + p];
  }
  return Array.from(sums, (s, c) => ({ cls: c, p: n ? s / n : 0 })).sort((a, b) => b.p - a.p);
}

// ---------------------------------------------------------------- .npy

/** Parse a little-endian C-order .npy holding float32/float16/float64 with shape (H, W, C). */
export function parseNpy(buf: ArrayBuffer): { data: Float32Array; shape: number[] } {
  const u8 = new Uint8Array(buf);
  const magic = String.fromCharCode(...u8.slice(1, 6));
  if (u8[0] !== 0x93 || magic !== "NUMPY") throw new Error("Not a .npy file");
  const major = u8[6];
  const view = new DataView(buf);
  const hlen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const hstart = major === 1 ? 10 : 12;
  const header = new TextDecoder("latin1").decode(u8.slice(hstart, hstart + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1];
  const shapeTxt = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1];
  if (!descr || !fortran || shapeTxt === undefined) throw new Error("Unreadable .npy header");
  if (fortran === "True") throw new Error("Fortran-ordered arrays are not supported; save with C order");
  const shape = shapeTxt.split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const off = hstart + hlen;
  const count = shape.reduce((a, b) => a * b, 1);
  const body = buf.slice(off);
  let data: Float32Array;
  if (descr === "<f4") data = new Float32Array(body, 0, count);
  else if (descr === "<f2") data = decodeFloat16(new Uint8Array(body, 0, count * 2));
  else if (descr === "<f8") data = Float32Array.from(new Float64Array(body, 0, count));
  else throw new Error(`Unsupported dtype ${descr}; use float16, float32 or float64`);
  return { data, shape };
}
