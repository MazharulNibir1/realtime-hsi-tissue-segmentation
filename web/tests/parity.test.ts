/**
 * Parity: the TypeScript preprocessing + onnxruntime-web (WASM backend, in Node) must reproduce the
 * Python pipeline (hsi.data.normalize + ONNX Runtime CPU). Reference written by
 * scripts/make_web_reference.py from the same files the website serves (public/models, public/samples).
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-web";
import { beforeAll, describe, expect, it } from "vitest";
import { SpectralTransform, annotatedAccuracy, decode, decodeFloat16, parseNpy, type SampleMeta } from "../src/core";

const here = dirname(fileURLToPath(import.meta.url));
const pub = join(here, "../public");
const ref = JSON.parse(readFileSync(join(here, "fixtures/reference.json"), "utf8"));
const readJson = (p: string) => JSON.parse(readFileSync(join(pub, p), "utf8"));
const bytes = (p: string) => new Uint8Array(readFileSync(join(pub, p)));

const INPUT_TOL = 1e-5; // transformed input (standardised units)
const LOGIT_TOL = 1e-3; // logits after 3-4 layers; float32 accumulation order differs

beforeAll(() => {
  ort.env.wasm.numThreads = 1;
});

for (const [sid, entry] of Object.entries<any>(ref.samples)) {
  describe(`sample ${sid}`, () => {
    const meta: SampleMeta = readJson(`samples/${sid}.json`);
    const [h, w] = meta.shape;
    const cube = decodeFloat16(bytes(`samples/${meta.cube_file}`));
    const labels = bytes(`samples/${meta.labels_file}`);

    for (const [bands, m] of Object.entries<any>(entry.models)) {
      it(`${bands}: preprocessing, logits and argmax match Python`, async () => {
        const tr = new SpectralTransform(readJson(`models/${bands}/transform.json`));
        const x = tr.apply(cube, h, w);
        const hw = h * w;
        let maxInput = 0;
        entry.pixels.forEach((p: number, i: number) => {
          for (let b = 0; b < tr.nBands; b++) maxInput = Math.max(maxInput, Math.abs(x[b * hw + p] - m.input_at_pixels[i][b]));
        });
        expect(maxInput).toBeLessThan(INPUT_TOL);

        const session = await ort.InferenceSession.create(readFileSync(join(pub, `models/${bands}/model.onnx`)), {
          executionProviders: ["wasm"],
        });
        const out = await session.run({ input: new ort.Tensor("float32", x, [1, tr.nBands, h, w]) });
        const logits = out.logits.data as Float32Array;
        const k = out.logits.dims[1];
        let maxLogit = 0;
        entry.pixels.forEach((p: number, i: number) => {
          for (let c = 0; c < k; c++) maxLogit = Math.max(maxLogit, Math.abs(logits[c * hw + p] - m.logits_at_pixels[i][c]));
        });
        expect(maxLogit).toBeLessThan(LOGIT_TOL);

        const { pred } = decode(logits, k, hw);
        const refPred = Buffer.from(m.argmax_b64, "base64");
        let same = 0;
        for (let i = 0; i < hw; i++) if (pred[i] === refPred[i]) same++;
        expect(same / hw).toBeGreaterThanOrEqual(0.999);
        console.info(
          `${sid}/${bands}: max|input diff|=${maxInput.toExponential(2)} max|logit diff|=${maxLogit.toExponential(2)} ` +
            `argmax agreement=${((100 * same) / hw).toFixed(3)}%`,
        );
        expect(annotatedAccuracy(pred, labels).accuracy).toBeCloseTo(m.annotated_accuracy, 3);
        expect(annotatedAccuracy(pred, labels).accuracy).toBeCloseTo(meta.demo_model_pixel_accuracy[bands], 3);
      }, 60_000);
    }
  });
}

describe("npy parser", () => {
  it("reads a float32 C-order array", () => {
    const header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3, 4), }";
    const pad = 64 - ((10 + header.length + 1) % 64);
    const h = header + " ".repeat(pad) + "\n";
    const buf = new Uint8Array(10 + h.length + 24 * 4);
    buf.set([0x93, ...new TextEncoder().encode("NUMPY"), 1, 0, h.length & 0xff, h.length >> 8]);
    buf.set(new TextEncoder().encode(h), 10);
    const vals = Float32Array.from({ length: 24 }, (_, i) => i / 10);
    buf.set(new Uint8Array(vals.buffer), 10 + h.length);
    const { data, shape } = parseNpy(buf.buffer);
    expect(shape).toEqual([2, 3, 4]);
    expect(Array.from(data)).toEqual(Array.from(vals));
  });
});
