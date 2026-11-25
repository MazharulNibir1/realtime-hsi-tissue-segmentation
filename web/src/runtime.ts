/**
 * ONNX Runtime Web loading. WebGPU when the browser offers an adapter (full `onnxruntime-web`
 * build), otherwise the smaller WASM-only build. Multithreaded WASM needs cross-origin isolation
 * (COOP/COEP headers, see vercel.json and vite.config.ts); without it, one thread is used.
 */
import type * as OrtNS from "onnxruntime-web";

type Ort = typeof OrtNS;
export type Backend = "webgpu" | "wasm";

export interface Runtime {
  ort: Ort;
  backend: Backend;
  threads: number;
  isolated: boolean;
  note: string;
}

let runtimePromise: Promise<Runtime> | null = null;

async function hasWebGpu(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    return (await gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

function configure(ort: Ort): { threads: number; isolated: boolean } {
  const isolated = globalThis.crossOriginIsolated === true;
  const threads = isolated ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1)) : 1;
  // The .wasm binaries are resolved by Vite from onnxruntime-web's `new URL(..., import.meta.url)`
  // references and emitted as content-hashed assets, so no wasmPaths override is needed.
  ort.env.wasm.numThreads = threads;
  ort.env.logLevel = "error";
  return { threads, isolated };
}

export function loadRuntime(force?: Backend): Promise<Runtime> {
  if (runtimePromise && !force) return runtimePromise;
  runtimePromise = (async () => {
    const gpu = force ? force === "webgpu" && (await hasWebGpu()) : await hasWebGpu();
    const ort: Ort = gpu ? await import("onnxruntime-web") : await import("onnxruntime-web/wasm");
    const { threads, isolated } = configure(ort);
    const note = gpu
      ? "WebGPU adapter found"
      : force === "wasm"
        ? "WebAssembly requested with ?backend=wasm"
        : "No WebGPU adapter in this browser, using WebAssembly on the CPU";
    return { ort, backend: gpu ? "webgpu" : "wasm", threads, isolated, note };
  })();
  return runtimePromise;
}

export interface LoadedModel {
  session: OrtNS.InferenceSession;
  backend: Backend;
  fellBack: boolean;
}

/** Create a session; if WebGPU fails for this model, fall back to WASM and say so. */
export async function createSession(rt: Runtime, modelBytes: Uint8Array): Promise<LoadedModel> {
  const opts = { graphOptimizationLevel: "all" as const };
  if (rt.backend === "webgpu") {
    try {
      const session = await rt.ort.InferenceSession.create(modelBytes, { ...opts, executionProviders: ["webgpu"] });
      return { session, backend: "webgpu", fellBack: false };
    } catch (e) {
      console.warn("WebGPU session failed, falling back to WASM", e);
    }
  }
  const session = await rt.ort.InferenceSession.create(modelBytes, { ...opts, executionProviders: ["wasm"] });
  return { session, backend: "wasm", fellBack: rt.backend === "webgpu" };
}

/** Bytes actually transferred for resources whose URL contains `needle` (0 if served from cache). */
export function transferredBytes(needle: string): { transfer: number; decoded: number } {
  let transfer = 0;
  let decoded = 0;
  for (const e of performance.getEntriesByType("resource") as PerformanceResourceTiming[]) {
    if (!e.name.includes(needle)) continue;
    transfer += e.transferSize;
    decoded += e.decodedBodySize;
  }
  return { transfer, decoded };
}
