// Build guard: fail if models or samples are Git LFS pointer files instead of real data.
// Happens when a host clones the repo without Git LFS (on Vercel: Settings -> Git -> Git LFS).
import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const pub = join(dirname(fileURLToPath(import.meta.url)), "../public");
const files = [];
const walk = (d) => {
  for (const f of readdirSync(d)) {
    const p = join(d, f);
    if (statSync(p).isDirectory()) walk(p);
    else if (/\.(onnx|bin)$/.test(f)) files.push(p);
  }
};
walk(pub);

const pointers = files.filter((p) => readFileSync(p).subarray(0, 64).toString("latin1").startsWith("version https://git-lfs"));
if (files.length === 0 || pointers.length > 0) {
  console.error(
    files.length === 0
      ? "check-assets: no .onnx/.bin files found under public/"
      : `check-assets: ${pointers.length} file(s) are Git LFS pointers, not data:\n  ` +
          pointers.map((p) => relative(pub, p)).join("\n  ") +
          "\nEnable Git LFS for this checkout (locally: git lfs install && git lfs pull; " +
          "on Vercel: Project Settings -> Git -> Git Large File Storage, then redeploy).",
  );
  process.exit(1);
}
console.log(`check-assets: ${files.length} model/sample files present (no LFS pointers)`);
