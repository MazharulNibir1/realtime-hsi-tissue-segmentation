# Deploying the browser demo to Vercel

The public demo is the static site in `web/` (Vite + TypeScript + ONNX Runtime Web). Vercel builds it
from GitHub and serves it; the model runs in each visitor's browser. You run every step; nothing in
this repo deploys by itself.

**Time:** about 15 minutes. **Cost:** Vercel **Hobby** plan (free, for personal and non-commercial
projects; see <https://vercel.com/pricing>).

---

## 0. Before you start

Make sure the repo on GitHub has everything (from the repo root):

```bash
git status                     # clean
git lfs ls-files | grep web/   # 2 model.onnx + 10 samples/*.bin under web/public/
make web-install web-test web-build   # parity test passes, build succeeds locally
```

## 1. Create the Vercel account

1. Go to <https://vercel.com/signup>.
2. Choose **Hobby** ("I'm working on personal projects").
3. **Continue with GitHub** and authorise Vercel.
4. When Vercel asks which repositories it may access, choose **Only select repositories** and pick
   `realtime-hsi-tissue-segmentation`.

## 2. Import the repository

1. Dashboard → **Add New… → Project** → find `realtime-hsi-tissue-segmentation` → **Import**.
2. On the *Configure Project* screen:

   | Field | Value |
   |---|---|
   | Project Name | `spectral-tissue` (becomes `spectral-tissue.vercel.app`; must be unique on Vercel) |
   | Framework Preset | **Vite** (detected) |
   | **Root Directory** | **`web`** (click *Edit* and select the `web` folder) |
   | Build Command | leave default; `web/vercel.json` sets `npm run build` |
   | Output Directory | leave default; `web/vercel.json` sets `dist` |
   | Install Command | leave default; `web/vercel.json` sets `npm ci` |
   | Environment Variables | none |

3. Click **Deploy**.

**The first build is expected to fail** with this message from the build guard:

```
check-assets: 12 file(s) are Git LFS pointers, not data: ...
```

That is correct. Vercel clones without Git LFS until you turn it on (next step). The guard
(`web/scripts/check-assets.mjs`) stops the build, so a site with broken models is never published.

## 3. Turn on Git LFS and redeploy

The ONNX models and sample cubes are stored with Git LFS.

1. Project → **Settings → Git** → **Git Large File Storage (LFS)** → turn **on** → **Save**.
2. **Deployments** → the failed deployment → **⋯ → Redeploy**. Leave *Use existing Build Cache*
   unticked.
3. The build log should now show:

```
check-assets: 12 model/sample files present (no LFS pointers)
✓ built in ...
```

Vercel does not charge for LFS, but every build downloads the LFS files (about 20 MB) from GitHub,
which counts against your GitHub LFS bandwidth quota
(see GitHub *Settings → Billing → Git LFS*). Normal pushes do not rebuild that often, so this is
rarely a problem.

**If the build fails at "Cloning" after enabling LFS** (reported by some users):

- redeploy once more without the build cache;
- check that the LFS objects exist on GitHub: open a `.onnx` file on github.com; it should say
  *Stored with Git LFS* and download a ~150 kB file;
- if it still fails, turn LFS off and use the fallback in the troubleshooting table below.

## 4. Choose the subdomain

- Your site is live at `https://<project-name>.vercel.app`.
- To change it: **Settings → General → Project Name** (the default `.vercel.app` domain follows the
  name), or **Settings → Domains → Add** to add another `something.vercel.app` alias.
- A custom domain you own (e.g. `demo.yourname.dev`): **Settings → Domains → Add**, then create the
  DNS record Vercel shows (usually a `CNAME` to `cname.vercel-dns.com`).

## 5. Verify the deployment

### Automated checks (from your Mac)

```bash
cd web
npm run build                                          # creates dist/assets with the same hashed names
scripts/verify-deploy.sh https://<project-name>.vercel.app
```

Expected: `HTTP 200`, `COOP same-origin`, `COEP require-corp`, both `.wasm` files with
`Content-Type application/wasm`, **compressed (br or gzip)** and `immutable` caching, and the models
and samples reported with their real sizes (158 kB / 115 kB / 3.84 MB). It ends with *All checks
passed.*

### In the browser

1. Open the site in desktop Chrome or Edge. Within a few seconds the title shows *Liver, pig P095*
   and the finding line reports the annotated-pixel accuracy and the ms per frame.
2. **Runtime on this device** panel:
   - *Backend: WebGPU* on a browser with WebGPU (current Chrome or Edge on most laptops).
   - *CPU threads: 4* (or your core count, up to 4). If it says *(page not cross-origin isolated)*,
     the COOP/COEP headers are missing; see troubleshooting.
   - *Downloaded runtime*: about 29.5 MB decoded, and a much smaller "over the network" figure if
     compression works.
3. Click **Run 50 frames**: the median and fps appear.
4. **WASM fallback:** open `https://<project-name>.vercel.app/?backend=wasm`. The panel should say
   *Backend: WebAssembly (CPU)* with the same accuracy in the finding line. Safari or Firefox
   without WebGPU take this path automatically.
5. **Phone:** open the URL on your phone. The layout stacks into one column; if the device needs more
   than 250 ms per frame, a yellow note says so.
6. **DevTools console** (F12): no red errors; `crossOriginIsolated` evaluates to `true`.

## 6. Updating

- Every push to `main` triggers a new production deployment automatically.
- Pushes to other branches and pull requests get **preview deployments** with their own URL. On
  Hobby, previews may require you to be logged in to Vercel (*Settings → Deployment Protection*);
  the production URL is always public.
- After retraining: `make export demo-models samples web-reference web-test` in the repo, commit,
  push. Vercel redeploys.
- To roll back: **Deployments** → an older deployment → **⋯ → Promote to Production**.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build fails: `check-assets: ... Git LFS pointers` | LFS off in Vercel | Step 3 |
| Build fails: `Cannot find module ...` / `npm ci` error | `web/package-lock.json` missing or out of date | run `npm install` in `web/`, commit `package-lock.json` |
| Build fails: `tsc` errors | TypeScript error in `web/src` | `make web-build` locally shows the same error |
| 404 on the site root | Root Directory not set to `web` | Settings → General → Root Directory → `web`, redeploy |
| Runtime panel: *page not cross-origin isolated*, 1 thread | `vercel.json` headers not applied | the file must be `web/vercel.json` (inside the Root Directory); check with `verify-deploy.sh` |
| *The demo could not start*, console: `Failed to fetch ... .wasm` | wasm blocked or wrong type | `verify-deploy.sh` shows the wasm `Content-Type`; it must be `application/wasm` |
| Model fails to load (`protobuf` / `invalid model`) | an LFS pointer was deployed | the build guard prevents this; if you removed it, enable LFS and redeploy |
| Very slow on a phone | no WebGPU, CPU-only WASM | expected; the page shows its own fps and a note |
| LFS clone keeps failing on Vercel | Vercel/GitHub LFS issue | fallback: in a separate branch, `git lfs migrate export --include="web/public/**"` to store those files as normal git objects (about 20 MB), then deploy that branch |

Sources: [Vercel docs: Git settings (LFS)](https://vercel.com/docs/project-configuration/git-settings),
[Vercel changelog: Git LFS support](https://vercel.com/changelog/git-lfs-support),
[Vercel docs: compression](https://vercel.com/docs/compression).
