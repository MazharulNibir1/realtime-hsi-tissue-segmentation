# Deploying the Gradio demo to Hugging Face Spaces

This guide publishes `app/` as a free, public Hugging Face Space (Gradio SDK, CPU basic hardware).
You run every command yourself. Nothing in this repo pushes to Hugging Face automatically.

Placeholders used below. Replace them everywhere:

| Placeholder | Example | Meaning |
|---|---|---|
| `<hf-user>` | `your-hf-username` | your Hugging Face username |
| `<space>` | `hsi-tissue-demo` | the Space name (lowercase, hyphens) |
| `~/hf-space` | | where the Space repo is cloned. It must be **outside** this project repo |

Prerequisites: `git`, `git-lfs` (`brew install git-lfs`), and this repo checked out with its LFS files
(`git lfs pull` if `app/models/*/model.onnx` is only a few hundred bytes).

---

## 1. Hugging Face account and access token

1. Sign up at <https://huggingface.co/join> and verify your email address (unverified accounts cannot
   create Spaces).
2. Open **Settings → Access Tokens** (<https://huggingface.co/settings/tokens>) → **Create new token**.
3. Token type **Write**. Name it, for example `spaces-deploy`. Copy the token now; it is shown only once.
   - Optionally choose **Fine-grained** instead and give it write access to just this Space, once the
     Space exists.
4. Keep the token private. Never commit it or paste it into a file in any repo.

When `git push` asks for credentials, the **username is your HF username** and the **password is the
token**, not your account password. On macOS, git stores it in the Keychain after the first push.

## 2. Create the Space

1. Go to <https://huggingface.co/new-space>.
2. Fill in:
   - **Owner:** `<hf-user>`, **Space name:** `<space>`
   - **License:** `mit` (code; the sample data is CC BY 4.0 and attributed in the app)
   - **Select the Space SDK:** **Gradio**, template **Blank**
   - **Space hardware:** **CPU basic** (free)
   - **Visibility:** **Public**
3. Click **Create Space**. The page shows an empty Space with clone instructions.

Hardware and limits change over time; check <https://huggingface.co/pricing> for the current CPU basic
specification and the free-tier sleep policy (free Spaces go to sleep after a period of inactivity and
restart on the next visit).

## 3. Clone the Space and copy the app into it

```bash
git lfs install                       # once per machine
git clone https://huggingface.co/spaces/<hf-user>/<space> ~/hf-space
cd "/path/to/realtime-hsi-tissue-segmentation"   # this project
scripts/prepare_space.sh ~/hf-space
```

`scripts/prepare_space.sh`:

- copies `app.py`, `inference.py`, `requirements.txt`, `README.md`, `models/` and `samples/` into the
  clone, and deletes files there that no longer exist in `app/`;
- keeps the Space's own `.git/` and `.gitattributes`;
- refuses to run if a model or sample is an LFS pointer instead of real data, if the target is not a
  `huggingface.co/spaces` clone, or if the target is inside this repo;
- checks that `sdk_version` in `README.md` equals the `gradio==` pin in `requirements.txt`;
- never commits or pushes. It prints the next commands.

`app/README.md` becomes the Space's README. Its YAML header configures the Space:

```yaml
title: Hyperspectral Tissue Classification
emoji: 🩺
sdk: gradio
sdk_version: 6.32.0      # must equal gradio==6.32.0 in requirements.txt
python_version: "3.11"
app_file: app.py
pinned: false
```

## 4. Track binaries with LFS, commit and push

```bash
cd ~/hf-space
git lfs install
git lfs track "*.onnx" "samples/*.bin"
git add .gitattributes
git add -A
git lfs ls-files        # must list 2 models/*/model.onnx + 10 samples/*.bin
git status              # review before committing
git commit -m "Deploy hyperspectral tissue demo"
git push                # username: <hf-user>, password: your write token
```

The Space's default `.gitattributes` already sends `*.onnx` and `*.bin` to LFS; the explicit
`git lfs track` makes it independent of that default. The upload is about 19 MB.

## 5. Build logs, common failures, updates

**Watch the build:** open `https://huggingface.co/spaces/<hf-user>/<space>`. The status badge goes
*Building → Running*. Click **Logs** (top of the Space page) and switch between **Build** (pip install)
and **Container** (`python app.py` output, Python tracebacks). The first build takes a few minutes.

**Check it works:** pick each sample, switch between the two model inputs, click a pixel (the
spectrum plot updates), and press **Run 50-frame benchmark**. The latency shown is the Space's own
CPU, which will be slower than a laptop.

| Symptom | Likely cause | Fix |
|---|---|---|
| Build fails on `pip install`, version conflict with gradio | `sdk_version` and `gradio==` pin differ | keep them equal (the prepare script checks this); bump both together |
| `ModuleNotFoundError` in Container logs | package missing from `app/requirements.txt` | add it with an exact pin, re-run steps 3–4 |
| Push rejected: binary files / files too large | binaries were committed as normal git objects | `git lfs track` them, then `git rm --cached <file> && git add <file>`, commit, push. If the bad commit is already in history, start from a fresh clone |
| App crashes on start: `InvalidProtobuf` / `could not load model`, or samples have the wrong size | LFS pointer files were pushed instead of data | `git lfs pull` in this repo, re-run `prepare_space.sh` (it refuses pointers), push again |
| `Authentication failed` on push | used the account password, or a read-only token | use a **Write** token as the password |
| Space shows *Runtime error* after working before | Space restarted after sleeping and something changed | read **Logs → Container**; *Factory reboot* in **Settings** rebuilds from scratch |
| Space is slow the first time after a while | free Space was asleep | expected; the first request wakes it |
| `python_version` ignored or unsupported | value not available on Spaces | remove the line (Spaces default) or pick a listed version |

**Update the Space later** after changing the app, models or samples in this repo:

```bash
# in this repo (only if models/samples changed)
make demo-models samples
make test
scripts/prepare_space.sh ~/hf-space

cd ~/hf-space
git add -A
git status
git commit -m "Update demo"
git push
```

Commit the same change in this repo too, so GitHub and the Space stay in sync.

## 6. Embed the Space in another website

Use the Space's direct URL `https://<hf-user>-<space>.hf.space`: lowercase, with `/` replaced by `-`.
The **⋮ → Embed this Space** menu on the Space page shows the exact URL and snippets.

**Option A: iframe** (works anywhere):

```html
<iframe
  src="https://<hf-user>-<space>.hf.space"
  title="Hyperspectral tissue classification demo"
  width="100%" height="1200" style="border:0"
  loading="lazy"
  allow="clipboard-write"></iframe>
```

**Option B: Gradio web component** (resizes to its content, loads lazily):

```html
<script type="module"
  src="https://gradio.s3-us-west-2.amazonaws.com/6.32.0/gradio.js"></script>

<gradio-app src="https://<hf-user>-<space>.hf.space"></gradio-app>
```

Keep the script version equal to `sdk_version`. A sleeping free Space shows a loading screen inside
the embed until it wakes. The Space must be **public** to be embedded without login.
