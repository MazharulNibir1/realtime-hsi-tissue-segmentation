#!/usr/bin/env bash
# Copy the Gradio app (app/) into a cloned Hugging Face Space repo. Does NOT commit or push.
#
#   scripts/prepare_space.sh ~/hf-space
#
# Copies: app.py, inference.py, requirements.txt, README.md, models/, samples/
# Keeps the Space's own .git and .gitattributes; removes files in the Space that no longer exist in app/.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)/app"
DEST="${1:-}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -n "$DEST" ]] || die "usage: $0 <path-to-cloned-space>"
[[ -d "$DEST/.git" ]] || die "$DEST is not a git clone (clone your Space first, see docs/DEPLOY_GRADIO.md)"
DEST="$(cd "$DEST" && pwd)"
[[ "$DEST" != "$(cd "$SRC/.." && pwd)"* ]] || die "the Space clone must be outside this project repo"
git -C "$DEST" remote get-url origin | grep -q "huggingface.co/spaces/" \
  || die "$DEST has no huggingface.co/spaces remote named origin"
command -v git-lfs >/dev/null || die "git-lfs is not installed (brew install git-lfs)"

# Refuse to copy LFS pointer files (happens if this repo was cloned without git-lfs).
while IFS= read -r -d '' f; do
  if head -c 40 "$f" | grep -q "git-lfs.github.com/spec"; then
    die "$f is an LFS pointer, not data. Run 'git lfs pull' in this repo first."
  fi
done < <(find "$SRC/models" "$SRC/samples" -type f \( -name '*.onnx' -o -name '*.bin' \) -print0)

for f in app.py inference.py requirements.txt README.md; do
  [[ -f "$SRC/$f" ]] || die "missing $SRC/$f"
done

rsync -a --delete \
  --exclude '.git/' --exclude '.gitattributes' \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' \
  "$SRC/" "$DEST/"

# Version guard: README sdk_version must equal the gradio pin (Spaces installs gradio from sdk_version).
sdk=$(grep -E '^sdk_version:' "$DEST/README.md" | awk '{print $2}')
pin=$(grep -E '^gradio==' "$DEST/requirements.txt" | cut -d= -f3)
[[ "$sdk" == "$pin" ]] || die "README sdk_version ($sdk) != requirements gradio==$pin"

echo "Copied app/ -> $DEST"
echo "Files:"; (cd "$DEST" && find . -path ./.git -prune -o -type f -print | sort | sed 's|^\./|  |')
echo "Total size: $(du -sh --exclude=.git "$DEST" 2>/dev/null | cut -f1 || du -sh "$DEST" | cut -f1)"
cat <<EOF

Next (run these yourself, see docs/DEPLOY_GRADIO.md step 4):
  cd "$DEST"
  git lfs install
  git lfs track "*.onnx" "samples/*.bin"
  git add .gitattributes
  git add -A
  git lfs ls-files        # expect 2 model.onnx + 10 samples/*.bin
  git commit -m "Deploy hyperspectral tissue demo"
  git push
EOF
