#!/usr/bin/env bash
# Check a deployed (or locally previewed) browser demo: headers, wasm type/compression, real assets.
#   web/scripts/verify-deploy.sh https://<project>.vercel.app
# Run `npm run build` in web/ first: the hashed .wasm file names are read from web/dist/assets.
set -uo pipefail

URL="${1:?usage: $0 <base-url>}"
URL="${URL%/}"
DIST="$(cd "$(dirname "$0")/.." && pwd)/dist/assets"
fail=0
ok() { printf '  ok    %s\n' "$*"; }
bad() { printf '  FAIL  %s\n' "$*"; fail=1; }
hdr() { curl -sI -H 'Accept-Encoding: br, gzip' "$1" | tr -d '\r' | awk -v k="$(echo "$2" | tr 'A-Z' 'a-z')" 'tolower($0) ~ "^"k":" {sub(/^[^:]*: */, ""); print; exit}'; }

echo "Page: $URL/"
code=$(curl -s -o /dev/null -w '%{http_code}' "$URL/")
if [[ "$code" == 200 ]]; then ok "HTTP 200"
else
  bad "HTTP $code"
  loc=$(curl -sI "$URL/" | tr -d '\r' | awk 'tolower($1)=="location:"{print $2}')
  if [[ "$loc" == *vercel.com/sso* || "$loc" == *vercel.com/login* ]]; then
    echo "  This URL is behind Vercel Deployment Protection (login). Use the public production domain"
    echo "  (Project -> Settings -> Domains) or relax protection (Settings -> Deployment Protection)."
    exit 1
  fi
fi
[[ "$(hdr "$URL/" Cross-Origin-Opener-Policy)" == same-origin ]] && ok "COOP same-origin" || bad "COOP missing (multithreaded WASM disabled)"
[[ "$(hdr "$URL/" Cross-Origin-Embedder-Policy)" == require-corp ]] && ok "COEP require-corp" || bad "COEP missing (multithreaded WASM disabled)"

echo "WebAssembly runtime:"
shopt -s nullglob
wasms=("$DIST"/*.wasm)
[[ ${#wasms[@]} -gt 0 ]] || { bad "no .wasm in web/dist/assets (run npm run build in web/ first)"; }
for f in "${wasms[@]}"; do
  u="$URL/assets/$(basename "$f")"
  ct=$(hdr "$u" Content-Type); enc=$(hdr "$u" Content-Encoding); cc=$(hdr "$u" Cache-Control)
  [[ "$ct" == application/wasm* ]] && ok "$(basename "$f"): Content-Type $ct" || bad "$(basename "$f"): Content-Type '$ct'"
  [[ -n "$enc" ]] && ok "  compressed ($enc)" || printf '  note  not compressed by this server\n'
  [[ "$cc" == *immutable* ]] && ok "  Cache-Control $cc" || printf '  note  Cache-Control: %s\n' "${cc:-none}"
done

echo "Models and samples are real data (not Git LFS pointers):"
for p in models/full/model.onnx models/snapshot16/model.onnx samples/liver_P095.cube.bin; do
  pcode=$(curl -s -o /dev/null -w '%{http_code}' "$URL/$p")
  head=$(curl -s -r 0-40 "$URL/$p" | head -c 40)
  size=$(curl -sI "$URL/$p" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
  if [[ "$pcode" != 200 && "$pcode" != 206 ]]; then bad "$p: HTTP $pcode"
  elif [[ "$head" == version\ https://git-lfs* ]]; then bad "$p is an LFS pointer (enable Git LFS in Vercel, redeploy)"
  else ok "$p (${size:-?} bytes)"; fi
done

[[ $fail == 0 ]] && echo "All checks passed." || echo "Some checks FAILED."
exit $fail
