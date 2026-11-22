#!/usr/bin/env python
"""Download the approved HeiPorSPECTRAL subset (resumable, CRC32-verified).

Only the selected members of the 845 GiB archive are fetched, via HTTP range requests:
  * intermediates/preprocessing/L1/<image>.blosc   (float16 cube, 480x640x100)
  * data/subjects/<pig>/<ts>/annotations/<ts>#polygon#<annotator>#<organ>#binary.png
  * data/dataset_settings.json                      (label mapping)

Steps: read the central directory (cached) -> read every per-image meta.json (cached) ->
select images (configs/data.yaml) -> write the manifest (tracked in git) -> download.
Re-running skips completed files and resumes partial ones.

Examples:
    python scripts/download_data.py --dry-run         # build manifest, print sizes only
    python scripts/download_data.py --limit 2         # smoke test: first 2 images
    python scripts/download_data.py                   # full subset
    python scripts/download_data.py --verify          # re-check CRC32 of every file on disk
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

from hsi.data.remote_zip import (
    HttpRangeReader,
    ZipMember,
    download_member,
    parse_central_directory,
    read_central_directory_bytes,
    read_member,
    safe_local_path,
    verify_file,
)
from hsi.data.subset import ImageRecord, parse_label_meta, select_images

ROOT = Path(__file__).resolve().parents[1]


def load_members(reader: HttpRangeReader, cache_dir: Path) -> dict[str, ZipMember]:
    cache = cache_dir / "central_directory.bin"
    if cache.exists():
        cd = cache.read_bytes()
    else:
        print("Fetching central directory (~34 MB) ...")
        cd = read_central_directory_bytes(reader)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(cd)
    return parse_central_directory(cd)


def load_all_meta(
    reader: HttpRangeReader,
    members: dict[str, ZipMember],
    prefix: str,
    cache_dir: Path,
    workers: int,
) -> list[ImageRecord]:
    """Fetch every subjects/*/annotations/*_meta.json (cached on disk) and parse it."""
    meta_members = [
        m
        for n, m in members.items()
        if n.startswith(f"{prefix}data/subjects/")
        and n.endswith("_meta.json")
        and "/annotations/" in n
    ]
    out_dir = cache_dir / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)

    def fetch(m: ZipMember) -> tuple[str, dict]:
        parts = m.name.split("/")
        image_name = f"{parts[3]}#{parts[4]}"
        path = out_dir / f"{image_name}.json"
        if not path.exists():
            path.write_bytes(read_member(reader, m))
        return image_name, json.loads(path.read_text())

    records, skipped = [], 0
    with ThreadPoolExecutor(max_workers=max(workers, 16)) as ex:
        futures = [ex.submit(fetch, m) for m in meta_members]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="meta.json", unit="file"):
            image_name, meta = fut.result()
            rec = parse_label_meta(image_name, meta)
            if rec is None:
                skipped += 1
            else:
                records.append(rec)
    print(f"Parsed {len(records)} single-organ images ({skipped} multi-organ images excluded)")
    return records


def member_entry(members: dict[str, ZipMember], name: str) -> dict:
    m = members[name]
    return {"name": name, "size": m.size, "crc32": m.crc32}


def build_manifest(
    cfg: dict, reader: HttpRangeReader, members: dict[str, ZipMember], cache_dir: Path
):
    prefix = cfg["dataset"]["archive_prefix"]
    sel = cfg["selection"]
    records = load_all_meta(reader, members, prefix, cache_dir, cfg["download"]["workers"])
    chosen = select_images(records, sel["angle"], sel["repetition"], sel["max_per_pair"])

    images = []
    for r in chosen:
        files = [f"{prefix}intermediates/preprocessing/L1/{r.image_name}.blosc"]
        ann_dir = f"{prefix}data/subjects/{r.subject}/{r.timestamp}/annotations"
        for a in sel["annotators"]:
            files.append(f"{ann_dir}/{r.timestamp}#polygon#{a}#{r.label}#binary.png")
        missing = [f for f in files if f not in members]
        if missing:
            print(f"WARNING: skipping {r.image_name}, missing members: {missing}")
            continue
        images.append({**r.to_dict(), "files": [member_entry(members, f) for f in files]})

    return {
        "dataset": cfg["dataset"]["name"],
        "source_url": cfg["dataset"]["url"],
        "archive_size": reader.size,
        "licence": "CC BY 4.0 (doi:10.1038/s41597-023-02315-8)",
        "selection": sel,
        "global_files": [member_entry(members, f"{prefix}data/dataset_settings.json")],
        "images": images,
    }


def summarise(manifest: dict) -> None:
    imgs = manifest["images"]
    total = sum(f["size"] for f in manifest["global_files"]) + sum(
        f["size"] for im in imgs for f in im["files"]
    )
    pairs = {(im["subject"], im["label"]) for im in imgs}
    print(f"Images: {len(imgs)} | (pig, organ) pairs: {len(pairs)} | total: {total / 1e9:.2f} GB")
    print("Images per pig:", dict(sorted(Counter(im["subject"] for im in imgs).items())))
    print("Images per organ:", dict(sorted(Counter(im["label"] for im in imgs).items())))
    non_pref = sum(
        1
        for im in imgs
        if (im["angle"], im["repetition"])
        != (manifest["selection"]["angle"], manifest["selection"]["repetition"])
    )
    print(f"Images using a fallback angle/repetition: {non_pref}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=str(ROOT / "configs/data.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="build/print manifest, download nothing")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images")
    ap.add_argument("--rebuild-manifest", action="store_true")
    ap.add_argument("--verify", action="store_true", help="CRC32-verify files on disk and exit")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    cache_dir = ROOT / cfg["paths"]["cache_dir"]
    manifest_path = ROOT / cfg["paths"]["manifest"]
    workers = args.workers or cfg["download"]["workers"]

    reader = HttpRangeReader(cfg["dataset"]["url"])
    if reader.size != cfg["dataset"]["expected_size"]:
        print(
            f"ERROR: remote archive size {reader.size} != expected {cfg['dataset']['expected_size']}"
        )
        return 2
    members = load_members(reader, cache_dir)

    if manifest_path.exists() and not args.rebuild_manifest:
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = build_manifest(cfg, reader, members, cache_dir)
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"Wrote manifest: {manifest_path.relative_to(ROOT)}")
    summarise(manifest)

    images = manifest["images"][: args.limit] if args.limit else manifest["images"]
    entries = manifest["global_files"] + [f for im in images for f in im["files"]]
    for e in entries:  # manifest must agree with the archive
        m = members[e["name"]]
        if (m.size, m.crc32) != (e["size"], e["crc32"]):
            print(f"ERROR: manifest entry disagrees with archive: {e['name']}")
            return 2
    jobs = [(members[e["name"]], safe_local_path(raw_dir, e["name"])) for e in entries]

    if args.verify:
        bad = [str(d) for m, d in tqdm(jobs, desc="verify", unit="file") if not verify_file(d, m)]
        print(f"{len(jobs) - len(bad)}/{len(jobs)} files OK")
        for b in bad[:20]:
            print("  BAD/MISSING:", b)
        return 1 if bad else 0
    if args.dry_run:
        return 0

    todo = [(m, d) for m, d in jobs if not (d.exists() and d.stat().st_size == m.size)]
    need = sum(m.compressed_size for m, _ in todo)
    free = shutil.disk_usage(raw_dir.parent if raw_dir.parent.exists() else ROOT).free
    print(f"To download: {len(todo)} files, {need / 1e9:.2f} GB (free disk: {free / 1e9:.1f} GB)")
    if need * 1.05 > free:
        print("ERROR: not enough free disk space")
        return 2

    bar = tqdm(total=need, unit="B", unit_scale=True, desc="download")
    lock = threading.Lock()

    def progress(n: int) -> None:
        with lock:
            bar.update(n)

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(download_member, reader, m, d, cfg["download"]["chunk_mb"] << 20, progress): m
            for m, d in todo
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:  # keep going; report at the end
                failures.append((futures[fut].name, repr(e)))
    bar.close()
    for name, err in failures:
        print(f"FAILED {name}: {err}")
    print(
        f"Done: {len(todo) - len(failures)}/{len(todo)} downloaded, {len(jobs) - len(todo)} already present"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
