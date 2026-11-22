"""Read individual members of a (possibly ZIP64) archive through byte-range reads.

HeiPorSPECTRAL is distributed as one ~845 GiB zip. The server supports HTTP range requests, so we
parse the central directory once and then fetch only the members we need. Every member is
verified against the CRC32 stored in the archive.

Resumability: compressed bytes are appended to `<dest>.part`; an interrupted download continues
from the current `.part` length. The member is inflated and CRC-checked only once complete.
"""

from __future__ import annotations

import http.client
import os
import ssl
import struct
import threading
import time
import urllib.parse
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOCATOR_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CDIR_SIG = b"PK\x01\x02"
LOCAL_SIG = b"PK\x03\x04"

STORED, DEFLATED = 0, 8


class ChecksumError(RuntimeError):
    pass


class RangeReader(Protocol):
    size: int

    def read(self, start: int, length: int) -> bytes: ...


class FileRangeReader:
    """Range reader over a local file (used in tests)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.size = self.path.stat().st_size

    def read(self, start: int, length: int) -> bytes:
        with self.path.open("rb") as f:
            f.seek(start)
            return f.read(length)


class HttpRangeReader:
    """Range reader over HTTP(S): one persistent keep-alive connection per thread, shared SSL
    context, retries with exponential backoff. Thread-safe."""

    def __init__(self, url: str, timeout: float = 60.0, retries: int = 6):
        self.url = url
        self.timeout = timeout
        self.retries = retries
        u = urllib.parse.urlsplit(url)
        self._https = u.scheme == "https"
        self._host = u.netloc
        self._path = u.path + (f"?{u.query}" if u.query else "")
        self._ssl = ssl.create_default_context() if self._https else None
        self._local = threading.local()
        status, headers, _ = self._request("HEAD", {})
        if status != 200:
            raise RuntimeError(f"HEAD {url} returned {status}")
        self.size = int(headers["Content-Length"])
        self.etag = (headers.get("ETag") or "").strip('"')
        if (headers.get("Accept-Ranges") or "").lower() != "bytes":
            raise RuntimeError(f"Server does not advertise byte ranges for {url}")

    def _conn(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            if self._https:
                conn = http.client.HTTPSConnection(
                    self._host, timeout=self.timeout, context=self._ssl
                )
            else:
                conn = http.client.HTTPConnection(self._host, timeout=self.timeout)
            self._local.conn = conn
        return conn

    def _drop_conn(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
        self._local.conn = None

    def _request(self, method: str, headers: dict[str, str], expect_len: int | None = None):
        delay = 2.0
        for attempt in range(self.retries + 1):
            try:
                conn = self._conn()
                conn.request(method, self._path, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                if resp.status >= 500:
                    raise ConnectionError(f"HTTP {resp.status}")
                if expect_len is not None and len(data) != expect_len:
                    raise ConnectionError(f"Short read: {len(data)} of {expect_len} bytes")
                return resp.status, resp.headers, data
            except (http.client.HTTPException, OSError) as e:  # OSError covers timeouts/resets
                self._drop_conn()
                if attempt == self.retries:
                    raise ConnectionError(f"{method} {self.url} failed: {e!r}") from e
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise AssertionError("unreachable")

    def read(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b""
        rng = {"Range": f"bytes={start}-{start + length - 1}"}
        status, _, data = self._request("GET", rng, expect_len=length)
        if status != 206:
            raise RuntimeError(f"Expected HTTP 206, got {status}")
        return data


@dataclass(frozen=True)
class ZipMember:
    name: str
    method: int
    compressed_size: int
    size: int
    header_offset: int
    crc32: int


def read_central_directory_bytes(reader: RangeReader) -> bytes:
    """Return the raw central directory (handles ZIP64)."""
    tail_len = min(reader.size, 65536 + 22)
    tail_start = reader.size - tail_len
    tail = reader.read(tail_start, tail_len)
    i = tail.rfind(EOCD_SIG)
    if i < 0:
        raise ValueError("End of central directory record not found")
    _, _, _, n_total, cd_size, cd_offset = struct.unpack("<HHHHII", tail[i + 4 : i + 20])
    if 0xFFFF in (n_total,) or 0xFFFFFFFF in (cd_size, cd_offset):
        loc = tail[i - 20 : i] if i >= 20 else reader.read(tail_start + i - 20, 20)
        if loc[:4] != EOCD64_LOCATOR_SIG:
            raise ValueError("ZIP64 locator not found")
        (eocd64_offset,) = struct.unpack("<Q", loc[8:16])
        rec = reader.read(eocd64_offset, 56)
        if rec[:4] != EOCD64_SIG:
            raise ValueError("ZIP64 end of central directory record not found")
        cd_size, cd_offset = struct.unpack("<QQ", rec[40:56])
    return reader.read(cd_offset, cd_size)


def parse_central_directory(cd: bytes) -> dict[str, ZipMember]:
    members: dict[str, ZipMember] = {}
    p = 0
    while p < len(cd):
        if cd[p : p + 4] != CDIR_SIG:
            raise ValueError(f"Bad central directory signature at {p}")
        method, crc, csize, usize, nlen, elen, clen = struct.unpack(
            "<HxxxxIIIHHH", cd[p + 10 : p + 34]
        )
        (offset,) = struct.unpack("<I", cd[p + 42 : p + 46])
        name = cd[p + 46 : p + 46 + nlen].decode("utf-8")
        extra = cd[p + 46 + nlen : p + 46 + nlen + elen]
        q = 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack("<HH", extra[q : q + 4])
            if hid == 0x0001:  # ZIP64 extended info: only fields that overflowed, in this order
                body, k = extra[q + 4 : q + 4 + hlen], 0
                if usize == 0xFFFFFFFF:
                    (usize,) = struct.unpack("<Q", body[k : k + 8])
                    k += 8
                if csize == 0xFFFFFFFF:
                    (csize,) = struct.unpack("<Q", body[k : k + 8])
                    k += 8
                if offset == 0xFFFFFFFF:
                    (offset,) = struct.unpack("<Q", body[k : k + 8])
            q += 4 + hlen
        members[name] = ZipMember(name, method, csize, usize, offset, crc)
        p += 46 + nlen + elen + clen
    return members


def _data_offset(reader: RangeReader, m: ZipMember) -> int:
    hdr = reader.read(m.header_offset, 30)
    if hdr[:4] != LOCAL_SIG:
        raise ValueError(f"Bad local header for {m.name}")
    nlen, elen = struct.unpack("<HH", hdr[26:30])
    return m.header_offset + 30 + nlen + elen


def _inflate(chunks: Iterator[bytes], method: int) -> Iterator[bytes]:
    if method == STORED:
        yield from chunks
        return
    if method != DEFLATED:
        raise NotImplementedError(f"Compression method {method} not supported")
    d = zlib.decompressobj(-15)
    for c in chunks:
        out = d.decompress(c)
        if out:
            yield out
    tail = d.flush()
    if tail:
        yield tail


def read_member(reader: RangeReader, m: ZipMember) -> bytes:
    """Fetch a (small) member fully into memory and verify its CRC32."""
    raw = reader.read(_data_offset(reader, m), m.compressed_size)
    out = b"".join(_inflate(iter([raw]), m.method))
    if len(out) != m.size or zlib.crc32(out) != m.crc32:
        raise ChecksumError(f"CRC/size mismatch for {m.name}")
    return out


def safe_local_path(root: Path, member_name: str) -> Path:
    """Map an archive member name to a path under `root`, rejecting path traversal."""
    p = PurePosixPath(member_name)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe member name: {member_name}")
    return root.joinpath(*p.parts)


def file_crc32(path: Path, chunk: int = 1 << 22) -> int:
    crc = 0
    with path.open("rb") as f:
        while b := f.read(chunk):
            crc = zlib.crc32(b, crc)
    return crc


def download_member(
    reader: RangeReader,
    m: ZipMember,
    dest: Path,
    chunk: int = 8 << 20,
    progress: Callable[[int], None] | None = None,
) -> str:
    """Download one member to `dest` (resumable, CRC-verified).

    Returns "skipped" if `dest` already exists with the expected size, else "downloaded".
    Full CRC re-verification of existing files is done by `verify_file`.
    """
    if dest.exists() and dest.stat().st_size == m.size:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > m.compressed_size:
        part.unlink()
        have = 0
    if progress and have:
        progress(have)
    start = _data_offset(reader, m)
    with part.open("ab") as f:
        while have < m.compressed_size:
            n = min(chunk, m.compressed_size - have)
            f.write(reader.read(start + have, n))
            f.flush()
            have += n
            if progress:
                progress(n)

    def _chunks() -> Iterator[bytes]:
        with part.open("rb") as f:
            while b := f.read(chunk):
                yield b

    tmp = dest.with_name(dest.name + ".tmp")
    crc, size = 0, 0
    with tmp.open("wb") as out:
        for b in _inflate(_chunks(), m.method):
            crc = zlib.crc32(b, crc)
            size += len(b)
            out.write(b)
    if crc != m.crc32 or size != m.size:
        tmp.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
        raise ChecksumError(f"CRC/size mismatch for {m.name}; partial data removed, rerun to retry")
    os.replace(tmp, dest)
    part.unlink()
    return "downloaded"


def verify_file(path: Path, m: ZipMember) -> bool:
    return path.exists() and path.stat().st_size == m.size and file_crc32(path) == m.crc32
