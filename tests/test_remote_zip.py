import os
import struct
import zipfile

import numpy as np
import pytest

from hsi.data.remote_zip import (
    ChecksumError,
    FileRangeReader,
    ZipMember,
    download_member,
    parse_central_directory,
    read_central_directory_bytes,
    read_member,
    safe_local_path,
    verify_file,
)


@pytest.fixture
def archive(tmp_path):
    rng = np.random.default_rng(0)
    payloads = {
        "D/small.json": b'{"a": 1}',
        "D/big.bin": rng.integers(0, 4, 300_000, dtype=np.uint8).tobytes(),  # compressible
        "D/stored.bin": os.urandom(5000),
    }
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w", allowZip64=True) as z:
        for name, data in payloads.items():
            zi = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED if "stored" in name else zipfile.ZIP_DEFLATED
            with z.open(zi, "w", force_zip64=True) as f:  # ZIP64 local headers
                f.write(data)
    return path, payloads


def test_parse_zip64_extra_field():
    name = b"x/y.bin"
    extra = struct.pack("<HHQQQ", 0x0001, 24, 5_000_000_000, 4_900_000_000, 6_000_000_000)
    rec = (
        b"PK\x01\x02"
        + struct.pack(
            "<HHHHHHIIIHHHHHII",
            45,
            45,
            0,
            8,
            0,
            0,
            1234,
            0xFFFFFFFF,
            0xFFFFFFFF,
            len(name),
            len(extra),
            0,
            0,
            0,
            0,
            0xFFFFFFFF,
        )
        + name
        + extra
    )
    (m,) = parse_central_directory(rec).values()
    assert (m.size, m.compressed_size, m.header_offset, m.crc32) == (
        5_000_000_000,
        4_900_000_000,
        6_000_000_000,
        1234,
    )


def _members(path):
    reader = FileRangeReader(path)
    return reader, parse_central_directory(read_central_directory_bytes(reader))


def test_parse_and_read_members(archive):
    path, payloads = archive
    reader, members = _members(path)
    for name, data in payloads.items():
        assert read_member(reader, members[name]) == data


def test_download_member_resume_and_verify(archive, tmp_path):
    path, payloads = archive
    reader, members = _members(path)
    m = members["D/big.bin"]
    dest = tmp_path / "out" / "big.bin"
    dest.parent.mkdir()
    # Simulate an interrupted download: a .part with the first third of the compressed bytes.
    from hsi.data.remote_zip import _data_offset

    start = _data_offset(reader, m)
    (dest.parent / "big.bin.part").write_bytes(reader.read(start, m.compressed_size // 3))
    assert download_member(reader, m, dest, chunk=4096) == "downloaded"
    assert dest.read_bytes() == payloads["D/big.bin"]
    assert not (dest.parent / "big.bin.part").exists()
    assert verify_file(dest, m)
    assert download_member(reader, m, dest) == "skipped"


def test_crc_mismatch_raises_and_cleans_up(archive, tmp_path):
    path, _ = archive
    reader, members = _members(path)
    m = members["D/small.json"]
    bad = ZipMember(m.name, m.method, m.compressed_size, m.size, m.header_offset, m.crc32 ^ 1)
    dest = tmp_path / "small.json"
    with pytest.raises(ChecksumError):
        download_member(reader, bad, dest)
    assert not dest.exists() and not (tmp_path / "small.json.part").exists()


def test_safe_local_path(tmp_path):
    assert safe_local_path(tmp_path, "a/b#c.png") == tmp_path / "a" / "b#c.png"
    for bad in ["../x", "/etc/passwd", "a/../../x"]:
        with pytest.raises(ValueError):
            safe_local_path(tmp_path, bad)
