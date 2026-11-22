"""Safe reader for the `.blosc` files shipped with HeiPorSPECTRAL (written by the `htc` library).

On-disk layout (htc.utils.blosc_compression.compress_file):
    single array: pickle((shape, dtype)) + blosc frame
    dict:         pickle({key: (shape, dtype, n_bytes)}) + concatenated blosc frames

`pickle.load` on downloaded data can execute arbitrary code, so the header is read with an
unpickler that only resolves `numpy.dtype`; anything else raises. Frames are decoded with
`blosc2`, which reads Blosc1 frames.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import blosc2
import numpy as np

_ALLOWED_GLOBALS = {("numpy", "dtype"): np.dtype}


class _HeaderUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return _ALLOWED_GLOBALS[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"Disallowed global in blosc header: {module}.{name}")


def _check_shape_dtype(shape, dtype) -> tuple[tuple[int, ...], np.dtype]:
    if not (isinstance(shape, tuple) and all(isinstance(s, int) and s >= 0 for s in shape)):
        raise ValueError(f"Invalid shape in blosc header: {shape!r}")
    if not isinstance(dtype, np.dtype):
        raise ValueError(f"Invalid dtype in blosc header: {dtype!r}")
    return shape, dtype


def _decode(frame: bytes, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    buf = blosc2.decompress(frame)
    expected = int(np.prod(shape)) * dtype.itemsize
    if len(buf) != expected:
        raise ValueError(f"Decoded {len(buf)} bytes, expected {expected}")
    return np.frombuffer(buf, dtype=dtype).reshape(shape)


def loads_blosc(data: bytes) -> np.ndarray | dict[str, np.ndarray]:
    f = io.BytesIO(data)
    header = _HeaderUnpickler(f).load()
    if isinstance(header, tuple):
        shape, dtype = _check_shape_dtype(*header)
        return _decode(data[f.tell() :], shape, dtype)
    if isinstance(header, dict):
        out, pos = {}, f.tell()
        for key, (shape, dtype, n) in header.items():
            shape, dtype = _check_shape_dtype(shape, dtype)
            out[key] = _decode(data[pos : pos + n], shape, dtype)
            pos += n
        return out
    raise ValueError(f"Unexpected blosc header type: {type(header).__name__}")


def load_blosc(path: str | Path) -> np.ndarray | dict[str, np.ndarray]:
    return loads_blosc(Path(path).read_bytes())
