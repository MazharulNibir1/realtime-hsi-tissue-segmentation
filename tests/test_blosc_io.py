import io
import pickle

import blosc2
import numpy as np
import pytest

from hsi.data.blosc_io import loads_blosc


def _htc_single(arr: np.ndarray) -> bytes:
    f = io.BytesIO()
    pickle.dump((arr.shape, arr.dtype), f)
    f.write(blosc2.compress(np.ascontiguousarray(arr).tobytes(), typesize=arr.dtype.itemsize))
    return f.getvalue()


def test_single_array_roundtrip():
    arr = np.random.default_rng(0).random((4, 5, 6)).astype(np.float16)
    np.testing.assert_array_equal(loads_blosc(_htc_single(arr)), arr)


def test_dict_roundtrip():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.ones((2, 2), dtype=np.uint8)
    frames = [blosc2.compress(x.tobytes(), typesize=x.dtype.itemsize) for x in (a, b)]
    f = io.BytesIO()
    pickle.dump(
        {"a": (a.shape, a.dtype, len(frames[0])), "b": (b.shape, b.dtype, len(frames[1]))}, f
    )
    for fr in frames:
        f.write(fr)
    out = loads_blosc(f.getvalue())
    np.testing.assert_array_equal(out["a"], a)
    np.testing.assert_array_equal(out["b"], b)


class _Evil:
    def __reduce__(self):
        return (print, ("pwned",))


def test_rejects_arbitrary_globals():
    data = pickle.dumps(_Evil()) + b"\x00" * 16
    with pytest.raises(pickle.UnpicklingError):
        loads_blosc(data)


def test_rejects_size_mismatch():
    arr = np.zeros((2, 2), dtype=np.float16)
    f = io.BytesIO()
    pickle.dump(((3, 3), arr.dtype), f)
    f.write(blosc2.compress(arr.tobytes(), typesize=2))
    with pytest.raises(ValueError):
        loads_blosc(f.getvalue())
