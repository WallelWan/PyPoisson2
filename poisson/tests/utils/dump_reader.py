import numpy as np


def _read_bin(path):
    with open(path, "rb") as f:
        count = np.fromfile(f, dtype=np.uint64, count=1)
        if count.size == 0:
            return np.array([], dtype=np.float64)
        count = int(count[0])
        data = np.fromfile(f, dtype=np.float64, count=count)
    return data


def read_density_file(path):
    return _read_bin(path)


def read_gradient_file(path):
    data = _read_bin(path)
    if data.size % 3 != 0:
        raise ValueError("Gradient data size is not divisible by 3")
    return data.reshape((-1, 3))
