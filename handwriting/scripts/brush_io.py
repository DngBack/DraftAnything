"""Load BRUSH samples without executing anything from the pickle.

BRUSH ships as pickles; only numpy array reconstruction is allowed through.
"""
import pickle
from pathlib import Path

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        import importlib
        return getattr(importlib.import_module(module), name)


def load_sample(path: Path):
    """Returns (sentence, drawing[N, 3] of x, y, eos, label[N, M])."""
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()
