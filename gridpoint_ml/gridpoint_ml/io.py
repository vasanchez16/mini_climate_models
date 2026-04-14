"""
io.py — Config loading and dynamic import of user training scripts.
"""
from __future__ import annotations

import importlib.util
import sys
import os
from types import ModuleType

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


def load_config(path: str) -> dict:
    """Load and return a TOML config file as a dict."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_training_module(script_path: str) -> ModuleType:
    """
    Dynamically import a user-provided training script.

    The script must expose:
      - train(X, y) -> model
      - save(model, path)
      - load(path) -> model
    """
    abs_path = os.path.abspath(script_path)
    spec = importlib.util.spec_from_file_location("_user_train_module", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load training script: {abs_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_user_train_module"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    for required in ("train", "save", "load"):
        if not hasattr(module, required):
            raise AttributeError(
                f"Training script '{abs_path}' must define a '{required}' function."
            )

    return module
