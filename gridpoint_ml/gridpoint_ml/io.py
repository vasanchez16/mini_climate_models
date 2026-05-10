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
    # convert the user-provided path from the TOML to an absolute file path
    # this ensures it works regardless of where the pipeline is run from
    abs_path = os.path.abspath(script_path)

    # create the module spec (blueprint) from the file path
    # "_user_train_module" is just an internal name we assign to this module
    spec = importlib.util.spec_from_file_location("_user_train_module", abs_path)

    # if the spec or its loader is None, the file couldn't be found or isn't a valid python file
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load training script: {abs_path}")

    # create the actual module object from the spec (still not executed yet)
    module = importlib.util.module_from_spec(spec)

    # register the module in Python's module registry so it can be referenced
    # internally if needed (important for multiprocessing to find the module)
    sys.modules["_user_train_module"] = module

    # actually execute the script file, loading all functions into the module object
    # type: ignore comment suppresses a mypy type checking warning here
    spec.loader.exec_module(module)

    # validate that the user's script actually defines the three required functions
    # if any are missing, raise a clear error telling the user what they forgot
    for required in ("train", "save", "load", "predict"):
        if not hasattr(module, required):
            raise AttributeError(
                f"Training script '{abs_path}' must define a '{required}' function."
            )

    # return the fully loaded module so the pipeline can call module.train(), 
    # module.save(), and module.load()
    return module
