"""A helper for loading the shared utilities of the recipes by their paths."""

import importlib.util
import os
from types import ModuleType

UTILS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "examples", "TEMPLATE", "utils"
)


def load_module(name: str) -> ModuleType:
    """Load a module of the shared utilities."""
    path = os.path.join(UTILS_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"template_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
