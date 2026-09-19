"""Load unchanged official SnAr modules without unrelated eager imports.

Only the summit, summit.strategies and summit.benchmarks package initializers
are bypassed. No stand-in Domain, Experiment, Transform, DataSet or ODE is used.
"""
import hashlib
import importlib
import importlib.machinery
import json
import math
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "vendor" / "summit"
COMMIT = "1de682d05e97adcfb96cd8376e876cef2d6160d3"
BOUNDS = {"tau": (0.5, 2.0), "equiv_pldn": (1.0, 5.0),
          "conc_dfnb": (0.1, 0.5), "temperature": (30.0, 120.0)}
UNITS = {"tau": "min", "equiv_pldn": "equivalents", "conc_dfnb": "mol/L", "temperature": "degC"}
SKIPPED_INITIALIZERS = ("summit", "summit.strategies", "summit.benchmarks")
_loaded = None


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameters(value):
    if not isinstance(value, dict) or set(value) != set(BOUNDS):
        raise ValueError("Exactly tau, equiv_pldn, conc_dfnb and temperature are required")
    out = {}
    for name, (lo, hi) in BOUNDS.items():
        x = value[name]
        if type(x) not in (int, float) or not math.isfinite(x) or not lo <= x <= hi:
            raise ValueError("Nonfinite, nonnumeric or out-of-bounds parameter: " + name)
        out[name] = float(x)
    return out


def load_official(source_root=None):
    global _loaded, UPSTREAM
    requested = Path(source_root).resolve() if source_root is not None else UPSTREAM.resolve()
    if _loaded is not None:
        if requested != UPSTREAM.resolve():
            raise ValueError("Cannot change loaded official source root")
        return _loaded
    UPSTREAM = requested
    lock_path = HERE / "source_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock["commit"] != COMMIT or lock["source_modified"] is not False:
        raise ValueError("Official source identity changed")
    for entry in lock["files"]:
        path = UPSTREAM / entry["path"]
        if not path.is_file() or path.is_symlink() or file_sha(path) != entry["sha256"]:
            raise ValueError("Official source file changed: " + entry["path"])
    for name in SKIPPED_INITIALIZERS:
        if name in sys.modules:
            raise RuntimeError("Summit already imported; use a dedicated oracle/baseline process")
        module = types.ModuleType(name)
        module.__path__ = [str(UPSTREAM.joinpath(*name.split(".")))]
        module.__package__ = name
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
    # These imports execute the original files in their original package names.
    cls = importlib.import_module("summit.benchmarks.snar").SnarBenchmark
    dataset = importlib.import_module("summit.utils.dataset").DataSet
    random_module = importlib.import_module("summit.strategies.random")
    executed = []
    for name, module in sorted(sys.modules.items()):
        if name.startswith("summit") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            relative = path.relative_to(UPSTREAM.resolve()).as_posix()
            expected = next((e for e in lock["files"] if e["path"] == relative), None)
            if expected is None or file_sha(path) != expected["sha256"]:
                raise ValueError("Unexpected executed Summit module: " + name)
            executed.append({"module": name, "path": relative, "sha256": expected["sha256"]})
    proof = {"repository": lock["repository"], "commit": COMMIT,
             "source_lock_sha256": file_sha(lock_path), "official_class": cls.__module__ + "." + cls.__name__,
             "loader_recipe": "source_namespace_skip_unrelated_eager_initializers_v1",
             "skipped_package_initializers": list(SKIPPED_INITIALIZERS),
             "official_source_modified": False, "executed_modules": executed}
    _loaded = cls, dataset, random_module.Random, random_module.LHS, proof
    return _loaded
