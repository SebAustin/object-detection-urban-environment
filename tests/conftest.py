"""Shared fixtures: compile the vendored TF OD API protos so configs can be parsed strictly."""

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTO_ROOT = REPO_ROOT / "tools" / "od_protos"
TRAINING_DIR = REPO_ROOT / "1_model_training"

# Make 1_model_training/od_helpers.py importable from tests.
sys.path.insert(0, str(TRAINING_DIR))


@pytest.fixture(scope="session")
def registry() -> dict:
    return json.loads((TRAINING_DIR / "candidates.json").read_text())


@pytest.fixture(scope="session")
def pipeline_pb2(tmp_path_factory):
    protoc = shutil.which("protoc")
    if protoc is None:
        pytest.skip("protoc is not installed; cannot compile the OD API protos")
    out_dir = tmp_path_factory.mktemp("od_pb2")
    protos = sorted((PROTO_ROOT / "object_detection" / "protos").glob("*.proto"))
    subprocess.run(
        [protoc, f"--proto_path={PROTO_ROOT}", f"--python_out={out_dir}", *map(str, protos)],
        check=True,
        capture_output=True,
    )
    for pkg in ("object_detection", "object_detection/protos"):
        (out_dir / pkg / "__init__.py").touch()
    sys.path.insert(0, str(out_dir))
    try:
        return importlib.import_module("object_detection.protos.pipeline_pb2")
    finally:
        sys.path.remove(str(out_dir))
