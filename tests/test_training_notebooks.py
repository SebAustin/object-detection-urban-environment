"""The committed training notebooks must match the generator (sources only; outputs may differ)."""

import ast
import json
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tools"))
import make_training_notebooks as gen  # noqa: E402

KEYS = [entry["key"] for entry in gen.load_registry()]


def sources(nb: dict) -> list[tuple[str, str]]:
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


@pytest.mark.parametrize("key", KEYS)
def test_notebook_is_up_to_date_with_generator(key):
    path = gen.notebook_path(key)
    assert path.is_file(), f"{path.name} missing: run python3 tools/make_training_notebooks.py"
    committed = json.loads(path.read_text())
    entry = next(e for e in gen.load_registry() if e["key"] == key)
    assert sources(committed) == sources(gen.build_notebook(key, entry["display_name"]))


@pytest.mark.parametrize("key", KEYS)
def test_notebook_targets_its_own_candidate(key):
    code = [src for kind, src in sources(json.loads(gen.notebook_path(key).read_text())) if kind == "code"]
    assert any(f'CANDIDATE_KEY = "{key}"' in src for src in code)
    for other in set(KEYS) - {key}:
        assert not any(other in src for src in code), f"{key} notebook mentions {other}"


@pytest.mark.parametrize("key", KEYS)
def test_python_cells_are_valid_syntax(key):
    """Cells with IPython magics (% / !) are skipped; plain Python cells must parse."""
    for kind, src in sources(gen.build_notebook(key, key)):
        lines = src.splitlines()
        if kind != "code" or any(line.lstrip().startswith(("%", "!")) for line in lines):
            continue
        ast.parse(src)


def test_original_single_model_notebook_was_replaced():
    assert not (REPO_ROOT / "1_model_training" / "1_train_model.ipynb").exists()
