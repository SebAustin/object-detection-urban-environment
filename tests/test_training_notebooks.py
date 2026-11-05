"""The committed training notebooks must match the generator (sources only; outputs may differ)."""

import ast
import re
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


# ---------- tensorflow/models pin (docs/adr/0003) ----------

PIN_FILE = REPO_ROOT / "1_model_training" / "docker" / "TF_MODELS_COMMIT"


def test_tf_models_commit_is_pinned_to_a_full_sha():
    assert re.fullmatch(r"[0-9a-f]{40}", PIN_FILE.read_text().strip())


@pytest.mark.parametrize("key", KEYS)
def test_notebook_checks_out_the_pinned_commit(key):
    code = "\n".join(src for kind, src in sources(gen.build_notebook(key, key)) if kind == "code")
    assert "docker/TF_MODELS_COMMIT" in code
    assert "git clone --depth 1 https://github.com/tensorflow/models.git" not in code, "unpinned clone"


def test_vendored_protos_come_from_the_pinned_commit():
    readme = (REPO_ROOT / "tools" / "od_protos" / "README.md").read_text()
    assert PIN_FILE.read_text().strip()[:7] in readme
