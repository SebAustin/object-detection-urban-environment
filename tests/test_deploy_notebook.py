"""The deploy notebook must install what it imports: conda_python3 on AL2023 ships without OpenCV."""

import json

from conftest import REPO_ROOT

DEPLOY_NB = REPO_ROOT / "2_run_inference" / "2_deploy_model.ipynb"


def _code_cells() -> list[str]:
    nb = json.loads(DEPLOY_NB.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_install_cell_installs_opencv_because_cv2_is_imported():
    cells = _code_cells()
    install = next(src for src in cells if src.lstrip().startswith("%pip install"))
    assert any("import cv2" in src for src in cells)
    assert "opencv-python-headless" in install


def test_artifact_comes_from_best_model_file():
    assert any("best_model.json" in src for src in _code_cells())
