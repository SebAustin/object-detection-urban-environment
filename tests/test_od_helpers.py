"""Unit tests for the helpers shared by the per-candidate training notebooks."""

import io
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

import od_helpers as h
from conftest import TRAINING_DIR

JOB_NAME_RE = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$")
TIMESTAMP_SUFFIX_LEN = 24  # SageMaker appends "-YYYYMMDDHHMMSS..." style suffixes


# ---------- registry ----------

def test_load_candidate_returns_registry_entry():
    c = h.load_candidate("ssd_resnet50_v1_fpn_640", TRAINING_DIR / "candidates.json")
    assert c.archive == "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8"
    assert c.checkpoint_url.endswith("/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz")


def test_load_candidate_rejects_unknown_key():
    with pytest.raises(KeyError, match="Unknown candidate"):
        h.load_candidate("yolo", TRAINING_DIR / "candidates.json")


# ---------- config ----------

def test_read_num_steps_reads_train_config_value():
    steps = h.read_num_steps(TRAINING_DIR / "configs" / "efficientdet_d1" / "pipeline.config")
    assert steps == 2000


def test_read_num_steps_fails_without_value(tmp_path):
    cfg = tmp_path / "pipeline.config"
    cfg.write_text("train_config {\n  batch_size: 8\n}\n")
    with pytest.raises(ValueError, match="num_steps"):
        h.read_num_steps(cfg)


# ---------- job naming ----------

@pytest.mark.parametrize("key", ["efficientdet_d1", "ssd_mobilenet_v2_fpnlite_640", "ssd_resnet50_v1_fpn_640"])
def test_job_base_name_is_valid_sagemaker_name(key):
    name = h.job_base_name(key)
    assert JOB_NAME_RE.match(name)
    assert len(name) + TIMESTAMP_SUFFIX_LEN <= 63


def test_tensorboard_prefix_is_per_candidate():
    assert h.tensorboard_prefix("my-bucket", "efficientdet_d1") == "s3://my-bucket/logs/efficientdet_d1/"


# ---------- staging ----------

def _fake_training_dir(tmp_path: Path) -> Path:
    src = tmp_path / "source_dir"
    src.mkdir()
    for name in h.REQUIRED_SCRIPTS:
        (src / name).write_text(f"# {name}\n")
    cfg = tmp_path / "configs" / "efficientdet_d1"
    cfg.mkdir(parents=True)
    (cfg / "pipeline.config").write_text("train_config {\n  num_steps: 10\n}\n")
    return tmp_path


def test_stage_build_dir_assembles_scripts_and_config(tmp_path):
    root = _fake_training_dir(tmp_path)
    build = h.stage_build_dir("efficientdet_d1", root)
    assert build == root / "build" / "efficientdet_d1"
    assert sorted(p.name for p in build.iterdir()) == sorted([*h.REQUIRED_SCRIPTS, "pipeline.config"])


def test_stage_build_dir_replaces_stale_files_but_keeps_checkpoint(tmp_path):
    root = _fake_training_dir(tmp_path)
    build = h.stage_build_dir("efficientdet_d1", root)
    (build / "stale.txt").write_text("old")
    (build / "checkpoint").mkdir()
    (build / "checkpoint" / "ckpt-0.index").write_text("x")
    h.stage_build_dir("efficientdet_d1", root)
    assert not (build / "stale.txt").exists()
    assert (build / "checkpoint" / "ckpt-0.index").exists()


def test_stage_build_dir_fails_when_od_scripts_not_copied(tmp_path):
    root = _fake_training_dir(tmp_path)
    (root / "source_dir" / "model_main_tf2.py").unlink()
    with pytest.raises(FileNotFoundError, match="model_main_tf2.py"):
        h.stage_build_dir("efficientdet_d1", root)


# ---------- checkpoint download ----------

def _make_archive(path: Path, archive: str, extra: dict[str, bytes] | None = None) -> None:
    members = {
        f"{archive}/checkpoint/ckpt-0.index": b"idx",
        f"{archive}/checkpoint/ckpt-0.data-00000-of-00001": b"data",
        f"{archive}/pipeline.config": b"zoo config",
        **(extra or {}),
    }
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_extract_checkpoint_keeps_only_checkpoint_files(tmp_path):
    archive_path = tmp_path / "a.tar.gz"
    _make_archive(archive_path, "zoo_model")
    dest = tmp_path / "out"
    h.extract_checkpoint(archive_path, "zoo_model", dest)
    assert sorted(p.name for p in dest.iterdir()) == ["ckpt-0.data-00000-of-00001", "ckpt-0.index"]


def test_extract_checkpoint_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "evil.tar.gz"
    _make_archive(archive_path, "zoo_model", {"zoo_model/checkpoint/../../escape.txt": b"x"})
    with pytest.raises(ValueError, match="Unsafe"):
        h.extract_checkpoint(archive_path, "zoo_model", tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_checkpoint_fails_when_no_checkpoint_present(tmp_path):
    archive_path = tmp_path / "a.tar.gz"
    with tarfile.open(archive_path, "w:gz"):
        pass
    with pytest.raises(ValueError, match="No checkpoint"):
        h.extract_checkpoint(archive_path, "zoo_model", tmp_path / "out")


# ---------- training job lookup ----------

class FakeSageMaker:
    def __init__(self, jobs: list[dict], details: dict[str, dict]):
        self._jobs, self._details = jobs, details
        self.calls: list[dict] = []

    def list_training_jobs(self, **kwargs):
        self.calls.append(kwargs)
        return {"TrainingJobSummaries": self._jobs}

    def describe_training_job(self, TrainingJobName):
        return self._details[TrainingJobName]


def test_latest_training_job_summarises_newest_job():
    t = datetime(2026, 11, 4, tzinfo=timezone.utc)
    fake = FakeSageMaker(
        jobs=[{"TrainingJobName": "tf2-od-x-2", "CreationTime": t}],
        details={
            "tf2-od-x-2": {
                "TrainingJobName": "tf2-od-x-2",
                "TrainingJobStatus": "Completed",
                "ModelArtifacts": {"S3ModelArtifacts": "s3://b/out/model.tar.gz"},
                "BillableTimeInSeconds": 3600,
                "TrainingTimeInSeconds": 3500,
            }
        },
    )
    job = h.latest_training_job(fake, "tf2-od-x")
    assert job == {
        "job_name": "tf2-od-x-2",
        "status": "Completed",
        "model_artifact": "s3://b/out/model.tar.gz",
        "billable_seconds": 3600,
        "training_seconds": 3500,
    }
    assert fake.calls[0]["NameContains"] == "tf2-od-x"
    assert fake.calls[0]["SortBy"] == "CreationTime"


def test_latest_training_job_raises_when_none_found():
    with pytest.raises(LookupError, match="No training job"):
        h.latest_training_job(FakeSageMaker([], {}), "tf2-od-x")
