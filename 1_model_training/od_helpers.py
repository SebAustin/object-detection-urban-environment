"""Helpers shared by the per-candidate training notebooks (1_train_model_<candidate>.ipynb).

Kept free of SageMaker SDK imports so it can be unit-tested locally (see tests/test_od_helpers.py).
Runs on the SageMaker notebook's conda_python3 kernel (Python 3.10+).
"""

from __future__ import annotations

import json
import re
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

REQUIRED_SCRIPTS = ("run_training.sh", "model_main_tf2.py", "exporter_main_v2.py")
JOB_NAME_PREFIX = "tf2-od-"
CHECKPOINT_DIRNAME = "checkpoint"
_NUM_STEPS_RE = re.compile(r"^\s*num_steps:\s*(\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Candidate:
    key: str
    display_name: str
    archive: str
    feature_extractor: str
    checkpoint_url: str


def load_candidate(key: str, registry_path: Path) -> Candidate:
    """Return the candidate entry from candidates.json, or raise KeyError."""
    registry = json.loads(Path(registry_path).read_text())
    base_url = registry["checkpoint_base_url"].rstrip("/")
    for entry in registry["candidates"]:
        if entry["key"] == key:
            return Candidate(
                key=entry["key"],
                display_name=entry["display_name"],
                archive=entry["archive"],
                feature_extractor=entry["feature_extractor"],
                checkpoint_url=f"{base_url}/{entry['archive']}.tar.gz",
            )
    known = ", ".join(e["key"] for e in registry["candidates"])
    raise KeyError(f"Unknown candidate '{key}'. Known candidates: {known}")


def read_num_steps(config_path: Path) -> int:
    """Read train_config.num_steps so the job's num_train_steps can never drift from the config."""
    match = _NUM_STEPS_RE.search(Path(config_path).read_text())
    if match is None:
        raise ValueError(f"No num_steps found in {config_path}")
    return int(match.group(1))


def job_base_name(key: str) -> str:
    """SageMaker job names allow only alphanumerics and hyphens."""
    return JOB_NAME_PREFIX + key.replace("_", "-")


def tensorboard_prefix(bucket: str, key: str) -> str:
    return f"s3://{bucket}/logs/{key}/"


def stage_build_dir(key: str, training_dir: Path) -> Path:
    """Assemble build/<key>/ = shared scripts + this candidate's pipeline.config.

    An existing build/<key>/checkpoint/ is kept so the checkpoint is downloaded only once.
    Everything else in the build dir is replaced, so stale files never reach a training job.
    """
    training_dir = Path(training_dir)
    source_dir = training_dir / "source_dir"
    missing = [name for name in REQUIRED_SCRIPTS if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing in {source_dir}: {', '.join(missing)}. "
            "Run the cell that clones tensorflow/models and copies the OD API scripts first."
        )
    config = training_dir / "configs" / key / "pipeline.config"
    if not config.is_file():
        raise FileNotFoundError(f"No pipeline.config for candidate '{key}' at {config}")

    build = training_dir / "build" / key
    build.mkdir(parents=True, exist_ok=True)
    for child in build.iterdir():
        if child.name == CHECKPOINT_DIRNAME:
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    for name in REQUIRED_SCRIPTS:
        shutil.copy2(source_dir / name, build / name)
    shutil.copy2(config, build / "pipeline.config")
    return build


def extract_checkpoint(archive_path: Path, archive: str, dest: Path) -> list[Path]:
    """Extract only <archive>/checkpoint/* from a Model Zoo tarball into dest, flattening the prefix."""
    prefix = PurePosixPath(archive) / CHECKPOINT_DIRNAME
    dest = Path(dest)
    planned: list[tuple[tarfile.TarInfo, Path]] = []
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if prefix not in path.parents:
                continue
            relative = path.relative_to(prefix)
            if member.issym() or member.islnk() or ".." in relative.parts or relative.is_absolute():
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.isfile():
                planned.append((member, dest.joinpath(*relative.parts)))
        if not planned:
            raise ValueError(f"No checkpoint files under {prefix}/ in {archive_path}")
        written = []
        for member, target in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read archive member: {member.name}")
            with source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)
            written.append(target)
    return written


def ensure_checkpoint(candidate: Candidate, build_dir: Path, download_dir: Path) -> Path:
    """Download and extract the candidate's checkpoint into build_dir/checkpoint, once."""
    ckpt_dir = Path(build_dir) / CHECKPOINT_DIRNAME
    if (ckpt_dir / "ckpt-0.index").is_file():
        return ckpt_dir
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / f"{candidate.archive}.tar.gz"
    if not archive_path.is_file():
        urllib.request.urlretrieve(candidate.checkpoint_url, archive_path)  # noqa: S310 (fixed zoo URL)
    extract_checkpoint(archive_path, candidate.archive, ckpt_dir)
    if not (ckpt_dir / "ckpt-0.index").is_file():
        raise FileNotFoundError(f"ckpt-0.index missing after extracting {archive_path}")
    return ckpt_dir


class SageMakerClient(Protocol):
    def list_training_jobs(self, **kwargs: Any) -> dict: ...

    def describe_training_job(self, TrainingJobName: str) -> dict: ...  # noqa: N803 (boto3 casing)


def latest_training_job(sm_client: SageMakerClient, base_name: str) -> dict:
    """Summarise the newest training job whose name contains base_name."""
    response = sm_client.list_training_jobs(
        NameContains=base_name, SortBy="CreationTime", SortOrder="Descending", MaxResults=1
    )
    summaries = response.get("TrainingJobSummaries", [])
    if not summaries:
        raise LookupError(f"No training job found whose name contains '{base_name}'")
    job = sm_client.describe_training_job(TrainingJobName=summaries[0]["TrainingJobName"])
    return {
        "job_name": job["TrainingJobName"],
        "status": job["TrainingJobStatus"],
        "model_artifact": job.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
        "billable_seconds": job.get("BillableTimeInSeconds"),
        "training_seconds": job.get("TrainingTimeInSeconds"),
    }
