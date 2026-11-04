"""Generate 1_model_training/1_train_model_<candidate>.ipynb for every candidate in candidates.json.

The three notebooks differ only in CANDIDATE_KEY, so they are generated from one template to avoid
copy-paste drift. Re-run after editing the template:  python3 tools/make_training_notebooks.py
tests/test_training_notebooks.py fails if a committed notebook is out of date (outputs are ignored).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "1_model_training"

KERNEL = {"display_name": "conda_python3", "language": "python", "name": "conda_python3"}


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def build_cells(key: str, display_name: str) -> list[dict]:
    return [
        _md(f"""
# Train candidate: {display_name}

This notebook runs **one Run**: it fine-tunes the `{key}` candidate on the Waymo training split and evaluates it on the validation split.
It was generated from `tools/make_training_notebooks.py`, which uses the same template for all three candidates; edit the template there, not here.

* Config: [`configs/{key}/pipeline.config`](configs/{key}/pipeline.config). The training recipe is shared by every candidate (see `docs/adr/0002`).
* `num_classes: 4`, because the label map uses ids 1, 2 and 4 (see `docs/adr/0001`).
* Results, the comparison and the write-up are in [`3_experiment_summary.ipynb`](3_experiment_summary.ipynb).

**Cost:** the training job runs on `ml.g5.xlarge` and stops on its own when it finishes. Stop the notebook instance when you step away.
"""),
        _code('%pip install tensorflow_io "sagemaker>=3,<4" -U'),
        _md("""
**Important:** after the install cell finishes, restart the kernel (**Kernel > Restart**) and run the notebook from the top again.
The next cell checks that SageMaker SDK v3 is loaded.
"""),
        _code("""
from importlib.metadata import version
import sagemaker

sagemaker_version = version("sagemaker")
print(f"SageMaker SDK version: {sagemaker_version}")
# SDK v2 defines sagemaker.__version__, v3 doesn't. If it is still there, v2 is loaded in this kernel.
if not sagemaker_version.startswith("3.") or hasattr(sagemaker, "__version__"):
    raise RuntimeError(
        f"SageMaker SDK {sagemaker_version} is installed, but this notebook needs v3 loaded in the kernel. "
        "Run the install cell above, restart the kernel (Kernel > Restart) and run the notebook from the top again."
    )
"""),
        _code(f"""
import json
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData, SourceCode, TensorBoardOutputConfig

import od_helpers as h

CANDIDATE_KEY = "{key}"
TRAINING_INSTANCE = "ml.g5.xlarge"

candidate = h.load_candidate(CANDIDATE_KEY, Path("candidates.json"))
config_path = Path("configs") / CANDIDATE_KEY / "pipeline.config"
num_train_steps = h.read_num_steps(config_path)  # read from the config, so the two never disagree
base_job_name = h.job_base_name(CANDIDATE_KEY)
print(candidate.display_name, "|", num_train_steps, "steps |", base_job_name)
"""),
        _code("""
role = get_execution_role()
print(role)
"""),
        _code("""
# Public S3 buckets created by Udacity for this project
inputs = {'train': 's3://cd2688-object-detection-tf2/train/',
          'val': 's3://cd2688-object-detection-tf2/val/'}

# One TensorBoard folder per candidate, so the summary notebook can compare the curves.
# It must be a bucket in your own account; the SageMaker default bucket is used here.
tensorboard_bucket = Session().default_bucket()
tensorboard_s3_prefix = h.tensorboard_prefix(tensorboard_bucket, CANDIDATE_KEY)
try:
    boto3.client('s3').head_bucket(Bucket=tensorboard_bucket)
except ClientError as e:
    raise RuntimeError(
        f"Cannot access the S3 bucket '{tensorboard_bucket}' ({e.response['Error']['Code']}). "
        "Use a bucket in your own AWS account for the TensorBoard logs."
    ) from e
print(f"Tensorboard logs will be saved to {tensorboard_s3_prefix}")
"""),
        _md("""
## Container

All candidates share one Docker image. The first notebook you run builds it and pushes it to ECR (about 10 to 20 minutes); the others reuse it through `docker/ecr_image_fullname.txt`.
To force a rebuild, delete that file.
"""),
        _code("""
%%bash
# Get the TF2 Object Detection API once, then copy its training and export scripts next to run_training.sh
if [ ! -d docker/models/research/object_detection ]; then
    git clone --depth 1 https://github.com/tensorflow/models.git docker/models
fi
cp docker/models/research/object_detection/exporter_main_v2.py source_dir
cp docker/models/research/object_detection/model_main_tf2.py source_dir
"""),
        _code("""
image_name = 'tf2-object-detection'
image_file = Path('docker') / 'ecr_image_fullname.txt'
if image_file.is_file():
    print(f"Reusing the image already pushed to ECR ({image_file})")
else:
    !bash -x ./docker/build_and_push.sh $image_name

container = image_file.read_text().strip()
print(container)
"""),
        _md(f"""
## Pre-trained model from the model zoo

The next cell builds `build/{key}/`. It holds the shared scripts, this candidate's `pipeline.config` and its COCO-trained checkpoint from the TF2 Model Zoo.
Only this folder is uploaded to the training job. The checkpoint is downloaded once and reused if you run the notebook again.
"""),
        _code("""
build_dir = h.stage_build_dir(CANDIDATE_KEY, Path('.'))
checkpoint_dir = h.ensure_checkpoint(candidate, build_dir, Path('/tmp/zoo'))
print(build_dir)
print(sorted(p.name for p in build_dir.iterdir()))
print(sorted(p.name for p in checkpoint_dir.iterdir()))
"""),
        _md("""
## Launch the training job

`run_training.sh` trains for `num_train_steps`, then evaluates the final checkpoint on the validation split (COCO metrics plus validation loss), then exports the model.
The evaluation metrics are printed at the end of the job logs below and also saved to the TensorBoard folder.
"""),
        _code("""
model_trainer = ModelTrainer(
    role=role,
    training_image=container,
    source_code=SourceCode(
        source_dir=f"{build_dir}/",
        entry_script='run_training.sh'
    ),
    hyperparameters={
        "model_dir": "/opt/training",
        "pipeline_config_path": "pipeline.config",
        "num_train_steps": str(num_train_steps),
        "sample_1_of_n_eval_examples": "1"
    },
    compute=Compute(
        instance_count=1,
        instance_type=TRAINING_INSTANCE
    ),
    base_job_name=base_job_name
).with_tensorboard_output_config(
    TensorBoardOutputConfig(
        s3_output_path=tensorboard_s3_prefix,
        local_path='/opt/training/'
    )
)

model_trainer.train(
    input_data_config=[InputData(channel_name=name, data_source=uri) for name, uri in inputs.items()]
)
"""),
        _md("""
## Record the Run

This cell saves the job name, model artifact and billed time to `results/<candidate>.json`. The summary and deploy notebooks read this file, so you don't need to copy S3 paths by hand.
"""),
        _code("""
run = h.latest_training_job(boto3.client('sagemaker'), base_job_name)
run.update(candidate=CANDIDATE_KEY, num_train_steps=num_train_steps,
           instance_type=TRAINING_INSTANCE, tensorboard_s3_prefix=tensorboard_s3_prefix)
if run["status"] != "Completed":
    raise RuntimeError(f"Training job {run['job_name']} ended with status {run['status']}")

results_path = Path('results') / f"{CANDIDATE_KEY}.json"
results_path.parent.mkdir(exist_ok=True)
results_path.write_text(json.dumps(run, indent=2) + "\\n")
print(json.dumps(run, indent=2))
"""),
    ]


def build_notebook(key: str, display_name: str) -> dict:
    cells = [{**cell, "id": f"cell-{i:02d}"} for i, cell in enumerate(build_cells(key, display_name))]
    return {
        "cells": cells,
        "metadata": {"kernelspec": KERNEL, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_path(key: str) -> Path:
    return TRAINING_DIR / f"1_train_model_{key}.ipynb"


def load_registry() -> list[dict]:
    return json.loads((TRAINING_DIR / "candidates.json").read_text())["candidates"]


def main() -> int:
    for entry in load_registry():
        path = notebook_path(entry["key"])
        path.write_text(json.dumps(build_notebook(entry["key"], entry["display_name"]), indent=1) + "\n")
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
