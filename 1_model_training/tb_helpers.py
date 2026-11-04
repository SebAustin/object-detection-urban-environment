"""Read the TF2 OD API TensorBoard logs of each Run (from S3) into tidy pandas frames.

Used by 3_experiment_summary.ipynb. Needs `tensorboard` (pure Python) and `pandas`, not TensorFlow.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Optional

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator as ea
from tensorboard.util import tensor_util

EVENTS_MARKER = "tfevents"
SPLITS = ("train", "eval")
TRAIN_TAIL_FRACTION = 0.9  # training loss = mean over the last 10% of steps (smooths batch noise)
NEAR_TIE_MAP = 0.01  # within this mAP of the leader, prefer better small-object mAP (pedestrians, cyclists)

LOSS_TAGS = ("Loss/classification_loss", "Loss/localization_loss", "Loss/regularization_loss", "Loss/total_loss")
METRIC_TAGS = {
    "mAP": "DetectionBoxes_Precision/mAP",
    "mAP@0.5": "DetectionBoxes_Precision/mAP@.50IOU",
    "mAP small": "DetectionBoxes_Precision/mAP (small)",
    "mAP medium": "DetectionBoxes_Precision/mAP (medium)",
    "mAP large": "DetectionBoxes_Precision/mAP (large)",
    "AR@100": "DetectionBoxes_Recall/AR@100",
}


def split_of(s3_key: str) -> Optional[str]:
    """'train' or 'eval' for an OD API event file, else None."""
    path = PurePosixPath(s3_key)
    if EVENTS_MARKER not in path.name:
        return None
    parent = path.parent.name
    return parent if parent in SPLITS else None


def download_event_files(s3_client: Any, bucket: str, prefix: str, dest: Path) -> dict[str, Path]:
    """Download every train/eval event file under s3://bucket/prefix into dest/<split>/. Returns split -> dir."""
    dirs: dict[str, Path] = {}
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            split = split_of(obj["Key"])
            if split is None:
                continue
            target_dir = Path(dest) / split
            target_dir.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(bucket, obj["Key"], str(target_dir / PurePosixPath(obj["Key"]).name))
            dirs[split] = target_dir
    return dirs


def _value(event: Any) -> float:
    return float(tensor_util.make_ndarray(event.tensor_proto))


def load_scalars(log_dir: Path) -> pd.DataFrame:
    """All scalar summaries in a directory of event files as columns step, tag, value (sorted)."""
    acc = ea.EventAccumulator(str(log_dir), size_guidance={ea.SCALARS: 0, ea.TENSORS: 0})
    acc.Reload()
    rows = []
    tags = acc.Tags()
    for tag in tags.get("scalars", []):
        rows += [(e.step, tag, float(e.value)) for e in acc.Scalars(tag)]
    for tag in tags.get("tensors", []):
        for e in acc.Tensors(tag):
            if e.tensor_proto.tensor_shape.dim:  # skip non-scalar tensors (images, text)
                continue
            rows.append((e.step, tag, _value(e)))
    df = pd.DataFrame(rows, columns=["step", "tag", "value"])
    return df.sort_values(["tag", "step"], ignore_index=True)


def _tail_mean(df: pd.DataFrame, tag: str) -> Optional[float]:
    series = df[df["tag"] == tag]
    if series.empty:
        return None
    tail = series[series["step"] >= TRAIN_TAIL_FRACTION * series["step"].max()]
    return float(tail["value"].mean())


def _last(df: pd.DataFrame, tag: str) -> Optional[float]:
    series = df[df["tag"] == tag]
    return None if series.empty else float(series.sort_values("step")["value"].iloc[-1])


def summarize_run(train: pd.DataFrame, evaluation: pd.DataFrame) -> dict[str, Optional[float]]:
    """One row of the comparison table: final training/validation losses and COCO metrics."""
    summary: dict[str, Optional[float]] = {}
    for tag in LOSS_TAGS:
        summary[f"train {tag}"] = _tail_mean(train, tag)
        summary[f"val {tag}"] = _last(evaluation, tag)
    for name, tag in METRIC_TAGS.items():
        summary[name] = _last(evaluation, tag)
    return summary


def pick_best(summaries: dict[str, dict[str, Optional[float]]]) -> str:
    """Best model rule: highest mAP; among runs within NEAR_TIE_MAP of the leader, highest small-object mAP."""
    scored = {k: s for k, s in summaries.items() if s.get("mAP") is not None}
    if not scored:
        raise ValueError("No evaluated run has an mAP value")
    leader = max(s["mAP"] for s in scored.values())
    contenders = {k: s for k, s in scored.items() if leader - s["mAP"] <= NEAR_TIE_MAP}
    return max(contenders, key=lambda k: (contenders[k].get("mAP small") or 0.0, contenders[k]["mAP"]))
