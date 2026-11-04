"""Tests for reading TensorBoard event files written by the TF2 OD API.

Needs the `tensorboard` package (pure Python, no TensorFlow). Run with:
    uv run --with pytest --with protobuf --with tensorboard --with pandas pytest tests
"""

from pathlib import Path

import pytest

pytest.importorskip("tensorboard")
pd = pytest.importorskip("pandas")

from tensorboard.compat.proto import event_pb2, summary_pb2  # noqa: E402
from tensorboard.summary.writer.event_file_writer import EventFileWriter  # noqa: E402
from tensorboard.util import tensor_util  # noqa: E402

import tb_helpers as tb  # noqa: E402


def write_events(log_dir: Path, points: list[tuple[int, str, float]], as_tensor: bool = True) -> None:
    writer = EventFileWriter(str(log_dir))
    for step, tag, value in points:
        if as_tensor:  # TF2 tf.summary.scalar writes tensors
            val = summary_pb2.Summary.Value(tag=tag, tensor=tensor_util.make_tensor_proto(value))
        else:  # TF1-style simple values
            val = summary_pb2.Summary.Value(tag=tag, simple_value=value)
        writer.add_event(event_pb2.Event(step=step, summary=summary_pb2.Summary(value=[val])))
    writer.close()


@pytest.mark.parametrize("as_tensor", [True, False])
def test_load_scalars_reads_tensor_and_simple_values(tmp_path, as_tensor):
    write_events(tmp_path, [(100, "Loss/total_loss", 1.5), (200, "Loss/total_loss", 1.0)], as_tensor)
    df = tb.load_scalars(tmp_path)
    assert list(df.columns) == ["step", "tag", "value"]
    assert df["step"].tolist() == [100, 200]
    assert df["value"].tolist() == pytest.approx([1.5, 1.0])


def test_load_scalars_returns_empty_frame_for_empty_dir(tmp_path):
    assert tb.load_scalars(tmp_path).empty


@pytest.mark.parametrize(
    ("key", "split"),
    [
        ("logs/efficientdet_d1/job-1/tensorboard-output/train/events.out.tfevents.1.host", "train"),
        ("logs/efficientdet_d1/job-1/tensorboard-output/eval/events.out.tfevents.2.host", "eval"),
        ("logs/efficientdet_d1/job-1/tensorboard-output/checkpoint", None),
        ("logs/efficientdet_d1/job-1/tensorboard-output/other/events.out.tfevents.3", None),
    ],
)
def test_split_of_classifies_event_files(key, split):
    assert tb.split_of(key) == split


def test_summarize_run_uses_tail_mean_for_training_and_last_eval():
    train = pd.DataFrame(
        {"step": [100, 1000, 1900, 2000], "tag": ["Loss/total_loss"] * 4, "value": [9.0, 5.0, 1.0, 0.8]}
    )
    evaluation = pd.DataFrame(
        {
            "step": [2000, 2000, 2000],
            "tag": ["DetectionBoxes_Precision/mAP", "DetectionBoxes_Precision/mAP@.50IOU", "Loss/total_loss"],
            "value": [0.12, 0.25, 1.4],
        }
    )
    summary = tb.summarize_run(train, evaluation)
    assert summary["train Loss/total_loss"] == pytest.approx(0.9)  # mean of steps >= 90% of 2000
    assert summary["val Loss/total_loss"] == pytest.approx(1.4)
    assert summary["mAP"] == pytest.approx(0.12)
    assert summary["mAP@0.5"] == pytest.approx(0.25)
    assert summary["mAP small"] is None  # missing metrics are explicit, not silently zero


# ---------- best-model rule (Q16) ----------

def test_pick_best_takes_highest_map():
    rows = {"a": {"mAP": 0.10, "mAP small": 0.05}, "b": {"mAP": 0.20, "mAP small": 0.01}}
    assert tb.pick_best(rows) == "b"


def test_pick_best_breaks_near_ties_on_small_object_map():
    rows = {"a": {"mAP": 0.205, "mAP small": 0.03}, "b": {"mAP": 0.200, "mAP small": 0.08}}
    assert tb.pick_best(rows) == "b"


def test_pick_best_ignores_runs_without_map():
    rows = {"a": {"mAP": None, "mAP small": None}, "b": {"mAP": 0.01, "mAP small": None}}
    assert tb.pick_best(rows) == "b"


def test_pick_best_fails_without_any_evaluated_run():
    with pytest.raises(ValueError, match="No evaluated run"):
        tb.pick_best({"a": {"mAP": None}})
