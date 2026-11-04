"""The experiment's shared recipe, enforced on every candidate's pipeline.config.

See docs/adr/0001 (num_classes 4) and docs/adr/0002 (shared learning-rate rule).
"""

import pytest
from google.protobuf import text_format

from conftest import TRAINING_DIR

EXPECTED_KEYS = {"efficientdet_d1", "ssd_mobilenet_v2_fpnlite_640", "ssd_resnet50_v1_fpn_640"}

NUM_CLASSES = 4  # label map ids 1, 2, 4; id 3 unused (ADR 0001)
TRAIN_STEPS = 2000
WARMUP_STEPS = 200
BASE_LR = 0.02
WARMUP_LR = 0.002
BATCH_SIZE = 8
AUGMENTATIONS = ["random_horizontal_flip", "random_scale_crop_and_pad_to_square"]

TRAIN_INPUT = "/opt/ml/input/data/train/*.tfrecord"
VAL_INPUT = "/opt/ml/input/data/val/*.tfrecord"
TRAIN_LABEL_MAP = "/opt/ml/input/data/train/label_map.pbtxt"
VAL_LABEL_MAP = "/opt/ml/input/data/val/label_map.pbtxt"


def config_path(key: str):
    return TRAINING_DIR / "configs" / key / "pipeline.config"


def load(pipeline_pb2, key: str):
    msg = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Parse(config_path(key).read_text(), msg)
    return msg


def test_registry_lists_the_three_agreed_candidates(registry):
    keys = {c["key"] for c in registry["candidates"]}
    assert keys == EXPECTED_KEYS


@pytest.fixture(params=sorted(EXPECTED_KEYS))
def candidate(request, registry):
    return next(c for c in registry["candidates"] if c["key"] == request.param)


def test_config_parses_against_od_api_schema(pipeline_pb2, candidate):
    assert config_path(candidate["key"]).is_file()
    load(pipeline_pb2, candidate["key"])  # raises ParseError on any unknown field


def test_architecture_matches_registry(pipeline_pb2, candidate):
    cfg = load(pipeline_pb2, candidate["key"])
    assert cfg.model.WhichOneof("model") == "ssd"
    assert cfg.model.ssd.feature_extractor.type == candidate["feature_extractor"]


def test_num_classes_covers_sparse_label_map(pipeline_pb2, candidate):
    assert load(pipeline_pb2, candidate["key"]).model.ssd.num_classes == NUM_CLASSES


def test_learning_rate_schedule_follows_shared_rule(pipeline_pb2, candidate):
    train = load(pipeline_pb2, candidate["key"]).train_config
    assert train.num_steps == TRAIN_STEPS
    assert train.batch_size == BATCH_SIZE
    assert train.use_bfloat16 is False
    opt = train.optimizer.momentum_optimizer
    assert opt.momentum_optimizer_value == pytest.approx(0.9)
    lr = opt.learning_rate.cosine_decay_learning_rate
    assert lr.total_steps == TRAIN_STEPS
    assert lr.warmup_steps == WARMUP_STEPS
    assert lr.learning_rate_base == pytest.approx(BASE_LR)
    assert lr.warmup_learning_rate == pytest.approx(WARMUP_LR)


def test_fine_tunes_from_detection_checkpoint(pipeline_pb2, candidate):
    train = load(pipeline_pb2, candidate["key"]).train_config
    assert train.fine_tune_checkpoint == "checkpoint/ckpt-0"
    assert train.fine_tune_checkpoint_type == "detection"


def test_augmentation_is_identical_across_candidates(pipeline_pb2, candidate):
    train = load(pipeline_pb2, candidate["key"]).train_config
    names = [opt.WhichOneof("preprocessing_step") for opt in train.data_augmentation_options]
    assert names == AUGMENTATIONS


def test_inputs_point_at_sagemaker_channels(pipeline_pb2, candidate):
    cfg = load(pipeline_pb2, candidate["key"])
    assert cfg.train_input_reader.label_map_path == TRAIN_LABEL_MAP
    assert list(cfg.train_input_reader.tf_record_input_reader.input_path) == [TRAIN_INPUT]
    assert cfg.eval_input_reader[0].label_map_path == VAL_LABEL_MAP
    assert list(cfg.eval_input_reader[0].tf_record_input_reader.input_path) == [VAL_INPUT]
    assert cfg.eval_config.metrics_set == ["coco_detection_metrics"]


def test_train_config_is_byte_identical_across_candidates(pipeline_pb2):
    blocks = {k: text_format.MessageToString(load(pipeline_pb2, k).train_config) for k in EXPECTED_KEYS}
    assert len(set(blocks.values())) == 1, "train_config must be shared by all candidates"
