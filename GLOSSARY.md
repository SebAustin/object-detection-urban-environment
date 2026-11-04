# Object Detection in an Urban Environment

Fine-tuning pretrained object detectors on Waymo camera images to find vehicles, pedestrians and cyclists, comparing them, and deploying the best one to annotate a driving video.

## Detection targets

**Class**:
One of the three object categories the detector must find: vehicle, pedestrian, cyclist.
_Avoid_: Label, category, type

**Label map**:
The dataset's fixed mapping from class id to class name (1 = vehicle, 2 = pedestrian, 4 = cyclist; id 3 is unused).
_Avoid_: Class list

## Models and experiments

**Pretrained model**:
A detector from the TF2 Detection Model Zoo, with COCO-trained weights, that serves as the starting point for fine-tuning.
_Avoid_: Base model, backbone (a backbone is only one part of a model)

**Candidate**:
A pretrained model that is being compared for deployment. This project has three: EfficientDet D1 (the default), SSD MobileNet V2 FPNLite 640, and SSD ResNet50 V1 FPN 640.
_Avoid_: Option, variant

**Run**:
One training job that fine-tunes one candidate with one configuration and evaluates it on the validation split.
_Avoid_: Experiment (that word is for the whole comparison), trial

**Experiment**:
The full set of runs being compared, together with the conclusion drawn from them.
_Avoid_: Study

**Best model**:
The candidate whose run has the highest mAP, and the only one that gets deployed.
_Avoid_: Winner, final model

## Metrics

**mAP**:
COCO mean average precision averaged over IoU thresholds 0.50 to 0.95, measured on the validation split. It is the headline accuracy number.
_Avoid_: Accuracy (when it means something else), score

**mAP@0.5**:
COCO mean average precision at a single IoU threshold of 0.50, reported alongside mAP.

**Training loss / Validation loss**:
The detector's combined classification, localization and regularization loss, measured on the training split and on the validation split, respectively.
_Avoid_: Error, cost

## Deployment

**Endpoint**:
The hosted inference service that runs the best model and returns detections for each image sent to it.
_Avoid_: API, server

**Output video**:
The test drive's frames with the best model's detections drawn on them, assembled into a video.
_Avoid_: Demo, animation
