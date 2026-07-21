# Bottle OBB + Four-Keypoint Detector

This folder is a ready-to-use inference package. It detects bottles, returns an oriented bounding box for each bottle, predicts four semantic keypoints, and classifies each bottle as clear or occluded.

## Output definition

- Class `0`: `occluded`
- Class `1`: `clear`
- `C`: cap center
- `N`: neck center
- `L`: main label center
- `B`: base center

The model was trained so that clear bottles have all four keypoints. An occluded bottle may contain only the keypoints that are actually visible.

## Files

- `model/best.pt`: trained model weights
- `obb_pose/`: custom model code required by the weights
- `detector_api.py`: importable Python interface
- `example_api.py`: minimal integration example
- `predict.py`: command-line inference and visualization
- `inference_config.json`: class names, keypoint names, thresholds, and model checksum

Keep `model/` and `obb_pose/` with the Python files. The custom weights cannot be loaded correctly by a standard Ultralytics OBB class alone.

## Installation

Python 3.12, PyTorch 2.7.0, and Ultralytics 8.4.19 were used for verification.

On an NVIDIA computer, install the PyTorch build that matches the CUDA environment first. Then run:

```bash
python -m pip install -r requirements.txt
```

## Fast test

Run inference on one image or a directory:

```bash
python predict.py --source path/to/image_or_directory --output outputs --device 0
```

Device values:

- NVIDIA GPU: `--device 0`
- CPU: `--device cpu`
- Apple Silicon GPU: `--device mps`

The command creates:

- `outputs/predictions.json`: numeric prediction results
- `outputs/visualizations/*.png`: images with OBBs and keypoints

## Use in another Python project

Copy the complete `visible_keypoints_obb4kpt_model` folder into the other project. Import it from the parent directory:

```python
from visible_keypoints_obb4kpt_model import BottleOBB4KPTDetector

detector = BottleOBB4KPTDetector(device="0")
result = detector.predict_one("rgb_image.png")

for item in result["detections"]:
    print(item["class_name"], item["confidence"])
    print(item["obb"]["center_px"])
    print(item["obb"]["corners_px"])
    print(item["keypoints"])

best_clear = detector.best_clear_candidate(result)
if best_clear is not None:
    print("Selected clear bottle:", best_clear)
```

The same detector object should be created once and reused for multiple frames:

```python
detector = BottleOBB4KPTDetector(device="0")

for image in image_stream:
    result = detector.predict_one(image)
```

`predict_one` accepts a file path, a NumPy image, or another source supported by Ultralytics. NumPy images must use BGR channel order, as returned by OpenCV.

## Detection result format

Each detection contains:

```text
id
class_id
class_name
confidence
is_grasp_candidate
obb.center_px
obb.size_px
obb.angle_rad
obb.angle_deg
obb.corners_px
keypoints.C / N / L / B
```

Each keypoint contains `xy_px`, `visibility`, and `visible`. A keypoint should be used only when `visible` is `true`.

`best_clear_candidate` returns the highest-confidence class-1 detection. It returns `None` when no clear bottle passes the threshold.

## Default settings

- Input size: `960`
- Detection confidence: `0.528`
- Rotated NMS IoU: `0.70`
- Keypoint visibility: `0.50`

These defaults are already used by `BottleOBB4KPTDetector`. They can be overridden in its constructor when required.

## Model identity

The expected SHA-256 value of `model/best.pt` is:

```text
bce43c2bcab4be15f90af7eee90b84fcc6fba579eb28e4fc2a4b354ac6a132fc
```
