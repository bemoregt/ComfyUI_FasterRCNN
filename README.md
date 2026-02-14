# ComfyUI Faster R-CNN Nodes

A custom node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that provides object detection using Faster R-CNN models from torchvision. Supports pretrained COCO models as well as custom-trained `.pth` checkpoints.

![이미지 스펙트럼 예시](https://github.com/bemoregt/ComfyUI_FasterRCNN/blob/main/ScrShot%204.png)

---

## Nodes

### 1. Faster R-CNN Model Loader

Loads a pretrained Faster R-CNN model from torchvision (COCO weights).

**Inputs**

| Name | Type | Description |
|---|---|---|
| `model_name` | Combo | Backbone architecture to load |

**Available models**

| Model | Backbone | Notes |
|---|---|---|
| `fasterrcnn_resnet50_fpn` | ResNet-50 + FPN | Standard baseline |
| `fasterrcnn_resnet50_fpn_v2` | ResNet-50 + FPN v2 | Improved accuracy |
| `fasterrcnn_mobilenet_v3_large_fpn` | MobileNetV3-Large | Faster, lower memory |
| `fasterrcnn_mobilenet_v3_large_320_fpn` | MobileNetV3-Large 320 | Fastest, smallest input |

**Outputs**

| Name | Type |
|---|---|
| `model` | `FRCNN_MODEL` |

---

### 2. Faster R-CNN Custom .pth Loader

Loads a custom-trained Faster R-CNN model from a local `.pth` file. Supports both `state_dict`-only and full-model checkpoints.

**Inputs**

| Name | Type | Description |
|---|---|---|
| `pth_path` | String | Absolute path to the `.pth` file |
| `load_mode` | Combo | `state_dict` or `full_model` (see below) |
| `backbone` | Combo | Architecture to rebuild *(state_dict mode only)* |
| `num_classes` | Int | Number of classes **including background** (COCO default: 91) |
| `class_names` | String | Comma-separated class labels, background excluded. e.g. `cat,dog,car` |

**Load modes**

| Mode | When to use | How the model was saved |
|---|---|---|
| `state_dict` | Most common | `torch.save(model.state_dict(), "model.pth")` |
| `full_model` | Entire object saved | `torch.save(model, "model.pth")` |

> **Checkpoint dict support:** If your file contains a dict with keys like `state_dict` or `model_state_dict` (e.g. from a training loop), the loader detects and unwraps them automatically.

**Outputs**

| Name | Type |
|---|---|
| `model` | `FRCNN_MODEL` |

---

### 3. Faster R-CNN Detector

Runs inference on an image using a loaded `FRCNN_MODEL` and returns an annotated image and bounding box data.

**Inputs**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | ComfyUI image tensor `[B, H, W, C]` |
| `model` | FRCNN_MODEL | — | Output from either loader node |
| `confidence_threshold` | Float | 0.5 | Minimum score to keep a detection |
| `draw_labels` | Boolean | True | Draw class name and score on each box |
| `line_width` | Int | 2 | Bounding box border thickness (px) |

**Outputs**

| Name | Type | Description |
|---|---|---|
| `annotated_image` | IMAGE | Input image with bounding boxes drawn |
| `bboxes_json` | STRING | Detections serialized as JSON |

**JSON output format**

```json
[
  {
    "label": "person",
    "label_id": 1,
    "score": 0.9821,
    "box": [x1, y1, x2, y2]
  },
  {
    "label": "car",
    "label_id": 3,
    "score": 0.7543,
    "box": [x1, y1, x2, y2]
  }
]
```

Coordinates are in pixel space (`xyxy` format). For a batch input the output is a list of lists.

---

## Installation

1. Clone or copy this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/ComfyUI_frcnn
```

2. Restart ComfyUI. The nodes will appear under the **detection/FasterRCNN** category.

### Dependencies

| Package | Notes |
|---|---|
| `torch` | Already required by ComfyUI |
| `torchvision` | Already required by ComfyUI |
| `Pillow` | Already required by ComfyUI |

No additional packages need to be installed.

---

## Device Selection

The nodes automatically select the best available device in this order:

```
CUDA  →  MPS (Apple Silicon)  →  CPU
```

On Apple Silicon Macs, MPS is used by default. If an unsupported MPS operation is encountered during inference, the node falls back to CPU for that batch automatically.

---

## Example Workflows

### Pretrained COCO detection

```
Load Image  →  Faster R-CNN Model Loader  →  Faster R-CNN Detector  →  Preview Image
                                                        ↓
                                                   bboxes_json  →  (Show Text / downstream nodes)
```

### Custom model inference

```
Load Image  →  Faster R-CNN Custom .pth Loader  →  Faster R-CNN Detector  →  Preview Image
               pth_path: /models/my_model.pth
               num_classes: 4
               class_names: cat,dog,car
```

---

## Training Tips (state_dict mode)

When saving your model during training, make sure the architecture and `num_classes` match what you use in the loader:

```python
# Save
torch.save(model.state_dict(), "my_model.pth")

# Or save with checkpoint metadata
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "checkpoint.pth")
```

In ComfyUI, set `load_mode = state_dict` and match `backbone` / `num_classes` to your training config.

---

## License

MIT
