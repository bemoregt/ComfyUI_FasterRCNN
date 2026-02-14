import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

# COCO 클래스 이름 (91개)
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

MODEL_OPTIONS = {
    "fasterrcnn_resnet50_fpn": (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights.DEFAULT),
    "fasterrcnn_resnet50_fpn_v2": (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
    "fasterrcnn_mobilenet_v3_large_fpn": (fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
    "fasterrcnn_mobilenet_v3_large_320_fpn": (fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT),
}

# 클래스별 색상 (재현 가능한 랜덤)
def _class_color(class_id: int):
    rng = np.random.default_rng(class_id * 7919)
    r, g, b = rng.integers(80, 230, size=3).tolist()
    return (r, g, b)


class FasterRCNNModelLoader:
    """Faster R-CNN 사전학습 모델을 로드합니다."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(MODEL_OPTIONS.keys()),),
            }
        }

    RETURN_TYPES = ("FRCNN_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "detection/FasterRCNN"

    def load_model(self, model_name: str):
        builder, weights = MODEL_OPTIONS[model_name]
        model = builder(weights=weights)
        model.eval()

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model = model.to(device)

        return ({"model": model, "device": device},)


class FasterRCNNDetector:
    """이미지에서 Faster R-CNN 추론을 수행합니다."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),               # ComfyUI IMAGE: [B, H, W, C] float32 0~1
                "model": ("FRCNN_MODEL",),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "draw_labels": ("BOOLEAN", {"default": True}),
                "line_width": ("INT", {"default": 2, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("annotated_image", "bboxes_json")
    FUNCTION = "detect"
    CATEGORY = "detection/FasterRCNN"

    def detect(self, image, model, confidence_threshold, draw_labels, line_width):
        net = model["model"]
        device = model["device"]
        # 커스텀 로더는 class_names를 포함할 수 있음, 없으면 COCO 기본값 사용
        class_names = model.get("class_names") or COCO_CLASSES

        # image: [B, H, W, C] float32 → 배치 처리
        batch_size = image.shape[0]
        annotated_list = []
        all_results = []

        for i in range(batch_size):
            # [H, W, C] → [C, H, W] tensor
            img_tensor = image[i].permute(2, 0, 1).to(device)

            with torch.no_grad():
                try:
                    outputs = net([img_tensor])
                except (RuntimeError, NotImplementedError):
                    # MPS 미지원 연산 발생 시 CPU로 fallback
                    net_cpu = net.to("cpu")
                    outputs = net_cpu([img_tensor.to("cpu")])
                    net.to(device)  # 모델은 다시 원래 device로

            output = outputs[0]
            boxes   = output["boxes"].cpu().numpy()    # [N, 4] xyxy
            scores  = output["scores"].cpu().numpy()   # [N]
            labels  = output["labels"].cpu().numpy()   # [N]

            # confidence threshold 필터링
            keep = scores >= confidence_threshold
            boxes  = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # PIL 이미지로 변환하여 박스 그리기
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            draw = ImageDraw.Draw(pil_img)

            frame_results = []
            for box, score, label_id in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                label_id = int(label_id)
                score = float(score)
                label_name = class_names[label_id] if label_id < len(class_names) else str(label_id)
                color = _class_color(label_id)

                # 바운딩박스 사각형
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

                if draw_labels:
                    text = f"{label_name} {score:.2f}"
                    # 텍스트 배경
                    try:
                        font = ImageFont.load_default(size=14)
                        bbox_text = draw.textbbox((x1, y1), text, font=font)
                    except TypeError:
                        # 구버전 Pillow
                        font = ImageFont.load_default()
                        tw, th = draw.textsize(text, font=font)
                        bbox_text = (x1, y1, x1 + tw, y1 + th)

                    pad = 2
                    draw.rectangle(
                        [bbox_text[0] - pad, bbox_text[1] - pad,
                         bbox_text[2] + pad, bbox_text[3] + pad],
                        fill=color,
                    )
                    draw.text((x1, y1), text, fill=(255, 255, 255), font=font)

                frame_results.append({
                    "label": label_name,
                    "label_id": label_id,
                    "score": round(score, 4),
                    "box": [round(v, 2) for v in [x1, y1, x2, y2]],
                })

            all_results.append(frame_results)

            # PIL → numpy → tensor [H, W, C] float32
            annotated_np = np.array(pil_img).astype(np.float32) / 255.0
            annotated_list.append(torch.from_numpy(annotated_np))

        # 배치 텐서 [B, H, W, C]
        annotated_batch = torch.stack(annotated_list, dim=0)

        # 단일 이미지면 results를 flat하게, 배치면 리스트로
        json_out = json.dumps(all_results[0] if batch_size == 1 else all_results, ensure_ascii=False, indent=2)

        return (annotated_batch, json_out)


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_model_skeleton(backbone: str, num_classes: int):
    """아키텍처만 구성하고 가중치는 비어있는 모델 반환."""
    builders = {
        "fasterrcnn_resnet50_fpn": fasterrcnn_resnet50_fpn,
        "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
        "fasterrcnn_mobilenet_v3_large_fpn": fasterrcnn_mobilenet_v3_large_fpn,
        "fasterrcnn_mobilenet_v3_large_320_fpn": fasterrcnn_mobilenet_v3_large_320_fpn,
    }
    # num_classes가 COCO(91)와 다르면 head를 교체해야 하므로 weights=None으로 생성
    model = builders[backbone](weights=None, num_classes=num_classes)
    return model


class FasterRCNNCustomModelLoader:
    """커스텀 학습된 .pth 파일로 Faster R-CNN 모델을 로드합니다.

    두 가지 형식을 지원합니다:
    - state_dict 형식: torch.save(model.state_dict(), "model.pth")
    - 전체 모델 형식: torch.save(model, "model.pth")
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pth_path": ("STRING", {
                    "default": "/path/to/model.pth",
                    "multiline": False,
                }),
                "load_mode": (["state_dict", "full_model"],),
            },
            "optional": {
                # state_dict 모드일 때만 사용
                "backbone": (list(MODEL_OPTIONS.keys()), {"default": "fasterrcnn_resnet50_fpn"}),
                "num_classes": ("INT", {
                    "default": 91,
                    "min": 2,
                    "max": 1000,
                    "tooltip": "background 포함 클래스 수 (COCO=91)",
                }),
                # 커스텀 클래스 이름 (쉼표 구분, 비워두면 label_id로 표시)
                "class_names": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "쉼표로 구분된 클래스 이름 (background 제외). 예: cat,dog,car",
                }),
            },
        }

    RETURN_TYPES = ("FRCNN_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "detection/FasterRCNN"

    def load_model(self, pth_path: str, load_mode: str,
                   backbone: str = "fasterrcnn_resnet50_fpn",
                   num_classes: int = 91,
                   class_names: str = ""):

        device = _get_device()
        map_location = torch.device(device)

        if load_mode == "full_model":
            # 모델 객체 전체가 저장된 경우
            model = torch.load(pth_path, map_location=map_location, weights_only=False)
        else:
            # state_dict만 저장된 경우 → 아키텍처 재구성 후 가중치 로드
            model = _build_model_skeleton(backbone, num_classes)
            state = torch.load(pth_path, map_location=map_location, weights_only=True)

            # 흔한 래핑 키 처리 ("model", "state_dict", "model_state_dict")
            if isinstance(state, dict):
                for key in ("model", "state_dict", "model_state_dict"):
                    if key in state:
                        state = state[key]
                        break

            model.load_state_dict(state)

        model.eval()
        model = model.to(device)

        # 커스텀 클래스 이름 파싱
        if class_names.strip():
            names = ["__background__"] + [c.strip() for c in class_names.split(",")]
        else:
            names = None  # None이면 Detector에서 label_id 그대로 사용

        return ({"model": model, "device": device, "class_names": names},)


NODE_CLASS_MAPPINGS = {
    "FasterRCNNModelLoader": FasterRCNNModelLoader,
    "FasterRCNNCustomModelLoader": FasterRCNNCustomModelLoader,
    "FasterRCNNDetector": FasterRCNNDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FasterRCNNModelLoader": "Faster R-CNN Model Loader",
    "FasterRCNNCustomModelLoader": "Faster R-CNN Custom .pth Loader",
    "FasterRCNNDetector": "Faster R-CNN Detector",
}
