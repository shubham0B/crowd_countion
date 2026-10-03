import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
IMAGES_DIR  = os.path.join(BASE_DIR, "images")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR,  exist_ok=True)

YOLO_MODELS = {
    "Nano  — Fastest  (0–30 people)"      : os.path.join(MODELS_DIR, "yolov8n.pt"),
    "Small — Fast     (0–60 people)"      : os.path.join(MODELS_DIR, "yolov8s.pt"),
    "Medium — Balanced (30–100 people)"   : os.path.join(MODELS_DIR, "yolov8m.pt"),
    "Large — Accurate (100+ people)"      : os.path.join(MODELS_DIR, "yolov8l.pt"),
    "X-Large — Maximum (extreme crowds)"  : os.path.join(MODELS_DIR, "yolov8x.pt"),
}
DEFAULT_MODEL = "Large — Accurate (100+ people)"
PERSON_CLASS_ID = 0
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD  = 0.45

INFERENCE_RESOLUTIONS = {
    "Standard  640p  (fastest)" : 640,
    "HD        960p"            : 960,
    "Full HD   1280p (recommended)": 1280,
    "Ultra HD  1920p (slowest)" : 1920,
}
DEFAULT_RESOLUTION = "Full HD   1280p (recommended)"

MIN_PERSON_WIDTH    = 15
MIN_PERSON_HEIGHT   = 30
MAX_FRAME_FRACTION  = 0.70
MIN_ASPECT_RATIO    = 0.15
MAX_ASPECT_RATIO    = 1.60
LOW_CONF_THRESHOLD  = 0.15

FACE_MODEL_URL  = "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8l-face/model.pt"
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8l-face.pt")
FACE_TILE_SIZE       = 640
FACE_TILE_SIZE_SMALL = 320
FACE_MIN_SIZE_PX     = 6
FACE_MAX_SIZE_PX     = 180
FACE_NMS_IOU         = 0.30

COLOR_BODY = (100, 255, 60)
COLOR_FACE = (0,  150, 255)
FONT            = 0
FONT_SCALE      = 0.45
FONT_THICKNESS  = 1
BOX_THICKNESS   = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
LOG_LEVEL  = "INFO"
