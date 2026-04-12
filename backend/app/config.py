import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
VOLCANO_API_KEY = os.getenv("VOLCANO_API_KEY", "")
VOLCANO_MODEL = os.getenv("VOLCANO_MODEL", "doubao-seed-2-0-lite-260215")
VOLCANO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/outputs")
DB_URL = os.getenv("DB_URL", "sqlite:///data/video_censor.db")
EXTRACT_FPS = 2  # base frame extraction rate
SCENE_THRESHOLD = 30.0  # scene change detection threshold
NUDENET_THRESHOLD = 0.4  # NudeNet confidence threshold
NUDENET_REVIEW_THRESHOLD = 0.5  # below this, EXPOSED detections marked as need_review
NUDENET_VLM_CONFIRM_THRESHOLD = 0.55  # EXPOSED below this score -> VLM confirmation before mosaic
VLM_SAMPLE_INTERVAL = 2  # seconds between VLM audit samples (lower = more accurate, higher cost)
MOSAIC_BLOCK_SIZE = 15  # mosaic pixel block size

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
