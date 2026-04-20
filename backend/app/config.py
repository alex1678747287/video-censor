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
NUDENET_EXPOSED_DIRECT_SCORE = max(
    NUDENET_VLM_CONFIRM_THRESHOLD,
    float(os.getenv("NUDENET_EXPOSED_DIRECT_SCORE", "0.78"))
)
NUDENET_COVERED_DIRECT_SCORE = max(
    0.65,
    float(os.getenv("NUDENET_COVERED_DIRECT_SCORE", "0.72"))
)
NUDENET_TEMPORAL_WINDOW_SECONDS = max(
    0.2,
    float(os.getenv("NUDENET_TEMPORAL_WINDOW_SECONDS", "1.2"))
)
VLM_VISUAL_TEMPORAL_WINDOW_SECONDS = max(
    0.5,
    float(os.getenv("VLM_VISUAL_TEMPORAL_WINDOW_SECONDS", "2.5"))
)
VLM_VISUAL_HIGH_DIRECT_MAX_BOX_AREA = max(
    25.0,
    float(os.getenv("VLM_VISUAL_HIGH_DIRECT_MAX_BOX_AREA", "324"))
)
VLM_BLOOD_RED_RATIO_MIN = min(
    1.0,
    max(0.01, float(os.getenv("VLM_BLOOD_RED_RATIO_MIN", "0.08")))
)
VLM_BLOOD_DARK_RED_RATIO_MIN = min(
    1.0,
    max(0.005, float(os.getenv("VLM_BLOOD_DARK_RED_RATIO_MIN", "0.02")))
)
VLM_BLOOD_STRONG_RED_RATIO_MIN = min(
    1.0,
    max(0.02, float(os.getenv("VLM_BLOOD_STRONG_RED_RATIO_MIN", "0.16")))
)
VLM_BLOOD_SPOT_RATIO_MIN = min(
    1.0,
    max(0.003, float(os.getenv("VLM_BLOOD_SPOT_RATIO_MIN", "0.012")))
)
VLM_BLOOD_SPOT_MIN_PIXELS = max(
    8, int(os.getenv("VLM_BLOOD_SPOT_MIN_PIXELS", "24"))
)
VLM_SAMPLE_INTERVAL = 2  # seconds between VLM audit samples (lower = more accurate, higher cost)
CLOUD_EXECUTION_PROFILE = os.getenv("CLOUD_EXECUTION_PROFILE", "local_first").strip().lower()
LOCAL_GPU_PROFILE = os.getenv("LOCAL_GPU_PROFILE", "rtx4060_8g").strip().lower()
VLM_LOCAL_FIRST_BASELINE_INTERVAL_SECONDS = max(
    float(VLM_SAMPLE_INTERVAL),
    float(os.getenv("VLM_LOCAL_FIRST_BASELINE_INTERVAL_SECONDS", "5.0")),
)
HIGHLIGHT_SAMPLE_INTERVAL_SECONDS = max(
    float(VLM_SAMPLE_INTERVAL),
    float(os.getenv("HIGHLIGHT_SAMPLE_INTERVAL_SECONDS", "4.0")),
)
VLM_LOCAL_FIRST_RISK_WINDOW_SECONDS = max(
    0.0,
    float(os.getenv("VLM_LOCAL_FIRST_RISK_WINDOW_SECONDS", "1.0")),
)
VLM_LOCAL_FIRST_BLOOD_FRAME_SCORE_MIN = max(
    0.0,
    float(os.getenv("VLM_LOCAL_FIRST_BLOOD_FRAME_SCORE_MIN", "0.16")),
)
VLM_LOCAL_FIRST_MAX_BLOOD_CANDIDATES = max(
    0,
    int(os.getenv("VLM_LOCAL_FIRST_MAX_BLOOD_CANDIDATES", "24")),
)
ARK_PRICE_INPUT_PER_MTOKEN = max(
    0.0,
    float(os.getenv("ARK_PRICE_INPUT_PER_MTOKEN", "0.6")),
)
ARK_PRICE_OUTPUT_PER_MTOKEN = max(
    0.0,
    float(os.getenv("ARK_PRICE_OUTPUT_PER_MTOKEN", "3.6")),
)
EST_VLM_FRAME_INPUT_TEXT_TOKENS = max(
    0,
    int(os.getenv("EST_VLM_FRAME_INPUT_TEXT_TOKENS", "900")),
)
EST_VLM_FRAME_IMAGE_INPUT_TOKENS = max(
    0,
    int(os.getenv("EST_VLM_FRAME_IMAGE_INPUT_TOKENS", "650")),
)
EST_VLM_FRAME_OUTPUT_TOKENS = max(
    0,
    int(os.getenv("EST_VLM_FRAME_OUTPUT_TOKENS", "180")),
)
EST_VLM_CONFIRM_INPUT_TEXT_TOKENS = max(
    0,
    int(os.getenv("EST_VLM_CONFIRM_INPUT_TEXT_TOKENS", "420")),
)
EST_VLM_CONFIRM_OUTPUT_TOKENS = max(
    0,
    int(os.getenv("EST_VLM_CONFIRM_OUTPUT_TOKENS", "60")),
)
EST_VLM_TEXT_BATCH_BASE_INPUT_TOKENS = max(
    0,
    int(os.getenv("EST_VLM_TEXT_BATCH_BASE_INPUT_TOKENS", "650")),
)
EST_VLM_TEXT_BATCH_OUTPUT_TOKENS = max(
    0,
    int(os.getenv("EST_VLM_TEXT_BATCH_OUTPUT_TOKENS", "160")),
)
EST_VLM_TEXT_CHAR_TO_TOKEN_RATIO = max(
    0.1,
    float(os.getenv("EST_VLM_TEXT_CHAR_TO_TOKEN_RATIO", "1.0")),
)
EST_VLM_HIGHLIGHT_BATCH_INPUT_TEXT_TOKENS = max(
    0,
    int(os.getenv("EST_VLM_HIGHLIGHT_BATCH_INPUT_TEXT_TOKENS", "700")),
)
EST_VLM_HIGHLIGHT_IMAGE_INPUT_TOKENS = max(
    0,
    int(os.getenv("EST_VLM_HIGHLIGHT_IMAGE_INPUT_TOKENS", "500")),
)
EST_VLM_HIGHLIGHT_BATCH_OUTPUT_TOKENS = max(
    0,
    int(os.getenv("EST_VLM_HIGHLIGHT_BATCH_OUTPUT_TOKENS", "260")),
)
MOSAIC_BLOCK_SIZE = 15  # mosaic pixel block size

# Drama pipeline
DRAMA_DIR = os.getenv("DRAMA_DIR", "data/dramas")
DRAMA_SPEED_DEFAULT = 1.2
DRAMA_MAX_DURATION = 900  # 15 minutes
DRAMA_EPISODE_TAIL_TRIM_SECONDS = max(
    0.0, float(os.getenv("DRAMA_EPISODE_TAIL_TRIM_SECONDS", "2.5"))
)
DRAMA_HOOK_DURATION = 3  # highlight hook clip seconds
DRAMA_HOOK_PRE_ROLL_SECONDS = max(
    0.0, float(os.getenv("DRAMA_HOOK_PRE_ROLL_SECONDS", "0.35"))
)
DRAMA_HOOK_END_GUARD_SECONDS = max(
    0.5, float(os.getenv("DRAMA_HOOK_END_GUARD_SECONDS", "1.0"))
)
DRAMA_HOOK_INTRO_SKIP_SECONDS = max(
    0.0, float(os.getenv("DRAMA_HOOK_INTRO_SKIP_SECONDS", "1.0"))
)
FONT_PATH = os.getenv("FONT_PATH", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
FONT_PATH_REGULAR = os.getenv("FONT_PATH_REGULAR", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
VLM_FRAME_BATCH_SIZE = max(1, int(os.getenv("VLM_FRAME_BATCH_SIZE", "40")))
PROCESSING_SLOTS = max(1, int(os.getenv("PROCESSING_SLOTS", "2")))
ETA_DEFAULT_SINGLE_SECONDS = max(60, int(os.getenv("ETA_DEFAULT_SINGLE_SECONDS", "240")))
ETA_DEFAULT_DRAMA_SECONDS = max(180, int(os.getenv("ETA_DEFAULT_DRAMA_SECONDS", "1200")))
ETA_MIN_REMAINING_SECONDS = max(10, int(os.getenv("ETA_MIN_REMAINING_SECONDS", "30")))
CELERY_VISIBILITY_TIMEOUT_SECONDS = max(
    3600, int(os.getenv("CELERY_VISIBILITY_TIMEOUT_SECONDS", "43200"))
)
TASK_HEARTBEAT_INTERVAL_SECONDS = max(
    10, int(os.getenv("TASK_HEARTBEAT_INTERVAL_SECONDS", "30"))
)
ORPHAN_REQUEUE_GRACE_SECONDS = max(
    60, int(os.getenv("ORPHAN_REQUEUE_GRACE_SECONDS", "120"))
)
MAX_AUTO_REQUEUE_ATTEMPTS = max(
    0, int(os.getenv("MAX_AUTO_REQUEUE_ATTEMPTS", "2"))
)
STALE_TASK_TIMEOUT_SECONDS = max(1800, int(os.getenv("STALE_TASK_TIMEOUT_SECONDS", "14400")))
STALE_DRAMA_TIMEOUT_SECONDS = max(3600, int(os.getenv("STALE_DRAMA_TIMEOUT_SECONDS", "28800")))

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DRAMA_DIR, exist_ok=True)
