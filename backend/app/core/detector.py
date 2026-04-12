"""NudeNet-based nudity/exposure detector."""
import logging
from nudenet import NudeDetector
from .. import config

logger = logging.getLogger(__name__)

# Singleton detector instance
_detector = None

# Classes that always need mosaic (truly exposed)
CENSOR_CLASSES = {
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED", "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}

# Classes that need mosaic when high confidence (revealing clothing in live-action)
CENSOR_IF_HIGH_SCORE = {
    "FEMALE_BREAST_COVERED",  # deep-V, low-cut = still needs mosaic in live-action
}

HIGH_SCORE_THRESHOLD = 0.55  # COVERED class mosaic threshold (triggers VLM confirmation in pipeline)

# Per-class minimum thresholds (override global NUDENET_THRESHOLD)
# Lower threshold for genitalia/anus to avoid misses
CLASS_THRESHOLDS = {
    "FEMALE_GENITALIA_EXPOSED": 0.35,
    "MALE_GENITALIA_EXPOSED": 0.35,
    "ANUS_EXPOSED": 0.35,
    "FEMALE_BREAST_EXPOSED": 0.45,
    "BUTTOCKS_EXPOSED": 0.45,
}

# All violation classes (for reporting — only classes relevant to censorship)
VIOLATION_CLASSES = CENSOR_CLASSES | {
    "FEMALE_BREAST_COVERED",
    "FEMALE_GENITALIA_COVERED",
}

# Classes to silently ignore (face detections are useless for censorship)
_IGNORED_CLASSES = {
    "FACE_FEMALE", "FACE_MALE",
    "FEET_COVERED", "FEET_EXPOSED",
}


def _get_providers():
    """Return ONNX providers, preferring GPU if available."""
    try:
        import onnxruntime
        available = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in available:
            logger.info("Using CUDA GPU for NudeNet")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return None  # default


def get_detector() -> NudeDetector:
    global _detector
    if _detector is None:
        providers = _get_providers()
        _detector = NudeDetector(providers=providers)
        logger.info(f"NudeNet detector loaded (providers={providers})")
    return _detector


def detect_frame(image_path: str) -> list[dict]:
    """Detect nudity in a single frame image.
    Returns list of detections with class, score, box.
    """
    det = get_detector()
    results = det.detect(image_path)
    violations = []
    for r in results:
        cls = r["class"]
        score = r["score"]
        # Skip face/hair/feet classes — irrelevant for censorship
        if cls in _IGNORED_CLASSES:
            continue
        # Skip classes not in violation set (e.g. MALE_GENITALIA_COVERED, etc.)
        if cls not in VIOLATION_CLASSES:
            continue
        # Use per-class threshold if available, otherwise global threshold
        min_score = CLASS_THRESHOLDS.get(cls, config.NUDENET_THRESHOLD)
        if score < min_score:
            continue
        # EXPOSED classes always get mosaiced
        # COVERED classes get mosaiced only when high confidence (revealing clothing)
        need_mosaic = (cls in CENSOR_CLASSES) or \
                      (cls in CENSOR_IF_HIGH_SCORE and score >= HIGH_SCORE_THRESHOLD)
        # NudeNet returns [x, y, w, h], convert to [x1, y1, x2, y2]
        bx, by, bw, bh = r["box"]
        box = [int(bx), int(by), int(bx + bw), int(by + bh)]
        violations.append({
            "class": cls,
            "score": round(score, 3),
            "box": box,
            "need_mosaic": need_mosaic,
        })
    return violations


def censor_image(image_path: str, output_path: str) -> list[dict]:
    """Detect and censor a single image. Returns detections."""
    det = get_detector()
    det.censor(image_path, output_path=output_path, classes=list(CENSOR_CLASSES))
    return detect_frame(image_path)
