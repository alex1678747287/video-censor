"""OCR-based subtitle text violation detector using RapidOCR + VLM."""
import logging
import re
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ocr_engine = None

# Fast-path keywords: instant violation, no VLM needed
# Only include unambiguous profanity here; context-dependent words go to VLM
INSTANT_KEYWORDS = [
    "捞女", "绿茶婊", "绿茶", "小三", "婊子", "婊", "贱人", "贱货",
    "荡妇", "骚货", "骚逼", "傻逼", "牛逼",
    "操你", "草你", "他妈的", "我操", "卧槽",
    "约炮", "开房", "潜规则", "包养", "二奶",
    "去死", "弄死", "杀了你", "打死",
]
_INSTANT_PATTERN = re.compile("|".join(re.escape(k) for k in INSTANT_KEYWORDS))

# Context-dependent keywords: only violating when NOT preceded by possessive pronouns
# e.g. "妈的" is profanity alone, but "我妈的姿势" is normal speech
CONTEXT_KEYWORDS = {
    "妈的": re.compile(r"(?<![我你他她它们的])妈的"),  # "妈的" not preceded by pronouns
}


# VLM text audit prompt: judge if subtitle text is violating (aligned with platform standards)
TEXT_AUDIT_PROMPT = """你是微短剧字幕审核专家，依据平台审核标准判定字幕违规。判断以下字幕文字是否包含违规内容。

违规类型（依据平台标准）：
- 显著使用脏话：涉性粗俗语言、密集出现激动情绪口语词
- 持续言语羞辱：人格贬损性粗俗语言、造谣损害名誉
- 低俗性言论：对性特征的暗示提及、描述性行为过程、暗示性能力
- 暴力威胁：宣扬极端复仇、教唆犯罪
- 歧视性语言：种族/地域/民族/性别歧视的侮辱性言论

判断规则：
- 正常剧情对话中的情绪表达（如"我恨你"、"去你的"、"滚"）不算违规
- 谐音脏话、变体脏话（如"f**k"、"妈卖批"）也算违规
- 只有明确、直接的违规内容才标记，模糊情况倾向于不违规
- severity=low 不要报告，只报告 high 和 medium

severity 等级：
- high：明确脏话、严重侮辱、性暗示、涉性粗俗语言
- medium：贬损性词汇、轻度侮辱、暴力威胁

字幕文字列表（格式: [序号] [时间戳] 文字内容）：
{text_list}

仅返回JSON格式：
{{"violations": [{{"index": 0, "text": "原文", "reason": "违规原因", "severity": "high/medium"}}], "has_violation": true/false}}
如果全部正常，返回 {{"violations": [], "has_violation": false}}"""


def get_ocr_engine():
    """Lazy-load RapidOCR engine, preferring GPU if available."""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        try:
            import onnxruntime
            available = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available:
                _ocr_engine = RapidOCR(
                    det_use_cuda=True, rec_use_cuda=True, cls_use_cuda=True
                )
                logger.info("RapidOCR engine loaded (CUDA GPU)")
            else:
                _ocr_engine = RapidOCR()
                logger.info("RapidOCR engine loaded (CPU)")
        except Exception:
            _ocr_engine = RapidOCR()
            logger.info("RapidOCR engine loaded (CPU fallback)")
    return _ocr_engine


def extract_subtitles(image_path: str) -> list[dict]:
    """Extract subtitle text and bbox from bottom region of a frame.
    Returns list of {"text", "box", "confidence"}.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        return []

    h, w = frame.shape[:2]
    y_start = int(h * 0.75)  # cover bottom 25% to catch higher subtitle positions
    subtitle_crop = frame[y_start:h, :]
    if subtitle_crop.size == 0:
        return []

    engine = get_ocr_engine()
    result, _ = engine(subtitle_crop)
    if not result:
        return []

    texts = []
    for item in result:
        bbox, text, conf = item
        if conf < 0.5 or not text.strip() or len(text.strip()) < 2:
            continue
        pts = np.array(bbox)
        x1 = max(0, int(pts[:, 0].min()) - 10)
        y1 = max(0, int(pts[:, 1].min()) + y_start - 5)
        x2 = min(w, int(pts[:, 0].max()) + 10)
        y2 = min(h, int(pts[:, 1].max()) + y_start + 5)
        texts.append({
            "text": text.strip(),
            "box": [x1, y1, x2, y2],
            "confidence": round(conf, 3),
        })
    return texts


def check_instant_violations(subtitles: list[dict]) -> list[dict]:
    """Fast-path: check subtitles against keyword list.
    Returns violations with matched keywords.
    """
    violations = []
    for sub in subtitles:
        matches = _INSTANT_PATTERN.findall(sub["text"])
        # Also check context-dependent keywords
        for kw, pattern in CONTEXT_KEYWORDS.items():
            if pattern.search(sub["text"]):
                matches.append(kw)
        if matches:
            violations.append({
                "text": sub["text"],
                "keywords": matches,
                "box": sub["box"],
                "confidence": sub["confidence"],
                "need_mosaic": True,
                "reason": f"含违规词: {','.join(matches)}",
            })
    return violations


def check_vlm_violations_sync(subtitles: list[dict], timestamps: list[float]) -> list[int]:
    """Use VLM to judge if subtitle texts are violating (sync version).
    Returns list of indices that are violating.
    """
    from . import vlm

    text_lines = []
    for i, (sub, ts) in enumerate(zip(subtitles, timestamps)):
        text_lines.append(f"[{i}] [{ts:.1f}s] {sub['text']}")

    prompt = TEXT_AUDIT_PROMPT.format(text_list="\n".join(text_lines))
    messages = [{"role": "user", "content": prompt}]

    try:
        import json as json_mod
        import httpx
        from .. import config
        # Direct API call to avoid _call_vlm format issues with text-only prompts
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.VOLCANO_API_KEY}",
        }
        payload = {
            "model": config.VOLCANO_MODEL,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.1,
        }
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{config.VOLCANO_BASE_URL}/chat/completions",
                headers=headers, json=payload,
            )
            data = resp.json()
        logger.info(f"VLM text audit response keys: {list(data.keys())}")
        raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not raw:
            logger.warning(f"VLM text audit empty response: {str(data)[:200]}")
            return []
        logger.info(f"VLM text audit raw: {raw[:300]}")
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            result = json_mod.loads(raw[start:end])
            if result.get("has_violation"):
                indices = []
                for v in result.get("violations", []):
                    idx = v.get("index")
                    text = v.get("text", "")
                    reason = v.get("reason", "")
                    severity = v.get("severity", "medium")
                    if isinstance(idx, int) and 0 <= idx < len(subtitles):
                        logger.info(f"VLM flagged text [{idx}] severity={severity}: {text} reason={reason}")
                        if severity in ("high", "medium"):
                            indices.append(idx)
                        else:
                            # low severity: log only, no mosaic
                            logger.info(f"VLM low-severity text [{idx}] skipped for mosaic: {text}")
                return indices
    except Exception as e:
        import traceback
        logger.warning(f"VLM text audit failed: {e}\n{traceback.format_exc()}")
    return []
