"""Volcano VLM-based content detection and highlight recognition."""
import base64
import json
import logging
import httpx
from .. import config

logger = logging.getLogger(__name__)

CONFIRM_PROMPT = """这是微短剧的一帧截图。AI检测系统标记了胸部区域可能存在暴露。
请判断画面中胸部区域是否需要打码。

必须打码：
- 胸部大面积裸露或乳沟明显可见
- 低胸/深V导致胸部轮廓大面积暴露
- 内衣外穿、透视可见内衣
- 抹胸、吊带等导致胸部特征过度突出

不需要打码：
- 高领、圆领、普通T恤等遮挡良好的服装
- 男性角色的正常着装
- 画面中没有明显的胸部暴露

注意：宁可多打也不要漏打，擦边情况判定为需要打码。
仅返回JSON：{"is_violation": true, "reason": "简短原因"} 或 {"is_violation": false, "reason": "简短原因"}"""

AUDIT_PROMPT = """你是微短剧视频内容审核专家，依据平台审核标准判定画面违规。请分析这张视频截图，检测以下违规内容：

1. 低俗色情（画面层面）：
   - high：直接裸露敏感部位、呈现性行为画面、人物整体观感无异于裸体
   - medium：着装低俗擦边服饰且伴有低俗动作、仅遮挡敏感部位、两性特征过度暴露突出
   - 不算违规：正常剧情服装（V领、吊带、连衣裙、低胸礼服、西装、正装）

2. 血腥暴力：
   - high：逼真血腥场景（血肉模糊特写、断肢）、极端施虐画面
   - medium：明显施暴场面、过度痛苦表情呈现受虐

3. 恐怖惊悚：
   - high：极端恐怖画风、人物外形严重扭曲恐怖
   - medium：画面整体阴冷昏暗+异常打光营造恐怖氛围

4. 违规文字：字幕/标题中的低俗用语、性暗示、暴力威胁、人身攻击。正常剧情对话不算违规。

5. 未成年负面：未成年人遭受暴力欺凌、未成年人实施违法行为

重要判定原则（必须严格遵守）：
- 这是微短剧，演员穿着时尚服装属于正常剧情需要，不算违规
- 只有真正的裸露、内衣外穿、透视可见内衣、极度暴露才算"低俗色情"
- 正常的打斗场景如果没有血腥画面，不算"血腥暴力"
- 宁可漏判也不要误判，模糊情况一律判定为safe
- severity=low 的内容不要报告，只报告 high 和 medium

对于"低俗色情"类型，返回精确的 bbox [x, y, w, h]（百分比0-100），仅框选违规的身体部位，不要框选整个人物。
bbox 宽高各不应超过 25（即画面的25%），超过说明框选范围太大。

仅返回JSON：
{"violations": [{"type": "类型", "description": "描述", "severity": "high/medium", "region": "位置", "bbox": [x, y, w, h]}], "safe": true/false}
如果没有违规，返回 {"violations": [], "safe": true}"""

HIGHLIGHT_PROMPT = """你是一个视频剪辑专家。请分析这组视频截图（按时间顺序排列），识别其中的高光片段。
每张截图前标注了对应的时间戳，如 [0.0s]、[5.0s] 等。
高光片段包括：精彩打斗、情感高潮、剧情反转、搞笑场景、视觉震撼等。

请根据截图的时间戳，判断每个高光片段的起止时间。如果高光只出现在一张截图中，end_time 设为 start_time 往后推一个采样间隔。

请返回JSON格式：
{"highlights": [{"start_time": "0.0s", "end_time": "5.0s", "description": "高光内容描述", "reason": "为什么是高光"}]}
如果没有明显高光，返回 {"highlights": []}"""


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _call_vlm(messages: list, max_tokens: int = 1024) -> str:
    """Call Volcano VLM API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.VOLCANO_API_KEY}",
    }
    payload = {
        "model": config.VOLCANO_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{config.VOLCANO_BASE_URL}/chat/completions",
            headers=headers, json=payload,
        )
        data = resp.json()
    if "error" in data:
        raise RuntimeError(f"VLM error: {data['error']}")
    return data["choices"][0]["message"]["content"]


async def _call_vlm_async(messages: list, max_tokens: int = 1024) -> str:
    """Async version of VLM API call."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.VOLCANO_API_KEY}",
    }
    payload = {
        "model": config.VOLCANO_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{config.VOLCANO_BASE_URL}/chat/completions",
            headers=headers, json=payload,
        )
        data = resp.json()
    if "error" in data:
        raise RuntimeError(f"VLM error: {data['error']}")
    return data["choices"][0]["message"]["content"]


async def detect_frame_vlm_async(image_path: str, frame_width: int = 1920,
                                  frame_height: int = 1080) -> dict:
    """Async version of detect_frame_vlm."""
    b64 = _encode_image(image_path)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": AUDIT_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]}]
    try:
        raw = await _call_vlm_async(messages)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(raw[start:end])
            for v in result.get("violations", []):
                v["need_mosaic"] = False
                bbox = v.get("bbox")
                if bbox and len(bbox) == 4:
                    # Convert percent bbox [x%, y%, w%, h%] to pixel coords
                    v["box"] = _bbox_percent_to_pixels(bbox, frame_width, frame_height)
                else:
                    region = v.get("region", "center")
                    v["box"] = _region_to_box(region, frame_width, frame_height)
            return result
        return {"violations": [], "safe": True}
    except Exception as e:
        logger.error(f"VLM async detection failed: {e}")
        return {"violations": [], "safe": True, "error": str(e)}


def _bbox_percent_to_pixels(bbox: list, width: int, height: int) -> list[int]:
    """Convert percent bbox [x%, y%, w%, h%] to pixel [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    x1 = int(x / 100 * width)
    y1 = int(y / 100 * height)
    x2 = int((x + w) / 100 * width)
    y2 = int((y + h) / 100 * height)
    # Clamp to frame bounds
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    return [x1, y1, x2, y2]


def _region_to_box(region: str, width: int = 1920, height: int = 1080) -> list[int]:
    """Convert region description to [x1, y1, x2, y2] box coordinates."""
    # Use tighter bounds (40% instead of 50%) to reduce over-mosaicking
    region_map = {
        "full":         [0, 0, width, height],
        "center":       [int(width * 0.3), int(height * 0.3), int(width * 0.7), int(height * 0.7)],
        "upper_body":   [width // 4, height // 6, width * 3 // 4, height * 2 // 3],
        "top-left":     [0, 0, int(width * 0.4), int(height * 0.4)],
        "top-right":    [int(width * 0.6), 0, width, int(height * 0.4)],
        "bottom-left":  [0, int(height * 0.6), int(width * 0.4), height],
        "bottom-right": [int(width * 0.6), int(height * 0.6), width, height],
        "top":          [0, 0, width, int(height * 0.4)],
        "bottom":       [0, int(height * 0.6), width, height],
        "left":         [0, int(height * 0.1), int(width * 0.4), int(height * 0.9)],
        "right":        [int(width * 0.6), int(height * 0.1), width, int(height * 0.9)],
        "subtitle":     [width // 6, int(height * 0.85), width * 5 // 6, int(height * 0.98)],
    }
    return region_map.get(region, region_map["center"])


def _text_region_to_box(region: str, width: int, height: int) -> list[int]:
    """Convert text violation region to precise subtitle bbox.
    Subtitles are typically in the bottom 12-15% of the frame, centered.
    """
    if region in ("subtitle", "bottom", "bottom-left", "bottom-right"):
        # Bottom subtitle area: centered, bottom 12%
        return [width // 6, int(height * 0.85), width * 5 // 6, int(height * 0.98)]
    elif region in ("top", "top-left", "top-right"):
        # Top title/danmaku area: top 10%
        return [width // 8, int(height * 0.02), width * 7 // 8, int(height * 0.12)]
    elif region == "center":
        # Center text overlay
        return [width // 6, int(height * 0.4), width * 5 // 6, int(height * 0.6)]
    else:
        # Default: bottom subtitle
        return [width // 6, int(height * 0.85), width * 5 // 6, int(height * 0.98)]


async def confirm_violation_async(image_path: str) -> dict:
    """Lightweight VLM confirmation for a single borderline NudeNet detection.
    Returns {"is_violation": bool, "reason": str}.
    """
    b64 = _encode_image(image_path)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": CONFIRM_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]}]
    try:
        raw = await _call_vlm_async(messages, max_tokens=256)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(raw[start:end])
            return result
        return {"is_violation": False, "reason": "parse_error"}
    except Exception as e:
        logger.error(f"VLM confirm failed: {e}")
        # On error, default to keeping the mosaic (conservative)
        return {"is_violation": True, "reason": f"error: {e}"}


def detect_frame_vlm(image_path: str, frame_width: int = 1920,
                     frame_height: int = 1080) -> dict:
    """Detect content violations using VLM. Returns parsed result.
    For medium/high severity violations, sets need_mosaic=True with estimated box.
    """
    b64 = _encode_image(image_path)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": AUDIT_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]}]
    try:
        raw = _call_vlm(messages)
        # Extract JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(raw[start:end])
            # Mark need_mosaic only for violations with precise regions
            # VLM regions are too coarse for auto-mosaic, mark for report only
            for v in result.get("violations", []):
                v["need_mosaic"] = False
                bbox = v.get("bbox")
                if bbox and len(bbox) == 4:
                    v["box"] = _bbox_percent_to_pixels(bbox, frame_width, frame_height)
                else:
                    region = v.get("region", "center")
                    v["box"] = _region_to_box(region, frame_width, frame_height)
            return result
        return {"violations": [], "safe": True}
    except Exception as e:
        logger.error(f"VLM detection failed: {e}")
        return {"violations": [], "safe": True, "error": str(e)}
