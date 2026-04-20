"""Heuristic cloud usage and cost estimation for VLM-heavy stages."""

from __future__ import annotations

from . import config


def _round_money(value: float) -> float:
    return round(max(0.0, float(value)), 4)


def _estimate_text_tokens(char_count: int) -> int:
    chars = max(0, int(char_count or 0))
    return int(round(chars * config.EST_VLM_TEXT_CHAR_TO_TOKEN_RATIO))


def estimate_vlm_cost(
    *,
    execution_profile: str | None = None,
    video_duration_seconds: float | None,
    total_extracted_frames: int,
    vlm_frame_calls: int,
    vlm_confirm_calls: int,
    text_batch_calls: int,
    text_batch_chars: int,
    highlight_images: int,
    highlight_batches: int,
) -> dict:
    """Estimate VLM usage for one processing run.

    The estimate is intentionally heuristic because provider-side image tokenization
    is not fully transparent from the public API responses used by this project.
    """
    frame_input_tokens = vlm_frame_calls * (
        config.EST_VLM_FRAME_INPUT_TEXT_TOKENS + config.EST_VLM_FRAME_IMAGE_INPUT_TOKENS
    )
    frame_output_tokens = vlm_frame_calls * config.EST_VLM_FRAME_OUTPUT_TOKENS

    confirm_input_tokens = vlm_confirm_calls * (
        config.EST_VLM_CONFIRM_INPUT_TEXT_TOKENS + config.EST_VLM_FRAME_IMAGE_INPUT_TOKENS
    )
    confirm_output_tokens = vlm_confirm_calls * config.EST_VLM_CONFIRM_OUTPUT_TOKENS

    text_input_tokens = text_batch_calls * config.EST_VLM_TEXT_BATCH_BASE_INPUT_TOKENS
    text_input_tokens += _estimate_text_tokens(text_batch_chars)
    text_output_tokens = text_batch_calls * config.EST_VLM_TEXT_BATCH_OUTPUT_TOKENS

    highlight_input_tokens = highlight_batches * config.EST_VLM_HIGHLIGHT_BATCH_INPUT_TEXT_TOKENS
    highlight_input_tokens += highlight_images * config.EST_VLM_HIGHLIGHT_IMAGE_INPUT_TOKENS
    highlight_output_tokens = highlight_batches * config.EST_VLM_HIGHLIGHT_BATCH_OUTPUT_TOKENS

    input_tokens = (
        frame_input_tokens +
        confirm_input_tokens +
        text_input_tokens +
        highlight_input_tokens
    )
    output_tokens = (
        frame_output_tokens +
        confirm_output_tokens +
        text_output_tokens +
        highlight_output_tokens
    )

    frame_cost = _round_money(
        frame_input_tokens / 1_000_000 * config.ARK_PRICE_INPUT_PER_MTOKEN +
        frame_output_tokens / 1_000_000 * config.ARK_PRICE_OUTPUT_PER_MTOKEN
    )
    confirm_cost = _round_money(
        confirm_input_tokens / 1_000_000 * config.ARK_PRICE_INPUT_PER_MTOKEN +
        confirm_output_tokens / 1_000_000 * config.ARK_PRICE_OUTPUT_PER_MTOKEN
    )
    text_cost = _round_money(
        text_input_tokens / 1_000_000 * config.ARK_PRICE_INPUT_PER_MTOKEN +
        text_output_tokens / 1_000_000 * config.ARK_PRICE_OUTPUT_PER_MTOKEN
    )
    highlight_cost = _round_money(
        highlight_input_tokens / 1_000_000 * config.ARK_PRICE_INPUT_PER_MTOKEN +
        highlight_output_tokens / 1_000_000 * config.ARK_PRICE_OUTPUT_PER_MTOKEN
    )
    total_cost = _round_money(frame_cost + confirm_cost + text_cost + highlight_cost)

    baseline_frame_calls = 0
    if video_duration_seconds:
        baseline_frame_calls = max(
            1,
            int(round(float(video_duration_seconds) / max(0.1, float(config.VLM_SAMPLE_INTERVAL)))),
        )
    frame_reduction_ratio = 0.0
    if baseline_frame_calls > 0:
        frame_reduction_ratio = round(
            max(0.0, 1.0 - (vlm_frame_calls / baseline_frame_calls)),
            4,
        )

    return {
        "cloud_profile": str(execution_profile or config.CLOUD_EXECUTION_PROFILE).lower(),
        "local_gpu_profile": config.LOCAL_GPU_PROFILE,
        "provider": "volcano_ark",
        "model": config.VOLCANO_MODEL,
        "video_duration_seconds": round(float(video_duration_seconds or 0.0), 2),
        "total_extracted_frames": int(total_extracted_frames or 0),
        "estimated_input_tokens": int(input_tokens),
        "estimated_output_tokens": int(output_tokens),
        "estimated_cost_cny": total_cost,
        "estimated_cost_breakdown_cny": {
            "frame_audit": frame_cost,
            "confirm": confirm_cost,
            "text_audit": text_cost,
            "highlight": highlight_cost,
        },
        "frame_audit_calls": int(vlm_frame_calls),
        "frame_audit_baseline_calls": int(baseline_frame_calls),
        "frame_call_reduction_ratio": frame_reduction_ratio,
        "confirm_calls": int(vlm_confirm_calls),
        "text_batch_calls": int(text_batch_calls),
        "text_batch_chars": int(text_batch_chars or 0),
        "highlight_images": int(highlight_images),
        "highlight_batches": int(highlight_batches),
        "estimation_method": "heuristic_token_model",
    }
