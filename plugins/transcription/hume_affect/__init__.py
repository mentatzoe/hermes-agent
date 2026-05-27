"""Hume Expression Measurement affect-overlay transcription provider.

This plugin keeps Zoe's Hume prosody overlay out of
``tools/transcription_tools.py``. Configure it as the active STT provider,
then point ``stt.hume_affect.primary_provider`` at the real speech-to-text
backend (``local``, ``groq``, ``openai``, ``mistral``, ``xai`` or a command
provider). The plugin runs the primary STT leg and the Hume batch prosody leg
in parallel, then injects compact affect tags into the transcript by timestamp
overlap.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.transcription_provider import TranscriptionProvider

logger = logging.getLogger(__name__)

HUME_BASE_URL = os.getenv("HUME_BASE_URL", "https://api.hume.ai/v0")

DEFAULT_HUME_AFFECT_THRESHOLD = 0.3
DEFAULT_HUME_AFFECT_TOP_K = 2
DEFAULT_HUME_POLL_INTERVAL_S = 0.5
DEFAULT_HUME_POLL_TIMEOUT_S = 60.0
DEFAULT_PRIMARY_PROVIDER = "local"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _get_env_value(name: str, default: Any = None) -> Any:
    """Read env values through Hermes config so ~/.hermes/.env is honored."""
    try:
        from hermes_cli.config import get_env_value

        return get_env_value(name, default)
    except Exception:
        return os.getenv(name, default)


def _load_stt_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        stt = cfg.get("stt") if isinstance(cfg, dict) else None
        return stt if isinstance(stt, dict) else {}
    except Exception as exc:
        logger.debug("Could not load stt config: %s", exc)
        return {}


def _plugin_config(stt_config: Dict[str, Any]) -> Dict[str, Any]:
    section = stt_config.get("hume_affect")
    return section if isinstance(section, dict) else {}


def _float_config(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_config(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_primary_provider(stt_config: Dict[str, Any]) -> str:
    cfg = _plugin_config(stt_config)
    provider = (
        cfg.get("primary_provider")
        or cfg.get("provider")
        or stt_config.get("hume_primary_provider")
        or DEFAULT_PRIMARY_PROVIDER
    )
    provider = str(provider).strip().lower()
    if not provider or provider in {"hume_affect", "hume-affect", "hume"}:
        return DEFAULT_PRIMARY_PROVIDER
    return provider


# ---------------------------------------------------------------------------
# Hume payload parsing and transcript merge
# ---------------------------------------------------------------------------


def _format_affect_tags(emotions: List[Dict[str, Any]], threshold: float, top_k: int) -> str:
    """Format Hume emotions as ``[Name 0.42, Other 0.38]``.

    If no emotion crosses the threshold, emit the dominant emotion only when it
    reaches half-threshold; otherwise return an empty tag for genuinely neutral
    audio.
    """
    ranked = sorted(
        (
            {
                "name": str(emo.get("name", "?")).strip() or "?",
                "score": _float_config(emo.get("score"), 0.0),
            }
            for emo in (emotions or [])
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    keep = [item for item in ranked if item["score"] >= threshold][: max(top_k, 1)]
    if not keep and ranked and ranked[0]["score"] >= threshold * 0.5:
        keep = [ranked[0]]
    if not keep:
        return ""
    return "[" + ", ".join(f"{item['name']} {item['score']:.2f}" for item in keep) + "]"


def _parse_hume_prediction_payload(payload: Any) -> List[Dict[str, Any]]:
    """Normalize Hume batch predictions into timed affect segments."""
    roots = payload if isinstance(payload, list) else [payload]
    segments: List[Dict[str, Any]] = []

    for root in roots:
        if not isinstance(root, dict):
            continue
        predictions = (root.get("results") or {}).get("predictions") or []
        for prediction in predictions:
            models = (prediction or {}).get("models") or {}
            for model_payload in models.values():
                grouped = (model_payload or {}).get("grouped_predictions") or []
                for group in grouped:
                    for utterance in (group or {}).get("predictions") or []:
                        time_obj = utterance.get("time") or {}
                        try:
                            begin = float(time_obj.get("begin", 0.0) or 0.0)
                            end = float(time_obj.get("end", begin) or begin)
                        except (TypeError, ValueError):
                            continue
                        emotions = utterance.get("emotions") or utterance.get("emotions_top") or []
                        segments.append({"begin": begin, "end": end, "emotions": emotions})

    return sorted(segments, key=lambda seg: (seg.get("begin", 0.0), seg.get("end", 0.0)))


def _merge_affect_overlay(
    primary_result: Dict[str, Any],
    affect_segments: List[Dict[str, Any]],
    stt_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge affect tags into the primary transcript by timestamp overlap."""
    hume_cfg = _plugin_config(stt_config)
    threshold = _float_config(hume_cfg.get("affect_threshold"), DEFAULT_HUME_AFFECT_THRESHOLD)
    top_k = _int_config(hume_cfg.get("affect_top_k"), DEFAULT_HUME_AFFECT_TOP_K)

    primary_segments = primary_result.get("segments") or []
    if not primary_segments:
        text = primary_result.get("transcript", "")
        if not affect_segments:
            return primary_result
        head_tag = _format_affect_tags(affect_segments[0].get("emotions", []), threshold, top_k)
        if head_tag:
            primary_result = dict(primary_result)
            primary_result["transcript"] = f"{head_tag} {text}".strip()
            primary_result["affect_provider"] = "hume"
        return primary_result

    annotated_parts: List[str] = []
    last_tag: Optional[str] = None
    for segment in primary_segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = _float_config(segment.get("start"), 0.0)
        end = _float_config(segment.get("end"), start)
        overlapping_emotions: List[Dict[str, Any]] = []
        for affect_segment in affect_segments:
            if _float_config(affect_segment.get("end"), 0.0) < start:
                continue
            if _float_config(affect_segment.get("begin"), 0.0) > end:
                continue
            overlapping_emotions.extend(affect_segment.get("emotions", []) or [])

        tag = ""
        if overlapping_emotions:
            by_name: Dict[str, float] = {}
            for emotion in overlapping_emotions:
                name = str(emotion.get("name", "?")).strip() or "?"
                score = _float_config(emotion.get("score"), 0.0)
                if score > by_name.get(name, 0.0):
                    by_name[name] = score
            tag = _format_affect_tags(
                [{"name": name, "score": score} for name, score in by_name.items()],
                threshold,
                top_k,
            )

        emit_tag = tag if tag and tag != last_tag else ""
        annotated_parts.append(f"{emit_tag} {text}".strip() if emit_tag else text)
        if tag:
            last_tag = tag

    primary_result = dict(primary_result)
    primary_result["transcript"] = " ".join(annotated_parts).strip() or primary_result.get("transcript", "")
    primary_result["affect_provider"] = "hume"
    return primary_result


# ---------------------------------------------------------------------------
# STT and Hume runners
# ---------------------------------------------------------------------------


def _run_primary_transcription(
    file_path: str,
    *,
    provider: str,
    model: Optional[str],
    stt_config: Dict[str, Any],
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Dispatch to the configured primary STT provider without recursion."""
    try:
        import tools.transcription_tools as stt_tools
    except Exception as exc:
        return {"success": False, "transcript": "", "provider": provider, "error": f"Could not import transcription tools: {exc}"}

    if provider == "local":
        local_cfg = stt_config.get("local", {}) if isinstance(stt_config.get("local"), dict) else {}
        model_name = stt_tools._normalize_local_model(  # noqa: SLF001 - current extension point has no public helper
            model or local_cfg.get("model", stt_tools.DEFAULT_LOCAL_MODEL)
        )
        return stt_tools._transcribe_local(file_path, model_name)  # noqa: SLF001

    if provider == "local_command":
        local_cfg = stt_config.get("local", {}) if isinstance(stt_config.get("local"), dict) else {}
        model_name = stt_tools._normalize_local_command_model(  # noqa: SLF001
            model or local_cfg.get("model", stt_tools.DEFAULT_LOCAL_MODEL)
        )
        return stt_tools._transcribe_local_command(file_path, model_name)  # noqa: SLF001

    if provider == "groq":
        return stt_tools._transcribe_groq(file_path, model or stt_tools.DEFAULT_GROQ_STT_MODEL)  # noqa: SLF001

    if provider == "openai":
        openai_cfg = stt_config.get("openai", {}) if isinstance(stt_config.get("openai"), dict) else {}
        return stt_tools._transcribe_openai(file_path, model or openai_cfg.get("model", stt_tools.DEFAULT_STT_MODEL))  # noqa: SLF001

    if provider == "mistral":
        mistral_cfg = stt_config.get("mistral", {}) if isinstance(stt_config.get("mistral"), dict) else {}
        return stt_tools._transcribe_mistral(file_path, model or mistral_cfg.get("model", stt_tools.DEFAULT_MISTRAL_STT_MODEL))  # noqa: SLF001

    if provider == "xai":
        return stt_tools._transcribe_xai(file_path, model or "grok-stt")  # noqa: SLF001

    # Command-type provider and plugin-provider fallback. Reuse upstream's
    # dispatcher helpers so this wrapper stays aligned with Hermes update.
    command_cfg = None
    try:
        command_cfg = stt_tools._resolve_command_stt_provider_config(provider, stt_config)  # noqa: SLF001
    except AttributeError:
        command_cfg = None
    if command_cfg is not None:
        return stt_tools._transcribe_command_stt(  # noqa: SLF001
            file_path,
            provider,
            command_cfg,
            stt_config,
            model_override=model,
        )

    plugin_cfg = stt_config.get(provider, {}) if isinstance(stt_config.get(provider), dict) else {}
    plugin_language = language or plugin_cfg.get("language")
    plugin_model = model or plugin_cfg.get("model")
    try:
        plugin_result = stt_tools._dispatch_to_plugin_provider(  # noqa: SLF001
            file_path,
            provider,
            stt_config,
            model=plugin_model,
            language=plugin_language,
        )
    except AttributeError:
        plugin_result = None
    if plugin_result is not None:
        return plugin_result

    return {
        "success": False,
        "transcript": "",
        "provider": provider,
        "error": f"No primary STT provider available for hume_affect.primary_provider={provider!r}",
    }


def _get_hume_affect_segments(file_path: str, stt_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run Hume Expression Measurement and return timed affect segments."""
    api_key = _get_env_value("HUME_API_KEY")
    if not api_key:
        return {"success": False, "segments": [], "error": "HUME_API_KEY not set"}

    try:
        import requests
    except Exception:
        return {"success": False, "segments": [], "error": "requests package not installed"}

    hume_cfg = _plugin_config(stt_config)
    model_name = str(hume_cfg.get("model") or "prosody")
    poll_interval = _float_config(hume_cfg.get("poll_interval_s"), DEFAULT_HUME_POLL_INTERVAL_S)
    poll_timeout = _float_config(hume_cfg.get("poll_timeout_s"), DEFAULT_HUME_POLL_TIMEOUT_S)
    base_url = str(hume_cfg.get("base_url") or HUME_BASE_URL).rstrip("/")

    request_json = {
        "models": {
            model_name: {
                "granularity": "utterance",
                "identify_speakers": False,
            }
        }
    }
    headers = {"X-Hume-Api-Key": str(api_key)}

    try:
        with open(file_path, "rb") as audio_file:
            submit = requests.post(
                f"{base_url}/batch/jobs",
                headers=headers,
                files={"file": (Path(file_path).name, audio_file)},
                data={"json": json.dumps(request_json)},
                timeout=30,
            )
        if submit.status_code >= 400:
            return {
                "success": False,
                "segments": [],
                "error": f"Hume submit failed (HTTP {submit.status_code}): {submit.text[:300]}",
            }
        submit_payload = submit.json()
        job_id = submit_payload.get("job_id") or submit_payload.get("id")
        if not job_id:
            return {"success": False, "segments": [], "error": "Hume submit response missing job_id"}

        deadline = time.monotonic() + poll_timeout
        while True:
            status_resp = requests.get(f"{base_url}/batch/jobs/{job_id}", headers=headers, timeout=15)
            if status_resp.status_code >= 400:
                return {
                    "success": False,
                    "segments": [],
                    "error": f"Hume poll failed (HTTP {status_resp.status_code}): {status_resp.text[:300]}",
                }
            status_payload = status_resp.json()
            state = status_payload.get("state") or {}
            status = str(state.get("status") or status_payload.get("status") or "").upper()
            if status == "COMPLETED":
                break
            if status in {"FAILED", "CANCELED", "CANCELLED"}:
                return {"success": False, "segments": [], "error": f"Hume job {job_id} ended with status {status}"}
            if time.monotonic() >= deadline:
                return {"success": False, "segments": [], "error": f"Hume job {job_id} timed out after {poll_timeout:.1f}s"}
            time.sleep(max(poll_interval, 0.1))

        predictions_resp = requests.get(f"{base_url}/batch/jobs/{job_id}/predictions", headers=headers, timeout=30)
        if predictions_resp.status_code >= 400:
            return {
                "success": False,
                "segments": [],
                "error": f"Hume predictions failed (HTTP {predictions_resp.status_code}): {predictions_resp.text[:300]}",
            }
        segments = _parse_hume_prediction_payload(predictions_resp.json())
        return {"success": True, "segments": segments, "provider": "hume", "job_id": job_id}
    except PermissionError:
        return {"success": False, "segments": [], "error": f"Permission denied: {file_path}"}
    except Exception as exc:
        logger.error("Hume affect overlay failed: %s", exc, exc_info=True)
        return {"success": False, "segments": [], "error": f"Hume affect overlay failed: {exc}"}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class HumeAffectTranscriptionProvider(TranscriptionProvider):
    @property
    def name(self) -> str:
        return "hume_affect"

    @property
    def display_name(self) -> str:
        return "Hume Affect Overlay"

    def is_available(self) -> bool:
        if not _get_env_value("HUME_API_KEY"):
            return False
        try:
            import requests  # noqa: F401
        except Exception:
            return False
        return True

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Hume Affect Overlay",
            "badge": "paid",
            "tag": "Runs primary STT plus Hume Expression Measurement prosody tags",
            "env_vars": [
                {
                    "key": "HUME_API_KEY",
                    "prompt": "Hume API key",
                    "url": "https://platform.hume.ai/settings/keys",
                },
            ],
        }

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        stt_config = _load_stt_config()
        provider = _resolve_primary_provider(stt_config)

        with ThreadPoolExecutor(max_workers=2) as executor:
            primary_future = executor.submit(
                _run_primary_transcription,
                file_path,
                provider=provider,
                model=model,
                stt_config=stt_config,
                language=language,
            )
            affect_future = executor.submit(_get_hume_affect_segments, file_path, stt_config)
            primary_result = primary_future.result()
            affect_result = affect_future.result()

        if not primary_result.get("success"):
            return primary_result

        if not affect_result.get("success"):
            logger.warning("Affect overlay failed: %s", affect_result.get("error"))
            return primary_result

        merged = _merge_affect_overlay(primary_result, affect_result.get("segments") or [], stt_config)
        merged["affect_provider"] = "hume"
        return merged


def register(ctx) -> None:
    """Plugin entry point — wire Hume affect overlay into STT registry."""
    ctx.register_transcription_provider(HumeAffectTranscriptionProvider())
