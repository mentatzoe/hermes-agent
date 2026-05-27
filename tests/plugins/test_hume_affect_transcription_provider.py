"""Tests for the Hume affect-overlay transcription provider plugin."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


PLUGIN_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "transcription"
    / "hume_affect"
    / "__init__.py"
)


@pytest.fixture()
def hume_affect_module():
    spec = importlib.util.spec_from_file_location("hume_affect_under_test", PLUGIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_format_affect_tags_threshold_and_half_threshold_fallback(hume_affect_module):
    emotions = [
        {"name": "Calmness", "score": 0.28},
        {"name": "Joy", "score": 0.12},
    ]

    assert (
        hume_affect_module._format_affect_tags(emotions, threshold=0.3, top_k=2)
        == "[Calmness 0.28]"
    )
    assert hume_affect_module._format_affect_tags(emotions, threshold=0.8, top_k=2) == ""


def test_parse_hume_prediction_payload(hume_affect_module):
    payload = [
        {
            "results": {
                "predictions": [
                    {
                        "models": {
                            "prosody": {
                                "grouped_predictions": [
                                    {
                                        "predictions": [
                                            {
                                                "time": {"begin": 1.2, "end": 2.4},
                                                "emotions": [{"name": "Interest", "score": 0.51}],
                                            }
                                        ]
                                    }
                                ]
                            }
                        }
                    }
                ]
            }
        }
    ]

    assert hume_affect_module._parse_hume_prediction_payload(payload) == [
        {"begin": 1.2, "end": 2.4, "emotions": [{"name": "Interest", "score": 0.51}]}
    ]


def test_merge_affect_overlay_dedupes_repeated_tags(hume_affect_module):
    primary = {
        "success": True,
        "provider": "local",
        "transcript": "hello world",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ],
    }
    affect = [
        {"begin": 0.0, "end": 2.0, "emotions": [{"name": "Calmness", "score": 0.7}]},
    ]

    result = hume_affect_module._merge_affect_overlay(
        primary,
        affect,
        {"hume_affect": {"affect_threshold": 0.3, "affect_top_k": 2}},
    )

    assert result["transcript"] == "[Calmness 0.70] hello world"
    assert result["affect_provider"] == "hume"


def test_provider_runs_primary_and_hume_overlay(hume_affect_module, tmp_path):
    audio = tmp_path / "sample.ogg"
    audio.write_bytes(b"fake ogg")
    cfg = {
        "provider": "hume_affect",
        "hume_affect": {
            "primary_provider": "local",
            "affect_threshold": 0.3,
            "affect_top_k": 2,
        },
    }
    primary = {
        "success": True,
        "provider": "local",
        "transcript": "hello",
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
    }
    affect = {
        "success": True,
        "provider": "hume",
        "segments": [{"begin": 0.0, "end": 1.0, "emotions": [{"name": "Interest", "score": 0.55}]}],
    }

    with patch.object(hume_affect_module, "_load_stt_config", return_value=cfg), \
         patch.object(hume_affect_module, "_run_primary_transcription", return_value=primary) as mock_primary, \
         patch.object(hume_affect_module, "_get_hume_affect_segments", return_value=affect) as mock_hume:
        result = hume_affect_module.HumeAffectTranscriptionProvider().transcribe(str(audio))

    assert result["success"] is True
    assert result["transcript"] == "[Interest 0.55] hello"
    assert result["affect_provider"] == "hume"
    mock_primary.assert_called_once()
    mock_hume.assert_called_once_with(str(audio), cfg)


def test_register_wires_provider_into_context(hume_affect_module):
    class DummyContext:
        def __init__(self):
            self.providers = []

        def register_transcription_provider(self, provider):
            self.providers.append(provider)

    ctx = DummyContext()
    hume_affect_module.register(ctx)

    assert len(ctx.providers) == 1
    assert ctx.providers[0].name == "hume_affect"
