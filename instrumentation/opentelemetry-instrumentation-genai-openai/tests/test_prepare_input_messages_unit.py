# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for input message preparation, including multimodal content."""

from dataclasses import asdict

from opentelemetry.instrumentation.genai.openai.utils import (
    _prepare_input_messages,
)
from opentelemetry.util.genai.types import GenericPart, TextPart
from opentelemetry.util.genai.utils import gen_ai_json_dumps


def test_string_content_maps_to_text_part():
    messages = [{"role": "user", "content": "hello"}]

    result = _prepare_input_messages(messages)

    assert result[0].parts == [TextPart(content="hello")]


def test_multimodal_content_keeps_text_part():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/cat.png"},
                },
            ],
        }
    ]

    result = _prepare_input_messages(messages)

    parts = result[0].parts
    assert parts[0] == TextPart(content="What is in this image?")
    assert isinstance(parts[1], GenericPart)
    assert parts[1].type == "image_url"
    assert parts[1].value == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/cat.png"},
    }


def test_input_audio_content_preserved_as_generic_part():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": "UklGRg==", "format": "wav"},
                },
            ],
        }
    ]

    result = _prepare_input_messages(messages)

    parts = result[0].parts
    assert [type(part) for part in parts] == [TextPart, GenericPart]
    assert parts[1].type == "input_audio"


def test_multimodal_assistant_history_content():
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "It is a cat."}],
        }
    ]

    result = _prepare_input_messages(messages)

    assert result[0].parts == [TextPart(content="It is a cat.")]


def test_none_content_produces_no_parts():
    messages = [{"role": "assistant", "content": None}]

    result = _prepare_input_messages(messages)

    assert result[0].parts == []


def test_multimodal_message_is_json_serializable():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "https://e/x.png"}},
            ],
        }
    ]

    result = _prepare_input_messages(messages)

    serialized = gen_ai_json_dumps([asdict(message) for message in result])
    assert '"type": "image_url"'.replace(" ", "") in serialized.replace(
        " ", ""
    )
