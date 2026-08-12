import asyncio
import copy
import inspect
import json

import httpx

from dp_indicator.core.llm import LLMClient, ProviderHTTPStatusError


def test_chat_raw_signature_is_explicit_and_backward_compatible():
    signature = inspect.signature(LLMClient.chat_raw)
    assert list(signature.parameters) == [
        "self",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "response_format",
        "max_retries",
        "task",
    ]
    assert signature.parameters["top_p"].default is None
    assert signature.parameters["response_format"].default is None
    chat_signature = inspect.signature(LLMClient.chat)
    assert list(chat_signature.parameters) == [
        "self",
        "messages",
        "max_tokens",
        "temperature",
        "max_retries",
        "task",
        "top_p",
        "response_format",
    ]
    assert chat_signature.parameters["top_p"].default is None
    assert chat_signature.parameters["response_format"].default is None


def test_chat_raw_sends_top_p_and_response_format_and_returns_full_response():
    captured = {}

    async def scenario():
        async def handler(request):
            captured["payload"] = __import__("json").loads(
                request.content.decode("utf-8")
            )
            return httpx.Response(
                200,
                json={
                    "id": "response-1",
                    "model": "provider/model-revision",
                    "choices": [
                        {"message": {"content": '{"answer": "ok"}'}}
                    ],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 3,
                    },
                },
            )

        client = LLMClient(model="bh:test-model", api_key="offline-test-key")
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        try:
            result = await client.chat_raw(
                [{"role": "user", "content": "hello"}],
                max_tokens=123,
                temperature=0.25,
                top_p=0.42,
                response_format={"type": "json_object"},
                max_retries=0,
                task="critic",
            )
            return result
        finally:
            await client.aclose()

    result = asyncio.run(scenario())

    assert captured["payload"] == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 123,
        "temperature": 0.25,
        "top_p": 0.42,
        "response_format": {"type": "json_object"},
    }
    assert result.content == '{"answer": "ok"}'
    assert result.raw_response["id"] == "response-1"
    assert result.usage == {"prompt_tokens": 7, "completion_tokens": 3}
    assert result.provider_model == "provider/model-revision"


def test_existing_chat_delegates_to_chat_raw_and_keeps_tuple_contract():
    captured = {}

    async def scenario():
        async def handler(request):
            captured["payload"] = __import__("json").loads(
                request.content.decode("utf-8")
            )
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "legacy"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                },
            )

        client = LLMClient(model="legacy-model", api_key="offline-test-key")
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        try:
            return await client.chat(
                [{"role": "user", "content": "legacy call"}],
                max_retries=0,
            )
        finally:
            await client.aclose()

    result = asyncio.run(scenario())

    assert result == (
        "legacy",
        {"prompt_tokens": 2, "completion_tokens": 1},
    )
    assert "top_p" not in captured["payload"]
    assert "response_format" not in captured["payload"]


def test_chat_raw_raises_safe_provider_http_error_with_diagnostics():
    secret = "offline-test-key"
    oversized = "x" * 5000

    async def scenario():
        async def handler(request):
            return httpx.Response(
                429,
                json={
                    "error": {
                        "type": "quota_exhausted",
                        "message": f"accessKey={secret} {oversized}",
                    }
                },
                headers={
                    "Retry-After": "17",
                    "X-Request-ID": "req-429",
                    "X-RateLimit-Remaining": "0",
                    "Authorization": f"Bearer {secret}",
                    "AccessKey": secret,
                },
            )

        client = LLMClient(model="test-model", api_key=secret)
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(ProviderHTTPStatusError) as caught:
                await client.chat_raw(
                    [{"role": "user", "content": "hello"}],
                    max_retries=0,
                )
            return caught.value
        finally:
            await client.aclose()

    import pytest

    error = asyncio.run(scenario())
    assert isinstance(error, httpx.HTTPStatusError)
    assert error.status_code == 429
    assert error.retry_after == "17"
    assert error.request_id == "req-429"
    assert error.safe_headers == {
        "content-type": "application/json",
        "retry-after": "17",
        "x-request-id": "req-429",
        "x-ratelimit-remaining": "0",
    }
    assert "quota_exhausted" in error.response_body
    assert secret not in error.response_body
    assert len(error.response_body) <= 4096
    assert secret not in str(error)
    assert secret not in repr(error.__dict__)
    assert "accesskey" not in error.request.headers
    assert "accesskey" not in error.response.headers


def test_structured_sends_json_format_without_mutating_caller_messages():
    captured = {}
    messages = [
        {"role": "system", "content": "original system"},
        {"role": "user", "content": "classify"},
    ]
    before = copy.deepcopy(messages)

    async def scenario():
        async def handler(request):
            captured["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": '{"answer": "ok"}'}}],
                    "usage": {},
                },
            )

        client = LLMClient(model="structured-model", api_key="offline-test-key")
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        try:
            return await client.structured(
                messages,
                schema={"type": "object"},
                max_retries=0,
            )
        finally:
            await client.aclose()

    result = asyncio.run(scenario())

    assert result == ({"answer": "ok"}, {})
    assert messages == before
    assert captured["payload"]["response_format"] == {
        "type": "json_object"
    }
    assert "Return JSON matching this schema" in (
        captured["payload"]["messages"][0]["content"]
    )
