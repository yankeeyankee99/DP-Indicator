from __future__ import annotations
import os
import json
import asyncio
import copy
import re
from dataclasses import dataclass
from typing import Any, Iterable
import httpx

# Default retry config
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0  # seconds
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
# Additional transient errors worth retrying (malformed response, connection reset, etc.)
# Includes ReadError/WriteError/CloseError for mid-stream network failures
RETRY_EXCEPTIONS = (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError, httpx.CloseError)
MAX_PROVIDER_ERROR_BODY_CHARS = 4096
SAFE_PROVIDER_HEADER_NAMES = frozenset({
    "content-type",
    "retry-after",
    "request-id",
    "traceparent",
    "x-b3-traceid",
    "x-request-id",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
})
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(accesskey|api[_-]?key|authorization)\b\s*[:=]\s*"
    r"(?:bearer\s+)?[^\s,;\"'}]+"
)


@dataclass(frozen=True)
class ChatRawResult:
    """Stable transport result including the provider's complete JSON body."""

    content: str
    raw_response: dict[str, Any]
    usage: dict[str, Any]
    provider_model: str | None


class ProviderHTTPStatusError(httpx.HTTPStatusError):
    """HTTP status failure carrying only bounded, safe provider diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        request: httpx.Request,
        response: httpx.Response,
        response_body: str,
        retry_after: str | None,
        request_id: str | None,
        safe_headers: dict[str, str],
    ):
        super().__init__(message, request=request, response=response)
        self.status_code = response.status_code
        self.response_body = response_body
        self.retry_after = retry_after
        self.request_id = request_id
        self.safe_headers = safe_headers

    @classmethod
    def from_response(
        cls,
        response: httpx.Response,
        *,
        secret_values: Iterable[str] = (),
    ) -> "ProviderHTTPStatusError":
        safe_headers = {
            key.casefold(): value
            for key, value in response.headers.items()
            if key.casefold() in SAFE_PROVIDER_HEADER_NAMES
        }
        request_id = next(
            (
                safe_headers[name]
                for name in (
                    "x-request-id",
                    "request-id",
                    "x-b3-traceid",
                    "traceparent",
                )
                if name in safe_headers
            ),
            None,
        )
        body = response.text
        for secret in secret_values:
            if isinstance(secret, str) and secret:
                body = body.replace(secret, "[REDACTED]")
        body = _CREDENTIAL_ASSIGNMENT.sub(r"\1=[REDACTED]", body)
        body = body[:MAX_PROVIDER_ERROR_BODY_CHARS]
        safe_request = httpx.Request(
            response.request.method,
            response.request.url,
        )
        safe_response = httpx.Response(
            response.status_code,
            headers=safe_headers,
            text=body,
            request=safe_request,
        )
        message = (
            f"Provider returned HTTP {response.status_code} "
            f"for {response.request.method} {response.request.url}"
        )
        return cls(
            message,
            request=safe_request,
            response=safe_response,
            response_body=body,
            retry_after=safe_headers.get("retry-after"),
            request_id=request_id,
            safe_headers=safe_headers,
        )


class LLMClient:
    def __init__(self, model: str, api_key: str = None,
                 base_url: str = "https://open.bohrium.com/openapi/v1",
                 timeout: float = 180,
                 semaphore: asyncio.Semaphore = None,
                 router: object = None):
        self.model = model.removeprefix("bh:") if model else model
        self.api_key = api_key or os.environ.get("BH_API_KEY", "")
        if not self.api_key:
            raise ValueError("BH_API_KEY not set")
        self.base_url = base_url
        self.timeout = timeout  # Configurable timeout
        self._semaphore = semaphore  # Optional concurrency limit (Model Router)
        self._router = router  # Optional ModelRouter for token tracking
        self._client = httpx.AsyncClient(timeout=self.timeout)
        self._total_cost = 0.0
        self._total_tokens = {"input": 0, "output": 0}

    async def _retry_with_backoff(self, func, *args, max_retries: int = DEFAULT_MAX_RETRIES, **kwargs):
        """Execute func with exponential backoff retry for transient errors."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code not in RETRY_STATUS_CODES or attempt >= max_retries:
                    raise
                wait = DEFAULT_RETRY_DELAY * (2 ** attempt)
                # Respect Retry-After header if present
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    try:
                        wait = max(float(retry_after), wait)
                    except ValueError:
                        pass
                print(f"  [LLM] HTTP {e.response.status_code}, retry {attempt+1}/{max_retries} in {wait:.1f}s", flush=True)
                await asyncio.sleep(wait)
            except RETRY_EXCEPTIONS as e:
                last_error = e
                if attempt >= max_retries:
                    raise
                wait = DEFAULT_RETRY_DELAY * (2 ** attempt)
                print(f"  [LLM] {type(e).__name__}, retry {attempt+1}/{max_retries} in {wait:.1f}s", flush=True)
                await asyncio.sleep(wait)
        raise last_error

    async def chat(self, messages: list[dict], max_tokens: int = 4096,
                   temperature: float = 0.7,
                   max_retries: int = DEFAULT_MAX_RETRIES,
                   task: str = None,
                   top_p: float | None = None,
                   response_format: dict | None = None) -> tuple[str, dict]:
        """Send chat completion request.
        
        Args:
            task: Optional task name for Model Router token tracking.
                  If provided and router is set, tokens are recorded.
        """
        result = await self.chat_raw(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
            max_retries=max_retries,
            task=task,
        )
        return result.content, result.usage

    async def chat_raw(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float | None = None,
        response_format: dict | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        task: str = None,
    ) -> ChatRawResult:
        """Send a chat request and retain the complete provider JSON response."""
        headers = {
            "accessKey": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if response_format is not None:
            payload["response_format"] = response_format

        async def _do_request():
            if self._semaphore:
                async with self._semaphore:
                    resp = await self._client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers, json=payload,
                    )
            else:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers, json=payload,
                )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ProviderHTTPStatusError.from_response(
                    resp,
                    secret_values=[self.api_key],
                ) from exc
            return resp.json()
        data = await self._retry_with_backoff(_do_request, max_retries=max_retries)
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self._total_tokens["input"] += usage.get("prompt_tokens", 0)
        self._total_tokens["output"] += usage.get("completion_tokens", 0)
        # Record tokens via Model Router if task is specified
        if task and self._router:
            self._router.record_tokens(task, usage)
        provider_model = data.get("model")
        if not isinstance(provider_model, str) or not provider_model:
            provider_model = None
        return ChatRawResult(
            content=choice,
            raw_response=data,
            usage=usage,
            provider_model=provider_model,
        )
    async def structured(self, messages: list[dict],
                         schema: dict = None, max_tokens: int = 4096,
                         max_retries: int = 2,
                         task: str = None,
                         temperature: float = 0.3) -> tuple[dict, dict]:
        original_messages = copy.deepcopy(messages)
        if schema:
            schema_text = (
                "\n\nReturn JSON matching this schema:\n"
                f"{json.dumps(schema, indent=2)}"
            )
            if original_messages and original_messages[0].get("role") == "system":
                original_messages[0]["content"] = (
                    str(original_messages[0].get("content", "")) + schema_text
                )
            else:
                original_messages.insert(
                    0,
                    {"role": "system", "content": schema_text.lstrip()},
                )
        retry_messages = None
        for attempt in range(max_retries + 1):
            content, usage = await self.chat(
                retry_messages if retry_messages else original_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                task=task,
                response_format={"type": "json_object"},
            )
            content = content.strip()
            # Strip markdown code fences
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            # Strip leading 'json' keyword
            if content.lower().startswith("json"):
                content = content[4:].strip()
            try:
                return json.loads(content), usage
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    # Retry with error feedback — build from original to avoid accumulation
                    error_msg = f"JSON parse error: {str(e)[:100]}. Please return ONLY valid JSON, no markdown, no explanation."
                    retry_messages = original_messages + [{"role": "assistant", "content": content},
                                                    {"role": "user", "content": error_msg}]
                else:
                    return {"error": "json_parse_failed", "raw": content}, usage
    async def aclose(self):
        await self._client.aclose()
    @property
    def stats(self):
        return {"cost_usd": self._total_cost, "tokens": self._total_tokens}
