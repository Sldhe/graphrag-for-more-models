import httpx
import json
from typing import Any, TYPE_CHECKING
from graphrag.language_model.response.base import (
    BaseModelOutput,
    BaseModelResponse,
    ModelResponse,
)
import asyncio
from typing import AsyncGenerator

if TYPE_CHECKING:
    from graphrag.config.models.language_model_config import LanguageModelConfig


class OneAPIChatLLM:
    """Chat provider via one-api (OpenAI-compatible)."""

    def __init__(self, *, name: str, config: "LanguageModelConfig", **kwargs: Any):
        self.name = name
        self.api_base = (config.api_base or "").rstrip("/")
        self.api_key = config.api_key or ""
        self.model = config.model
        self.timeout = config.request_timeout or 180.0  # 使用配置中的超时设置

        if not (self.api_base and self.api_key and self.model):
            raise ValueError(
                f"缺少必要参数: api_base={self.api_base}, api_key={'***' if self.api_key else None}, model={self.model}"
            )

    async def achat(self, prompt: str, history: list | None = None, **kwargs: Any) -> ModelResponse:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        messages = list(history or [])
        messages.append({"role": "user", "content": prompt})

        payload = {"model": self.model, "messages": messages, "stream": False}

        timeout = httpx.Timeout(self.timeout)

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{self.api_base}/v1/chat/completions",
                                     headers=headers, json=payload)
            data = resp.json()

        if "choices" not in data:
            raise ValueError(f"Invalid chat response: {data}")

        content = data["choices"][0]["message"]["content"]

        parsed_response = None
        # 支持 Graphrag 的 json=True/json_model
        if kwargs.get("json") and "json_model" in kwargs and kwargs["json_model"] is not None:
            try:
                json_model = kwargs["json_model"]
                parsed_response = json_model.parse_raw(content)
            except Exception as e:
                raise ValueError(f"Failed to parse model output as {kwargs['json_model']}: {e}\nOutput: {content}")

        return BaseModelResponse(
            output=BaseModelOutput(content=content, full_response=data),
            parsed_response=parsed_response,
            history=messages,
            cache_hit=False,
            tool_calls=[],
            metrics={},
        )

    async def achat_stream(self, prompt: str, history: list | None = None, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Stream responses from the OneAPI chat completions endpoint."""
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "text/event-stream"}
        messages = list(history or [])
        messages.append({"role": "user", "content": prompt})

        payload = {"model": self.model, "messages": messages, "stream": True}
        timeout = httpx.Timeout(self.timeout)

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self.api_base}/v1/chat/completions", headers=headers, json=payload) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    error_text = await response.aread()
                    if isinstance(error_text, (bytes, bytearray)):
                        error_body = error_text.decode("utf-8", "ignore")
                    else:
                        error_body = str(error_text)
                    raise ValueError(f"Streaming request failed: {exc.response.status_code} - {error_body}") from exc

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices")
                    if not choices:
                        continue

                    delta = choices[0].get("delta") or choices[0].get("message") or {}
                    content = delta.get("content")
                    if content is not None:
                        yield content

    def chat(self, prompt: str, history: list | None = None, **kwargs: Any) -> ModelResponse:
        import anyio
        return anyio.run(self.achat, prompt, history, **kwargs)



class OneAPIEmbeddingLLM:
    """Embedding provider via one-api (OpenAI-compatible)."""

    def __init__(self, *, name: str, config: "LanguageModelConfig", **kwargs: Any):
        self.name = name
        self.api_base = (config.api_base or "").rstrip("/")
        self.api_key = config.api_key or ""
        self.model = config.model
        self.timeout = config.request_timeout or 300.0  # 使用配置中的超时设置

        if not (self.api_base and self.api_key and self.model):
            raise ValueError(
                f"缺少必要参数: api_base={self.api_base}, api_key={'***' if self.api_key else None}, model={self.model}"
            )

    async def aembed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": text_list}

        # 使用配置的超时时间
        timeout = httpx.Timeout(self.timeout)

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{self.api_base}/v1/embeddings",
                                     headers=headers, json=payload)
            data = resp.json()

        if "data" not in data:
            raise ValueError(f"Invalid embedding response: {data}")

        return [item["embedding"] for item in data["data"]]

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        results = await self.aembed_batch([text], **kwargs)
        return results[0]

    def embed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        import anyio
        return anyio.run(self.aembed_batch, text_list, **kwargs)

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        import anyio
        return anyio.run(self.aembed, text, **kwargs)
