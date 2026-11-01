"""LLM 客户端：OpenAI 兼容 /chat/completions。

用标准库 urllib + asyncio.to_thread，不引入额外依赖。
兼容 DeepSeek、通义、Moonshot、智谱、OpenAI、Ollama(/v1)、各类中转。
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import log
from .config import LLMConfig

log_ = log.get("llm")


class LLMError(RuntimeError):
    def __init__(self, msg: str, *, retryable: bool = True) -> None:
        super().__init__(msg)
        self.retryable = retryable


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _attach_images(messages: list[dict], images: list[str]) -> list[dict]:
    """图片挂到最后一条 user 消息上，文本原样保留。

    浅拷贝一层再改：同一个 messages 在重试循环里会被复用，直接改原对象会让第二轮
    再挂一遍图片块。content 已经是数组时不动（幂等）。
    """
    out = [dict(m) for m in messages]
    for msg in reversed(out):
        if msg.get("role") != "user":
            continue
        text = msg.get("content")
        if isinstance(text, str):
            msg["content"] = [{"type": "text", "text": text}] + [
                {"type": "image_url", "image_url": {"url": u}} for u in images
            ]
        break
    return out


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._key = cfg.resolve_key()
        self.last_finish: str | None = None
        """上一次调用的 finish_reason。length 表示被 max_tokens 截断。"""

    @property
    def ready(self) -> bool:
        return bool(self._key)

    def reload(self, cfg: LLMConfig) -> None:
        """热更新配置（WebUI 改完模型参数后调用，不用重启）。"""
        self.cfg = cfg
        self._key = cfg.resolve_key()

    def _post(self, payload: dict, timeout: float | None = None) -> dict:
        url = f"{self.cfg.base_url}/chat/completions"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
                "User-Agent": "persona-workshop/0.1",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.cfg.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:500]
            except Exception:  # noqa: BLE001
                pass
            raise LLMError(f"HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"网络错误: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMError("请求超时") from exc
        except json.JSONDecodeError as exc:
            raise LLMError("响应不是合法 JSON") from exc

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        timeout: float | None = None,
        images: list[str] | None = None,
    ) -> str:
        """单轮调用。

        `images` 是 data URI（`data:image/png;base64,...`）列表，挂在**最后一条 user
        消息**上。文本部分原样保留，调用方不用自己拼多模态的 content 数组。
        """
        if not self._key:
            raise LLMError(f"缺少 API Key（环境变量 {self.cfg.api_key_env}）")

        if images:
            messages = _attach_images(messages, images)

        payload = {
            "model": model or self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }

        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = await asyncio.to_thread(self._post, payload, timeout)
                choices = resp.get("choices") or []
                if not choices:
                    raise LLMError(f"响应无 choices: {json.dumps(resp, ensure_ascii=False)[:300]}")
                content = (choices[0].get("message") or {}).get("content") or ""
                self.last_finish = choices[0].get("finish_reason")
                if not str(content).strip():
                    reasoning = (choices[0].get("message") or {}).get("reasoning_content") or ""
                    if reasoning:
                        # 推理型模型把预算全花在思考上，正文一个字都没轮到。
                        # 重试没用（结果一样），直接把原因说清楚，让人去调 max_tokens。
                        raise LLMError(
                            f"模型把 {payload['max_tokens']} token 全花在思考上了"
                            f"（finish_reason={self.last_finish}，思考 {len(reasoning)} 字），"
                            "正文是空的。调大该用途的 max_tokens，或换一个非推理模型",
                            retryable=False,
                        )
                    raise LLMError("模型返回空内容")
                return str(content).strip()
            except LLMError as exc:
                last_err = exc
                # 4xx（除 429）不重试，参数或鉴权问题重试也没用
                msg = str(exc)
                no_retry = (not getattr(exc, "retryable", True)) or (
                    msg.startswith("HTTP 4") and not msg.startswith("HTTP 429")
                )
                if no_retry or attempt >= self.cfg.max_retries:
                    break
                wait = 1.5 * (2**attempt)
                log_.warning("调用失败(%s)，%.1fs 后重试", msg[:80], wait)
                await asyncio.sleep(wait)

        raise LLMError(str(last_err) if last_err else "未知错误")

    async def complete(self, system: str, user: str, *, max_tokens: int = 300) -> str:
        """单轮调用，用于摘要等辅助任务。"""
        return await self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
