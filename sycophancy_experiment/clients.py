from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class APIError(RuntimeError):
    pass


def _join_messages(messages: List[Dict[str, str]]) -> str:
    chunks = []
    for m in messages:
        chunks.append(f"{m['role'].upper()}: {m['content']}")
    return "\n\n".join(chunks)


@dataclass
class OpenAICompatibleClient:
    model: str
    api_key: str
    base_url: str
    timeout_seconds: int = 120

    @retry(
        retry=retry_if_exception_type((requests.RequestException, APIError)),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
        # GPT-5 style models may require max_completion_tokens instead of max_tokens.
        if response.status_code == 400 and "max_completion_tokens" in response.text and "max_tokens" in response.text:
            payload_alt = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            }
            response = requests.post(url, headers=headers, json=payload_alt, timeout=self.timeout_seconds)

        if response.status_code >= 400:
            raise APIError(f"{self.model} error {response.status_code}: {response.text[:800]}")

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise APIError(f"Unexpected {self.model} response: {data}") from exc

        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            content = "".join(text_parts)
        return str(content).strip()


@dataclass
class AnthropicClient:
    model: str
    api_key: str
    timeout_seconds: int = 120

    @retry(
        retry=retry_if_exception_type((requests.RequestException, APIError)),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        system = "\n\n".join(system_parts).strip() if system_parts else None
        convo = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]

        payload = {
            "model": self.model,
            "messages": convo,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
        if response.status_code >= 400:
            raise APIError(f"{self.model} error {response.status_code}: {response.text[:800]}")

        data = response.json()
        try:
            blocks = data["content"]
        except KeyError as exc:
            raise APIError(f"Unexpected {self.model} response: {data}") from exc

        text = "".join(block.get("text", "") for block in blocks if isinstance(block, dict))
        return text.strip()


@dataclass
class BedrockAnthropicClient:
    model_id: str
    region: str
    profile: str = ""
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        try:
            import boto3
            from botocore.config import Config
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("boto3/botocore required for Bedrock usage. Install `boto3`.") from exc

        session = boto3.Session(profile_name=self.profile) if self.profile else boto3.Session()
        cfg = Config(
            connect_timeout=10,
            read_timeout=self.timeout_seconds,
            retries={"max_attempts": 5, "mode": "standard"},
        )
        self.client = session.client("bedrock-runtime", region_name=self.region, config=cfg)

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=12),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        system = "\n\n".join(system_parts).strip()

        convo = []
        for m in messages:
            if m["role"] == "system":
                continue
            convo.append(
                {
                    "role": m["role"],
                    "content": [{"type": "text", "text": str(m["content"])}],
                }
            )

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": convo,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system

        try:
            resp = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json",
            )
            body = json.loads(resp["body"].read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise APIError(f"{self.model_id} Bedrock invoke error: {exc}") from exc

        try:
            blocks = body["content"]
            text = "".join(block.get("text", "") for block in blocks if isinstance(block, dict))
        except Exception as exc:  # noqa: BLE001
            raise APIError(f"Unexpected Bedrock response for {self.model_id}: {body}") from exc

        return text.strip()


@dataclass
class BedrockConverseClient:
    model_id: str
    region: str
    profile: str = ""
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        try:
            import boto3
            from botocore.config import Config
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("boto3/botocore required for Bedrock usage. Install `boto3`.") from exc

        session = boto3.Session(profile_name=self.profile) if self.profile else boto3.Session()
        cfg = Config(
            connect_timeout=10,
            read_timeout=self.timeout_seconds,
            retries={"max_attempts": 5, "mode": "standard"},
        )
        self.client = session.client("bedrock-runtime", region_name=self.region, config=cfg)

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=12),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        system_text = "\n\n".join(system_parts).strip()

        convo = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = m["role"]
            if role not in {"user", "assistant"}:
                continue
            convo.append(
                {
                    "role": role,
                    "content": [{"text": str(m["content"])}],
                }
            )

        payload = {
            "modelId": self.model_id,
            "messages": convo,
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system_text:
            payload["system"] = [{"text": system_text}]

        try:
            resp = self.client.converse(**payload)
            blocks = resp["output"]["message"]["content"]
            text = "".join(block.get("text", "") for block in blocks if isinstance(block, dict))
            return text.strip()
        except Exception as exc:  # noqa: BLE001
            raise APIError(f"{self.model_id} Bedrock converse error: {exc}") from exc


@dataclass
class GeminiClient:
    model: str
    api_key: str
    timeout_seconds: int = 120

    @retry(
        retry=retry_if_exception_type((requests.RequestException, APIError)),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_output_tokens: int = 1024) -> str:
        # Gemini API receives the prompt as combined text.
        prompt = _join_messages(messages)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        response = requests.post(url, json=payload, timeout=self.timeout_seconds)
        if response.status_code >= 400:
            raise APIError(f"{self.model} error {response.status_code}: {response.text[:800]}")

        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise APIError(f"Unexpected {self.model} response: {data}") from exc

        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        return text.strip()


def build_clients_from_env() -> Dict[str, object]:
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    gemini_key = os.getenv("GOOGLE_API_KEY", "").strip()
    claude_backend = os.getenv("CLAUDE_BACKEND", "anthropic").strip().lower()
    qwen_key = os.getenv("QWEN_API_KEY", "").strip()
    qwen_base_url = os.getenv("QWEN_BASE_URL", "").strip()
    bedrock_region = os.getenv("BEDROCK_REGION", "us-east-1").strip()
    bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", "").strip()
    llama_bedrock_model_id = os.getenv("LLAMA_BEDROCK_MODEL_ID", "").strip()
    aws_profile = os.getenv("AWS_PROFILE", "").strip()

    openai_model = os.getenv("OPENAI_MODEL", "gpt-5.2")
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-1")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    qwen_model = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    clients: Dict[str, object] = {}

    if openai_key:
        clients["gpt-5.2"] = OpenAICompatibleClient(
            model=openai_model,
            api_key=openai_key,
            base_url="https://api.openai.com/v1",
        )
    if deepseek_key:
        clients["deepseek-v3.2"] = OpenAICompatibleClient(
            model=deepseek_model,
            api_key=deepseek_key,
            base_url="https://api.deepseek.com/v1",
        )
    if claude_backend == "bedrock":
        if bedrock_model_id:
            clients["claude-opus-4.6"] = BedrockAnthropicClient(
                model_id=bedrock_model_id,
                region=bedrock_region,
                profile=aws_profile,
            )
    elif anthropic_key:
        clients["claude-opus-4.6"] = AnthropicClient(model=anthropic_model, api_key=anthropic_key)
    if llama_bedrock_model_id:
        clients["llama3.1-8b-bedrock"] = BedrockConverseClient(
            model_id=llama_bedrock_model_id,
            region=bedrock_region,
            profile=aws_profile,
        )
    if gemini_key:
        clients["gemini-2.0-flash"] = GeminiClient(
            model=gemini_model,
            api_key=gemini_key,
        )
    if qwen_base_url:
        clients["qwen2.5-7b"] = OpenAICompatibleClient(
            model=qwen_model,
            api_key=qwen_key,
            base_url=qwen_base_url,
        )

    if not clients:
        raise ValueError(
            "No model clients configured. Set at least one provider key/base URL "
            "(e.g., OPENAI_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or QWEN_BASE_URL)."
        )
    return clients
