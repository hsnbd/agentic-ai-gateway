"""DeepEval judge model that calls a model through the gateway.

Judging through the gateway means evaluations get its routing, fallback,
budgets, and request log like any other traffic, and any model the gateway
serves (hosted or local) can be the judge.
"""

from __future__ import annotations

import json
import re
from typing import Any

import openai
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

#: Tags judge requests so they can be told apart from the traffic being judged.
JUDGE_TAG = "eval-judge"


class GatewayJudge(DeepEvalBaseLLM):
    def __init__(self, model: str, *, base_url: str, api_key: str) -> None:
        self._base_url = f"{base_url.rstrip('/')}/v1"
        self._api_key = api_key
        self._client = openai.OpenAI(base_url=self._base_url, api_key=api_key, max_retries=2)
        self._async: openai.AsyncOpenAI | None = None
        super().__init__(model)

    def load_model(self, *args: Any, **kwargs: Any) -> GatewayJudge:
        return self

    def _request(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "extra_body": {"aigw": {"tags": [JUDGE_TAG]}},
        }

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        response = self._client.chat.completions.create(**self._request(prompt))
        return parse_reply(response.choices[0].message.content or "", schema)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        if self._async is None:
            self._async = openai.AsyncOpenAI(
                base_url=self._base_url, api_key=self._api_key, max_retries=2
            )
        response = await self._async.chat.completions.create(**self._request(prompt))
        return parse_reply(response.choices[0].message.content or "", schema)

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return f"gateway:{self.name}"


def parse_reply(text: str, schema: type[BaseModel] | None) -> Any:
    """The judge's reply, validated against DeepEval's schema when it asks for one.

    Models often wrap JSON in prose or a code fence, so the outermost object is
    extracted first.
    """
    if schema is None:
        return text
    body = re.sub(r"\A\s*```(?:json)?\s*|\s*```\s*\Z", "", text.strip(), flags=re.IGNORECASE)
    start, end = body.find("{"), body.rfind("}")
    if start != -1 and end > start:
        body = body[start : end + 1]
    return schema.model_validate(json.loads(body))
