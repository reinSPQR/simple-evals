import json
import time
from typing import Any

import openai
from openai import OpenAI
import requests

from ..models.oai_models import ChatCompletionResponse

from ..eval_types import MessageList, SamplerBase, SamplerResponse

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class LocalSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "no-need",
        model: str = "no-need",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
    ):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.enable_thinking = enable_thinking
        self.base_url = base_url
        self.api_key = api_key

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                final_payload = {
                    "model": self.model,
                    "messages": message_list,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "chat_template_kwargs": {
                        "enable_thinking": self.enable_thinking,
                    },
                    "stream": False,
                }

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=final_payload,
                    headers={
                        'Authorization': f'Bearer {self.api_key}'
                    },
                    timeout=600
                )
                response.raise_for_status()

                # print("URL: ", f"{self.base_url}/chat/completions")
                # print("Payload: ", json.dumps(final_payload, indent=2))
                print("Response: ", response)
                response = ChatCompletionResponse.model_validate(response.json())
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit (or other) exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
