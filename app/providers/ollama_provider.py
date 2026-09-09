from ollama import Client
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class OllamaProvider:

    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434"
    ):
        self.model = model
        self.client = Client(host=host)

    def generate(self, prompt: str) -> str:

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    def generate_structured(
    self,
    prompt: str,
    schema: Type[T],
    system_prompt: str | None = None
) -> T:

        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        response = self.client.chat(
            model=self.model,
            messages=messages,
            format=schema.model_json_schema(),
            options={
                "temperature": 0
            }
        )

        content = response["message"]["content"]

        return schema.model_validate_json(content)

        