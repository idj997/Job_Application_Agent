"""Structural interface shared by local and API model providers."""

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class ModelProvider(Protocol):
    """Providers need not inherit from this protocol to satisfy it."""

    def generate(self, prompt: str) -> str: ...

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> T: ...
