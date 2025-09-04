from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from faker import Faker
from langchain_core.language_models.llms import LLM as LCBase


class SyntheticLLM(LCBase):
    """LangChain-compatible synthetic LLM: always returns a 5-word sentence.
    - Deterministic per prompt via base_seed + md5(prompt).
    """

    base_seed: int = 1337  # pydantic field

    # pydantic v2 hook to init runtime objects
    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._faker = Faker()
        self._faker.seed_instance(self.base_seed)

    @property
    def _llm_type(self) -> str:
        return "synthetic"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"base_seed": self.base_seed}

    def _seed_for_prompt(self, prompt: str) -> None:
        h = int(hashlib.md5(prompt.encode("utf-8")).hexdigest(), 16) & 0x7FFFFFFF
        self._faker.random.seed((self.base_seed + h) % 2_147_483_647)

    def _five_word_sentence(self) -> str:
        words: list[str] = self._faker.words(nb=5)
        if not words:
            return ""
        words[0] = words[0].capitalize()
        return " ".join(words) + "."

    def _call(self, prompt: str, stop: list[str] | None = None, **kwargs) -> str:
        self._seed_for_prompt(prompt)
        return self._five_word_sentence()
