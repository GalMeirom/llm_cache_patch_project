import hashlib
from typing import Any

from faker import Faker
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class SyntheticLLM(LLM):
    """A synthetic LLM that generates 5-word responses using Faker.

    This LLM uses the prompt as a seed to generate deterministic but varied
    responses using the Faker library. Each prompt will always generate the
    same response, but different prompts will generate different responses.
    """

    def __init__(self, buffer=0.1, **kwargs):
        super().__init__(**kwargs)
        self.buffer = buffer

    @property
    def _llm_type(self) -> str:
        """Return identifier of llm type."""
        return "synthetic"

    @property
    def _identifying_params(self) -> dict:
        """Get the identifying parameters for this LLM."""
        return {
            "model_name": "synthetic_faker_llm",
            "version": "1.0.0",
        }

    def _generate_seed_from_prompt(self, prompt: str) -> int:
        """Generate a deterministic seed from the prompt."""
        # Use SHA-256 hash of the prompt to generate a consistent seed
        hash_object = hashlib.sha256(prompt.encode())
        hex_dig = hash_object.hexdigest()
        # Convert first 8 characters of hex to integer for seed
        return int(hex_dig[:8], 16)

    def _generate_five_words(self, seed: int) -> str:
        """Generate exactly 5 words using Faker with the given seed."""
        import time

        time.sleep(getattr(self, "buffer", 0))

        fake = Faker()
        fake.seed_instance(seed)

        # Generate different types of words to create variety
        word_generators = [
            fake.word,
            fake.color_name,
            fake.first_name,
            fake.city,
            fake.company,
            fake.catch_phrase,
            fake.bs,
        ]

        words = []
        for i in range(5):
            # Use modulo to cycle through generators and add some variety
            generator_index = (seed + i) % len(word_generators)
            try:
                # Some generators might return phrases, so split and take first word
                generated = word_generators[generator_index]()
                first_word = str(generated).split()[0].lower()
                # Remove any punctuation
                clean_word = "".join(c for c in first_word if c.isalnum())
                if clean_word:
                    words.append(clean_word)
                else:
                    # Fallback to simple word if cleaning resulted in empty string
                    words.append(fake.word())
            except Exception:
                # Fallback to simple word if any generator fails
                words.append(fake.word())

        return " ".join(words[:5])  # Ensure exactly 5 words

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Run the synthetic LLM on the given prompt.

        Args:
            prompt: The input prompt to use as seed for generation
            stop: Stop words (not used in this implementation)
            run_manager: Callback manager for the run
            **kwargs: Additional arguments (not used in this implementation)

        Returns:
            A string containing exactly 5 generated words
        """
        if run_manager:
            run_manager.on_text(f"Generating synthetic response for prompt: {prompt[:50]}...")

        # Generate seed from prompt
        seed = self._generate_seed_from_prompt(prompt)

        # Generate 5 words using the seed
        response = self._generate_five_words(seed)

        if run_manager:
            run_manager.on_text(f"Generated response: {response}")

        return response


# Example usage
if __name__ == "__main__":
    # Create an instance of the synthetic LLM
    llm = SyntheticLLM()

    # Test with different prompts
    test_prompts = [
        "Hello world",
        "What is the weather like today?",
        "Tell me about artificial intelligence",
        "Hello world",  # Same as first to show deterministic behavior
    ]

    print("Testing SyntheticLLM:")
    print("-" * 50)

    for prompt in test_prompts:
        response = llm.invoke(prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("-" * 50)
