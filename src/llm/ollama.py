import ollama


class OllamaClient:
    def __init__(self, model: str = "qwen3:8b"):
        self.model = model

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            think=False,
        )

        if response.message.content is None:
            raise ValueError("Ollama returned no content.")

        return response.message.content