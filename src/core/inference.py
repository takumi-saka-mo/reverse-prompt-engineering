from src.llm.ollama import OllamaClient


def infer_prompt(observed_output: str) -> str:
    client = OllamaClient()

    meta_prompt = f"""
以下は、あるLLMが生成した出力です。

【Observed Output】
{observed_output}

この出力を生成した可能性のある元のPromptを推定してください。

推定したPromptだけを出力してください。
説明や前置きは不要です。
"""

    return client.generate(meta_prompt)
    