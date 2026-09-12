from src.llm.ollama import OllamaClient


def evaluate_output(
    observed_output: str,
    candidate_output: str,
) -> float:
    client = OllamaClient()

    meta_prompt = f"""
以下の2つの文章を比較してください。

【Observed Output】
{observed_output}

【Candidate Output】
{candidate_output}

Candidate OutputがObserved Outputと
どの程度意味的・内容的に一致しているかを
0.0〜1.0のスコアで評価してください。

0.0 = まったく一致していない
1.0 = ほぼ完全に一致している

数値だけを出力してください。
"""

    result = client.generate(meta_prompt)

    return float(result.strip())