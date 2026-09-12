from src.llm.ollama import OllamaClient
from src.core.inference import infer_prompt
from src.core.evaluation import evaluate_output

def main():
    client = OllamaClient()

    original_prompt = """
        あなたは広告コピーライターです。
        20代向けのカフェの広告文を作成してください。
        親しみやすく、短く、SNSに適した文章にしてください。
    """

    observed_output = client.generate(original_prompt)

    candidate_prompt = infer_prompt(observed_output)

    print("=== Original Prompt ===")
    print(original_prompt)

    print("\n=== Observed Output ===")
    print(observed_output)

    print("\n=== Candidate Prompt ===")
    print(candidate_prompt)

    score = evaluate_output(observed_output, candidate_prompt)
    print(f"\n=== Evaluation Score ===")
    print(score)

if __name__ == "__main__":
    main()