"""
Task 1 helper — benchmark two local models on the SAME prompt.

Measures, per model:
  - load time (first call, model cold -> loaded into RAM)
  - tokens/second (completion tokens / generation time)
  - completion token count
RAM is read separately from the OS (see the report); Ollama doesn't
report process RSS over its API.

Run with Ollama serving locally:  python benchmark.py
"""

import time

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

MODELS = ["llama3.2:3b", "qwen2.5:0.5b"]
PROMPT = "Explain what an inference engine is and why someone would self-host one, in about 4 sentences."


def bench(model: str) -> None:
    # Cold call: includes loading the weights into RAM.
    t0 = time.perf_counter()
    first = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
    )
    load_plus_gen = time.perf_counter() - t0

    # Warm call: model already resident, so this time is pure generation.
    t1 = time.perf_counter()
    warm = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
    )
    gen_time = time.perf_counter() - t1

    toks = warm.usage.completion_tokens or 0
    tps = toks / gen_time if gen_time > 0 else 0.0

    print(f"\n=== {model} ===")
    print(f"cold call (load + generate): {load_plus_gen:.2f}s")
    print(f"warm call (generate only):   {gen_time:.2f}s")
    print(f"completion tokens:           {toks}")
    print(f"tokens/sec (warm):           {tps:.1f}")
    print("--- answer (warm) ---")
    print(warm.choices[0].message.content.strip())


if __name__ == "__main__":
    for m in MODELS:
        bench(m)
