"""
Task 2 — Hit the local Ollama endpoint from Python.

Ollama exposes an OpenAI-compatible HTTP API on http://localhost:11434.
That means the SAME client code you used for a hosted API works here —
you only change the base URL (and the API key is a dummy value locally).

Run Ollama first (it starts a server automatically when you `ollama run`
or `ollama serve`), then:

    pip install -r requirements.txt
    python local_client.py

------------------------------------------------------------------------
WHY THIS IS "THE SAME SHAPE" AS YESTERDAY'S HOSTED GEMINI CALL
------------------------------------------------------------------------
Calling an LLM is not magic and it is not tied to any one vendor's cloud.
It is just an HTTP POST to an inference server:

    POST <base_url>/chat/completions
    Authorization: Bearer <api_key>
    body: { "model": ..., "messages": [ {role, content}, ... ] }

Yesterday's hosted call and today's local call differ in only two fields:

    hosted Gemini   ->  base_url = https://...googleapis.../  ,  real API key
    local Ollama    ->  base_url = http://localhost:11434/v1  ,  dummy key

Same request shape, same `messages` array, same `choices[0].message.content`
response shape. The model weights now live on THIS machine instead of in a
data center, but the *interface* — and therefore your client code — is
identical. That is the whole point of an OpenAI-compatible endpoint: the
server is swappable, the contract is not.
"""

import sys
import time

from openai import OpenAI

# Point the OpenAI client at your LOCAL Ollama server instead of the cloud.
# This is the whole insight of the lab: "calling an LLM" is just an HTTP
# request to an inference server — wherever that server happens to run.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required by the client, but ignored by Ollama
)

MODEL = "llama3.2:3b"  # a model you pulled with `ollama pull`

PROMPT = "In one sentence, what is an inference engine?"


def main() -> None:
    try:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": PROMPT},
            ],
        )
        elapsed = time.perf_counter() - start
    except Exception as exc:  # connection refused, model not pulled, etc.
        print(f"Could not reach Ollama at http://localhost:11434.\n  {exc}")
        print("Is Ollama running (`ollama serve`) and is the model pulled "
              f"(`ollama pull {MODEL}`)?")
        sys.exit(1)

    print(f"Model:   {MODEL}")
    print(f"Prompt:  {PROMPT}")
    print("-" * 60)
    print(response.choices[0].message.content.strip())
    print("-" * 60)

    # Same response object shape as a hosted OpenAI/Gemini-compatible call.
    usage = response.usage
    if usage is not None:
        toks = usage.completion_tokens or 0
        tps = toks / elapsed if elapsed > 0 else 0.0
        print(f"{toks} completion tokens in {elapsed:.2f}s "
              f"(~{tps:.1f} tok/s) — generated locally, no network round-trip.")


if __name__ == "__main__":
    main()
