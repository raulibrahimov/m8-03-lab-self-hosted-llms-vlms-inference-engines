"""
Task 3 helper — run the LOCAL vision-language model (moondream) on an image.

We use Ollama's native /api/chat endpoint with a base64-encoded image,
because the OpenAI-compatible image_url path is not consistently supported
for local VLMs. Same idea either way: image + question -> text answer.

Run:  python vlm_local.py
"""

import base64
import time

import requests

IMAGE = "sample_chart.png"
# Caption task. NOTE: moondream returns an empty string for pointed factual
# questions about this chart (e.g. "what is the title?" / "how many bars?"),
# but answers an open "describe this image" prompt — see the report.
QUESTION = "Describe this image."
MODEL = "moondream"


def main() -> None:
    with open(IMAGE, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("utf-8")

    t0 = time.perf_counter()
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": QUESTION, "images": [b64]}],
            "stream": False,
        },
        timeout=300,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()

    print(f"Model:    {MODEL}")
    print(f"Image:    {IMAGE}")
    print(f"Question: {QUESTION}")
    print("-" * 60)
    print(data["message"]["content"].strip())
    print("-" * 60)
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
