# Self-Hosting Report

Engine: **Ollama 0.30.8**. Host: Windows 11, models served on GPU (Ollama
reported `100% GPU`). Same prompt sent to both models:

> *"Explain what an inference engine is and why someone would self-host one,
> in about 4 sentences."*

Method: each model gets a **cold call** (first request — loads the weights
into memory, then generates) and a **warm call** (weights already resident,
so the time is pure generation). Tokens/sec is measured on the warm call
from `completion_tokens / generation_time`. RAM is the resident size Ollama
reports in `ollama ps`. Measured with `benchmark.py`.

## Task 1 — Benchmark two local models

| Model | Approx size / quant | Load time (s) | Tokens/sec | RAM used | Quality note |
|-------|---------------------|---------------|------------|----------|--------------|
| llama3.2:3b  | 3B / Q4_K_M (2.0 GB on disk)   | 45.9 (cold call) | 115.8      | 2.6 GB   | Accurate, well-structured 4-sentence answer; correctly framed self-hosting around control, data privacy, and maintenance. |
| qwen2.5:0.5b | 0.5B / Q4_K_M (397 MB on disk) | 24.7 (cold call) | 281.1      | 481 MB   | Much faster but looser: longer than asked, a slightly wrong definition ("a type of ML model"), and it hallucinated a trailing fake forum Q&A. |

**Trade-off you observed (2–3 sentences):**

> The 0.5B model is ~2.4× faster per token (281 vs 116 tok/s) and uses ~5×
> less memory (481 MB vs 2.6 GB), so it loads and responds almost instantly
> on a laptop. But that speed costs quality: it drifted off the definition,
> ignored the length limit, and tacked on a hallucinated forum exchange,
> while the 3B model stayed accurate and on-format. The practical rule:
> reach for the tiny model when latency/footprint dominate and the task is
> simple, and the 3B when the answer actually has to be right.

## Task 3 — VLM: local vs hosted

Image used: `sample_chart.png` (provided), committed in this repo. **No
Gemini API key was available**, so per the lab's stated fallback this
compares **two local VLMs** (moondream vs llava-phi3) instead of hosted
Gemini. Both run on Ollama on the same machine. Task performed: **caption**
(prompt `"Describe this image."`) with a factual follow-up
(`"What is the title and how many bars?"`).

Ground truth (what the chart actually says): title **"Inference Speed by
Model"**, subtitle "tokens per second (higher is faster)", **4 bars** —
Qwen2.5 0.5B = 98, Llama 3.2 3B = 61, Gemma 3 4B = 44, Llama 3.1 8B = 27.

| System | Answer (short) | Speed | Cost |
|--------|----------------|-------|------|
| Local VLM (moondream, 1.8B) | Caption: bar chart, 4 bars (blue/orange/green/red), title ≈ "Inference Speed by Model", y-axis misread as "Tasks per second". **Read no numeric values or model labels.** Returned an **empty string** to the pointed title/count question. | ~2–3 s (warm), 1.2 GB VRAM | free / local |
| Local VLM #2 (llava-phi3, 3.8B) | Caption: 4 bars, colors right, read bar **values** ≈ 84/61/43/27 (truth 98/61/44/27 — 3 of 4 close). But OCR'd title as "**Influence** Speed by Model" and **hallucinated** the x-labels as car models (Corsa, Cupra, Honda Civic). Answered the follow-up: title (misread) + "four bars" ✓. | ~6.5 s (warm), 3.9 GB VRAM | free / local |

**Comparison (2–3 sentences):**

> Neither tiny local VLM is trustworthy for reading this chart, but they
> fail differently. moondream is ~2–3× faster and lighter (1.2 vs 3.9 GB)
> and gets the gist (4 bars, near-correct title), but it reads no data
> values and silently returns an empty answer to direct questions — a
> dangerous failure mode. llava-phi3 is slower and heavier yet noticeably
> "tries harder": it actually estimated bar heights (3 of 4 within a few
> units) and always answers, but it OCR'd the title wrong and invented
> car-model labels that aren't in the image. Cost is identical here (both
> free/local, no per-token billing), which is exactly the trade you make
> versus a hosted model: zero marginal cost and full data privacy, paid for
> with weaker OCR/grounding than a frontier multimodal API would give.
