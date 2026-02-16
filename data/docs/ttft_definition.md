# Time To First Token (TTFT) in Large Language Model Serving

## Introduction

Time To First Token (TTFT) is a key metric in LLM serving systems. It measures the time between submitting a request and receiving the first generated token.

TTFT significantly impacts user experience in interactive applications.

---

## Definition

TTFT = Time from request submission → first output token emission.

It does not measure full generation time.

---

## Why TTFT Matters

In chat systems:

- Users perceive responsiveness based on initial output.
- Even if total generation takes longer, early feedback improves UX.

Lower TTFT results in:

- Faster perceived response
- Higher user satisfaction
- More interactive system feel

---

## Factors Affecting TTFT

- Prompt length
- Model size
- GPU memory availability
- Engine optimization
- Kernel launch overhead
- Batch queue delay (in Triton)

---

## TTFT vs Total Latency

Total latency measures:

Request → full output completion

TTFT measures:

Request → first token only

Both must be reported in benchmarks.

---

## TTFT and Optimized Inference

TensorRT-LLM reduces TTFT by:

- Eliminating graph interpretation
- Fusing kernels
- Optimizing memory access

---

## Routing Implications

Routing systems may:

- Prefer optimized backend for short prompts to reduce TTFT
- Use fallback for memory-heavy requests

---

## Conclusion

TTFT is a critical metric in LLM serving. Optimized inference frameworks significantly improve TTFT, making it a central benchmark for evaluating GPU-aware routing systems.
