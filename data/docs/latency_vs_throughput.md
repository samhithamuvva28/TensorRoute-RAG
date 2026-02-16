# Latency vs Throughput in LLM Serving Systems

## Definitions

Latency:
Total time required to generate a response.

Throughput:
Number of tokens generated per second.

---

## TTFT vs Latency

TTFT measures responsiveness.
Total latency measures completion time.

---

## Throughput Optimization

Increasing batch size increases throughput.
However, it may increase latency.

---

## Tradeoffs in Production

- Real-time chat → prioritize latency
- Batch processing → prioritize throughput

---

## GPU Constraints

Memory and compute limits influence achievable throughput.

---

## Routing Implications

Routing systems may:

- Choose optimized engines for throughput
- Choose baseline engines for memory stability

---

## Conclusion

Balancing latency and throughput is central to LLM serving optimization.
