# Branch 7 — LLMOps, high-throughput serving & systems

Build an inference service whose memory use, latency, and failure modes you can explain.
LLMOps means the operational work that keeps a large language model service reliable, measurable, and safe to change.
This branch allocates exactly **7 hours**: **4 hours for 7.1** and **3 hours for 7.2**.
The result feeds **Project 4 — High-throughput agentic service** in the [project appendix](../index.md).

Start with the [teaching notes](notes.md).
Attempt the [problems](problems.md) before revealing their solutions.
Use the CPU exercises when you do not have a compatible GPU.
Do not treat a successful CPU exercise as a serving benchmark.

## Learning objectives

After this branch, you can:

- **7.1 — Derive** weight and key-value cache memory for 7B-, 8B-, and 70B-scale models in bytes and GiB.
- **7.1 — Calculate** a concurrency ceiling from a GPU memory budget and state which runtime allocations the calculation excludes.
- **7.1 — Explain** block allocation, continuous batching, chunked prefill, and prefix reuse with a request timeline.
- **7.1 — Compare** AWQ, GPTQ, dynamic quantization, and GGUF k-quants without confusing storage formats with algorithms.
- **7.1 — Estimate** speculative decoding speedup from acceptance probability and draft/verification costs.
- **7.2 — Select** an inference engine against a documented workload and hardware constraint.
- **7.2 — Implement** an OpenAI-style Server-Sent Events stream with cancellation cleanup.
- **7.2 — Measure** time to first token, time per output token, throughput, and queue depth without mixing their denominators.
- **7.2 — Trace** a model call and define admission controls for the Project 4 agent service.

## Prerequisites

- Read [Branch 4 notes, subtopic 4.2: transformer architecture, attention, and KV caching](../branch-04-transformers-nlp/notes.md).
  Revisit query heads, key/value heads, head dimension, and autoregressive decoding.
- Read [Branch 5 notes, subtopic 5.2: fine-tuning and quantization basics](../branch-05-llm-fine-tuning/notes.md).
  Revisit precision, scaling factors, low-bit weights, and the distinction between training and inference memory.
- Bring Python async/await and HTTP request/response knowledge.
  You need Python 3.11 or later for the CPU exercises.
- For GPU labs, bring a supported accelerator, a matching driver/runtime, and sufficient memory for the selected checkpoint.
  Use a separate environment for vLLM because its supported PyTorch/CUDA combination constrains installation.

## Hour budget

| Subtopic | Hours | Evidence you produce |
|---|---:|---|
| 7.1 Inference mechanics & memory bottlenecks | 4 | Memory worksheet, paging explanation, quantization decision, speculation estimate |
| 7.2 Serving engines & production stacks | 3 | Engine choice, working SSE stream, trace, concurrency report |
| **Total** | **7** | **4 + 3 = 7 hours** |

## Seven-session schedule

Each session lasts one hour.
Video time replaces part of the reading time; it does not add hours.
The longer Project 4 build remains outside the 60-hour core syllabus.

| Hour | Subtopic | Study and practice | Completion signal |
|---|---|---|---|
| 1 | 7.1 | Weights, activations, KV cache; run the memory calculator; attempt A1–A3 | Recompute 128 KiB/token and 35 GB of INT4 weights |
| 2 | 7.1 | Watch the vLLM talk; trace block tables and continuous batching; attempt A4 | Explain why a freed block can serve another request |
| 3 | 7.1 | Watch the quantization video; compare AWQ/GPTQ, dynamic precision, and GGUF; attempt B1 | Distinguish raw weight bytes from total residency |
| 4 | 7.1 | Watch the SGLang talk; study speculation, chunking, and prefix reuse; attempt A5–A6 | Compute expected tokens per verification step |
| 5 | 7.2 | Compare engines; start vLLM; implement the streaming endpoint; attempt B2 | Observe incremental chunks and a terminal event |
| 6 | 7.2 | Watch the Langfuse walkthrough; instrument a call; attempt B3 and A7 | Identify a generation span and its parent request |
| 7 | 7.2 | Run the load generator at three concurrency levels; attempt B4 and A8; review | Report TTFT, TPOT, throughput, and admission policy |

## Subtopic map

| Subtopic | Notes | Practice |
|---|---|---|
| 7.1 Memory accounting | [Memory footprint accounting](notes.md#memory-footprint-accounting) | A1–A3, B1 |
| 7.1 Cache management | [PagedAttention and cache management](notes.md#pagedattention-and-cache-management) | A4 |
| 7.1 Quantization | [Quantization algorithms and formats](notes.md#quantization-algorithms-and-formats) | A2, B1 |
| 7.1 Decoding optimizations | [Speculative decoding](notes.md#speculative-decoding), [Chunked prefill and prefix caching](notes.md#chunked-prefill-and-prefix-caching) | A5–A6 |
| 7.2 Engine selection | [Production engine comparison](notes.md#production-engine-comparison) | A8 |
| 7.2 Streaming and gateways | [Streaming with SSE](notes.md#streaming-with-sse), [Gateways and rate limiting](notes.md#gateways-and-rate-limiting) | B2 |
| 7.2 Operations | [Observability and tracing](notes.md#observability-and-tracing), [Production metrics and load testing](notes.md#production-metrics-and-load-testing) | A7, B3–B4 |

## Video playlist

Titles and channels match YouTube's oEmbed metadata.
Durations come from YouTube's video search metadata.
Links and titles were verified on **2026-10-16**.
Watch these four videos in the scheduled hours.

| Title | Channel | Duration | Why it earns study time |
|---|---|---|---|
| [Fast LLM Serving with vLLM and PagedAttention](https://www.youtube.com/watch?v=5ZlavKF_98U) | Anyscale | 32:07 | Connect the original paging design to memory waste and serving throughput in hour 2. |
| [Which Quantization Method is Right for You? (GPTQ vs. GGUF vs. AWQ)](https://www.youtube.com/watch?v=mNE_d-C82lI) | Maarten Grootendorst | 15:51 | Build an algorithm-versus-format decision framework in hour 3. |
| [Efficient LLM Inference with SGLang, Lianmin Zheng, xAI](https://www.youtube.com/watch?v=Ny4xxErgFgQ) | AMD Developer Central | 24:37 | Relate shared-prefix workloads to runtime design in hour 4. |
| [10 min Walkthrough of Langfuse – Open Source LLM Observability, Evaluation, and Prompt Management](https://www.youtube.com/watch?v=2E8iTvGo9Hs) | Langfuse | 10:10 | Identify the trace, observation, and evaluation views you need in hour 6. |

## Embedded playlist

Play the four scheduled videos here when you use the built documentation site.
Use the linked table above when your Markdown viewer does not support embedded players.

<iframe width="560" height="315" src="https://www.youtube.com/embed/5ZlavKF_98U" title="Fast LLM Serving with vLLM and PagedAttention — Anyscale" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/mNE_d-C82lI" title="Which Quantization Method is Right for You? (GPTQ vs. GGUF vs. AWQ) — Maarten Grootendorst" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/Ny4xxErgFgQ" title="Efficient LLM Inference with SGLang, Lianmin Zheng, xAI — AMD Developer Central" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/2E8iTvGo9Hs" title="10 min Walkthrough of Langfuse – Open Source LLM Observability, Evaluation, and Prompt Management — Langfuse" frameborder="0" allowfullscreen></iframe>

## Key sources

Read the paper sections and API pages that answer the question in front of you.
Do not attempt every source from start to finish within seven hours.
All links below pass a live HTTP check.
The book entry uses the author's books page because the supplied dedicated book path returns 404.

| Author or organization | Title | Type | Link | Covers | Verification |
|---|---|---|---|---|---|
| Kwon et al. | Efficient Memory Management for Large Language Model Serving with PagedAttention | Primary paper, 2023 | [arXiv 2309.06180](https://arxiv.org/abs/2309.06180) | 7.1: allocation waste, paging, sharing | accessed 2026-10-16 |
| vLLM project | vLLM documentation | Official documentation | [Documentation](https://docs.vllm.ai/) | 7.1–7.2: engine, scheduling, quantization, serving | accessed 2026-10-16 |
| Meta | Llama 3 reference model implementation | Primary source code | [Model source](https://github.com/meta-llama/llama3/blob/main/llama/model.py) | 7.1: GQA and head/cache tensor shapes | accessed 2026-10-16 |
| Frantar et al. | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | Primary paper, 2022 | [arXiv 2210.17323](https://arxiv.org/abs/2210.17323) | 7.1: second-order weight quantization | accessed 2026-10-16 |
| Lin et al. | AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | Primary paper, 2023 | [arXiv 2306.00978](https://arxiv.org/abs/2306.00978) | 7.1: activation-aware scaling | accessed 2026-10-16 |
| Leviathan, Kalman, and Matias | Fast Inference from Transformers via Speculative Decoding | Primary paper, 2022 preprint | [arXiv 2211.17192](https://arxiv.org/abs/2211.17192) | 7.1: acceptance correction and expected progress | accessed 2026-10-16 |
| Zheng et al. | SGLang: Efficient Execution of Structured Language Model Programs | Primary paper, 2023 preprint | [arXiv 2312.07104](https://arxiv.org/abs/2312.07104) | 7.1–7.2: RadixAttention and runtime design | accessed 2026-10-16 |
| ggml-org | llama.cpp | Official source and documentation | [Repository](https://github.com/ggml-org/llama.cpp) | 7.1: GGUF deployment and k-quants | accessed 2026-10-16 |
| Hugging Face | bitsandbytes quantization | Official documentation | [Quantization guide](https://huggingface.co/docs/transformers/quantization/bitsandbytes) | 7.1: 8-bit and 4-bit loading | accessed 2026-10-16 |
| Tim Dettmers | Hardware and quantization articles | Author's technical blog | [Blog](https://timdettmers.com/) | 7.1: hardware context and low-precision tradeoffs | accessed 2026-10-16 |
| NVIDIA | TensorRT-LLM | Official source and documentation | [Repository](https://github.com/NVIDIA/TensorRT-LLM) | 7.2: NVIDIA-focused optimized inference | accessed 2026-10-16 |
| Hugging Face | Text Generation Inference | Official documentation | [Documentation](https://huggingface.co/docs/text-generation-inference) | 7.2: serving architecture and maintenance status | accessed 2026-10-16 |
| SGLang project | SGLang | Official source and documentation | [Repository](https://github.com/sgl-project/sglang) | 7.2: runtime and supported deployments | accessed 2026-10-16 |
| MDN contributors | Using server-sent events | Protocol documentation | [SSE guide](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) | 7.2: event framing and stream behavior | accessed 2026-10-16 |
| OpenTelemetry | Documentation | Official documentation | [Overview](https://opentelemetry.io/docs/) | 7.2: traces, spans, and propagation | accessed 2026-10-16 |
| OpenTelemetry | Python instrumentation | Official documentation | [Python guide](https://opentelemetry.io/docs/languages/python/instrumentation/) | 7.2: tracer provider and span creation | accessed 2026-10-16 |
| Langfuse | Documentation | Official documentation | [Overview](https://langfuse.com/docs) | 7.2: LLM observability and evaluation | accessed 2026-10-16 |
| Langfuse | SDK overview | Official documentation | [SDK guide](https://langfuse.com/docs/observability/sdk/overview) | 7.2: generation observations and Python SDK | accessed 2026-10-16 |
| Arize | Phoenix documentation | Official documentation | [Phoenix](https://arize.com/docs/phoenix) | 7.2: OpenTelemetry-based LLM tracing and evaluation | accessed 2026-10-16 |
| Chip Huyen | Designing Machine Learning Systems | Author's book listing | [Books](https://huyenchip.com/books/) | 7.2: operational design and monitoring context | accessed 2026-10-16 |

## Exit criteria

- [ ] Recompute weight storage for all four precisions without confusing GB and GiB.
- [ ] Derive 131,072 KV bytes per token for the specified 8B-scale architecture.
- [ ] Explain why 70B INT4 weights alone exceed a 24 GB device.
- [ ] Calculate an illustrative batch ceiling and identify allocator, activation, and workspace reserves.
- [ ] Explain block tables, copy-on-write, and iteration-level admission.
- [ ] Describe one quality risk for each quantization option.
- [ ] Compute expected speculative tokens per step and include verification/draft costs.
- [ ] Stream content through FastAPI without buffering the entire model response.
- [ ] State which engine you select and why another engine is unsuitable for your constraints.
- [ ] Run the CPU coding exercises and reproduce their stated outputs.
- [ ] Run the GPU labs when hardware is available and record the actual model/runtime versions.
- [ ] Explain TTFT, TPOT, aggregate throughput, and queue depth from one workload report.
- [ ] Score at least 70% on Part A and reproduce every Part B reference behavior.
- [ ] Carry a memory budget, trace schema, and admission policy into Project 4.

Return to the [syllabus](../index.md) or continue to [Branch 8](../branch-08-ai-safety/index.md).
