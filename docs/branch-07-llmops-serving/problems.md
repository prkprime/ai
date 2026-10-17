# Branch 7 problems — LLMOps, high-throughput serving & systems

Attempt these exercises after the corresponding [teaching notes](notes.md).
Write down units, assumptions, and denominators before you calculate.
Open each solution only after you produce an answer.
Difficulty uses ● for direct application, ●● for multiple interacting steps, and ●●● for design or diagnosis.

There are **12 problems: 8 pen-and-paper and 4 coding**.
The CPU reference programs do not require a model download.
The serving extension requires the local vLLM and FastAPI processes described in the notes.
Save code into your own lab directory rather than modifying this syllabus.

## Part A — Pen and paper

### A1 — Derive the cache payload ●

An 8B-scale decoder has 32 layers, 32 query heads, eight KV heads, and a head dimension of 128.
It stores keys and values in FP16 at two bytes per scalar.
Assume full attention, no prefix sharing, and no block rounding.

1. Derive bytes per cached token per sequence.
2. Calculate memory at 4,096 cached tokens for one sequence and eight sequences.
3. Replace the eight KV heads with 32 and calculate the reduction factor provided by GQA.
4. Repeat bytes per token for a 70B-scale model with 80 layers and eight KV heads.

<details markdown="1">
<summary>Solution</summary>

First count the stored scalar values.
Each token stores one key and one value for every layer, KV head, and head coordinate.
The leading factor of two counts the key and value tensors.

$$
c=2\times32\times8\times128\times2=131{,}072\text{ bytes/token}.
$$

Divide by 1,024 to convert bytes to KiB:

$$
131{,}072/1024=128\text{ KiB/token}.
$$

For one sequence, multiply by 4,096 positions:

$$
131{,}072\times4096=536{,}870{,}912\text{ bytes}=0.5\text{ GiB}.
$$

For eight equal-length sequences, multiply by eight:

$$
536{,}870{,}912\times8=4{,}294{,}967{,}296\text{ bytes}=4\text{ GiB}.
$$

With 32 KV heads, bytes per token become 524,288 and the eight-sequence cache becomes 16 GiB.
The original GQA model keeps $8/32=1/4$ of that KV payload.
Its memory reduction factor is therefore four.
The number of query heads does not enter the GQA cache formula directly.

For the 70B-scale configuration:

$$
2\times80\times8\times128\times2=327{,}680\text{ bytes/token}=320\text{ KiB/token}.
$$

At 4,096 positions, that is 1,342,177,280 bytes, or 1.25 GiB per sequence.
All results exclude allocator overhead, scales, sharing, and temporary activations.

</details>

### A2 — Separate payload from deployment memory ●●

Treat a 70B model as exactly 70 billion parameters.
Calculate its raw weight payload in GB and GiB for FP32, FP16, INT8, and packed INT4.
Then assume the INT4 representation stores one FP16 scale for every 128 parameters, with no zero point or padding.
Calculate the scale overhead and revised weight payload.
Decide whether it fits on a 24 GB GPU without offload.

<details markdown="1">
<summary>Solution</summary>

Multiply parameter count by bytes per stored value.
Convert GB using $10^9$ and GiB using $2^{30}$.

| Precision | Calculation in bytes | GB | GiB |
|---|---:|---:|---:|
| FP32 | $70\times10^9\times4$ | 280 | 260.77 |
| FP16 | $70\times10^9\times2$ | 140 | 130.39 |
| INT8 | $70\times10^9\times1$ | 70 | 65.19 |
| INT4 | $70\times10^9\times0.5$ | 35 | 32.60 |

Count scale groups:

$$
70{,}000{,}000{,}000/128=546{,}875{,}000\text{ groups}.
$$

Each FP16 scale takes two bytes:

$$
546{,}875{,}000\times2=1{,}093{,}750{,}000\text{ bytes}=1.09375\text{ GB}.
$$

Add scales to packed weights:

$$
35+1.09375=36.09375\text{ GB}\approx33.61\text{ GiB}.
$$

The scales add $1.09375/35=3.125\%$ over the raw packed payload.
Even the raw 35 GB does not fit in 24 GB.
The revised payload cannot fit either, before considering KV cache or workspace.
The assumed metadata layout is an arithmetic exercise, not a claim that every AWQ, GPTQ, or GGUF artifact uses this exact layout.

</details>

### A3 — Admit a batch with an explicit reserve ●●

A hypothetical device budget contains exactly 24,000,000,000 bytes.
Measured weights occupy 14,000,000,000 bytes.
Reserve 2,000,000,000 bytes for non-KV runtime allocations.
Use A1's 131,072-byte cache cost per token and 4,096 cached tokens per sequence.

1. Calculate the available KV bytes.
2. Find the maximum equal-length batch size under these assumptions.
3. Show why the next batch size fails.
4. Explain how the result changes if you mistakenly read 24 GB as 24 GiB while keeping the other byte figures fixed.

<details markdown="1">
<summary>Solution</summary>

Subtract persistent and reserved allocations first:

$$
M_{kv}=24{,}000{,}000{,}000-14{,}000{,}000{,}000-2{,}000{,}000{,}000=8{,}000{,}000{,}000.
$$

One sequence requires:

$$
131{,}072\times4096=536{,}870{,}912\text{ bytes}.
$$

Take the floor because you cannot admit a fractional complete sequence:

$$
B_{\max}=\lfloor8{,}000{,}000{,}000/536{,}870{,}912\rfloor=14.
$$

Fourteen sequences use 7,516,192,768 bytes and leave 483,807,232 bytes in the KV budget.
Fifteen sequences use 8,053,063,680 bytes and exceed it by 53,063,680 bytes.
Therefore 14 is the largest admitted equal-length batch in this simplified model.

If you change only device capacity to 24 GiB, the capacity becomes 25,769,803,776 bytes.
The remaining KV budget becomes 9,769,803,776 bytes.
Dividing by 536,870,912 and taking the floor gives 18 sequences.
The unit error therefore over-admits four sequences relative to the stated 24 GB device.

The measured 14 GB weight allocation is a supplied assumption, not the exact-8B FP16 table entry.
Production admission must also enforce growth limits and account for cache-block metadata and scheduler policy.

</details>

### A4 — Follow a block table ●●

An engine uses 16-token blocks and the 8B-scale FP16 cache from A1.
Request A contains 35 cached tokens.
Its block table maps logical blocks 0, 1, and 2 to physical blocks 7, 2, and 11.

1. Calculate the allocated cache bytes and unused token slots.
2. Identify the logical block and within-block offset for token position 34, using zero-based indexing.
3. Identify its physical block.
4. Explain why requests sharing the first two physical blocks cannot both overwrite the same partial third block.

<details markdown="1">
<summary>Solution</summary>

Round the sequence length up to whole blocks:

$$
n_{blocks}=\lceil35/16\rceil=3.
$$

One physical block costs:

$$
16\times131{,}072=2{,}097{,}152\text{ bytes}=2\text{ MiB}.
$$

Three blocks cost 6,291,456 bytes, or 6 MiB.
They allocate 48 token slots, so $48-35=13$ slots remain unused.
The live-token payload is $35\times131{,}072=4{,}587{,}520$ bytes, or 4.375 MiB.

For token position 34, use quotient and remainder:

$$
34=2\times16+2.
$$

The logical block is 2 and the within-block offset is 2.
The table maps logical block 2 to physical block 11.
The kernel reads that token's keys and values from position 2 in physical block 11 for each layer.

Shared full prefix blocks can remain immutable and use reference counts.
If requests diverge while sharing a partial block, the writer needs a private block before appending different KV values.
Copy-on-write preserves the other request's prefix state.
Noncontiguous allocation does not relax correctness or isolation requirements.

</details>

### A5 — Estimate speculative progress and speedup ●●

A draft proposes four tokens per verification step.
Each next proposed token has conditional acceptance probability 0.8.
Ignore end-of-sequence termination.
After the accepted draft prefix, the algorithm emits one correction token or one extra target token.
Ordinary target decoding costs 20 ms/token.
Draft decoding costs 2 ms/token, and verification costs 20 ms/step.

Calculate expected accepted draft tokens, total emitted tokens per step, mean time per emitted token, and speedup.
Then find the speedup if verification instead costs 40 ms.

<details markdown="1">
<summary>Solution</summary>

The probability of reaching and accepting draft token $i$ is $\alpha^i$ under the stated constant conditional acceptance model.
Sum these probabilities for the four proposed positions:

$$
E[N_{draft}]=0.8+0.8^2+0.8^3+0.8^4=2.3616.
$$

Add the correction or extra target token:

$$
E[N]=1+2.3616=3.3616.
$$

The draft stage costs $4\times2=8$ ms.
The complete speculative step costs $8+20=28$ ms.
Divide cost by expected progress:

$$
28/3.3616\approx8.33\text{ ms/token}.
$$

Compare with ordinary target decoding:

$$
\mathrm{speedup}=20/(28/3.3616)=3.3616\times20/28\approx2.40.
$$

With 40 ms verification, the step costs 48 ms.
The speedup falls to $3.3616\times20/48\approx1.40$.
Acceptance rate alone does not determine speedup.
Real measurements must include scheduling overhead, draft residency, and workload-dependent verification cost.

</details>

### A6 — Distinguish prefix reuse from chunking ●●

Two requests share an exact 2,048-token prefix but have different 512-token suffixes.
Assume a cache hit reuses the entire shared prefix.
For arithmetic only, assume linear prefill work at 4,096 tokens/s and ignore queueing and lookup costs.

1. Calculate the second request's prefill time with and without a cache hit.
2. Calculate the saving.
3. Explain whether this removes decode attention over the prefix.
4. Explain how chunked prefill addresses a different latency problem.
5. State two reasons the apparently shared prefix might not be reusable.

<details markdown="1">
<summary>Solution</summary>

Without reuse, the second request processes all $2048+512=2560$ prompt tokens.
Its simplified prefill time is $2560/4096=0.625$ s.
With a hit, it processes only 512 new prompt tokens, costing $512/4096=0.125$ s.
The saving is $0.625-0.125=0.5$ s.

This calculation models prefill work only.
Attention during subsequent decode still reads the relevant cached prefix keys and values.
The request also pays any queue, lookup, network, and generation costs.
Thus the 500 ms saving is not a guarantee about total response latency.

Chunked prefill splits prompt processing into scheduling units.
It lets the runtime interleave a long prefill with existing decode requests.
It can reduce pauses for existing streams even when no prefix is reusable.
Prefix reuse instead avoids repeated work for matching prefixes.

Reuse can fail because the token IDs differ after whitespace or template changes.
It can fail because the model revision or adapter differs.
It can also fail because blocks were evicted or tenant isolation forbids sharing.
Any two of these reasons suffice.

</details>

### A7 — Compute latency without mixing denominators ●

A request starts at time 0.
It receives first content at 300 ms and its last content token at 2,300 ms.
The engine reports 101 output tokens.
The final usage event arrives at 2,340 ms.

1. Calculate TTFT and decode TPOT using the notes' definitions.
2. Calculate last-content latency and response-completion latency.
3. Explain why $2340/101$ is not decode TPOT.
4. A 10-second run completes 20 such requests and generates 2,020 tokens; calculate request and output throughput.

<details markdown="1">
<summary>Solution</summary>

TTFT is the first-content time minus request submission time:

$$
\mathrm{TTFT}=300-0=300\text{ ms}.
$$

There are 100 intervals between 101 tokens.
Use first and last content timestamps:

$$
\mathrm{TPOT}=(2300-300)/(101-1)=2000/100=20\text{ ms}.
$$

Last-content latency is 2,300 ms.
Response-completion latency is 2,340 ms under the stated final-event convention.
The additional 40 ms does not describe token generation intervals.
The quotient $2340/101\approx23.17$ ms includes TTFT and trailing metadata time and uses the wrong denominator for decode TPOT.

For the complete workload interval:

$$
\mathrm{request\ throughput}=20/10=2\text{ requests/s},
$$

$$
\mathrm{output\ throughput}=2020/10=202\text{ output tokens/s}.
$$

These throughput figures do not imply that a single stream produces 202 tokens/s.
They aggregate all completed work in the stated interval.
Measure error and rejection counts separately.

</details>

### A8 — Diagnose an overloaded serving stack ●●●

Your NVIDIA deployment serves a shared-prefix agent workload.
At concurrency 4, output throughput is 400 tokens/s, p95 TTFT is 0.5 s, and queue depth stays near zero.
At concurrency 16, throughput is 460 tokens/s, p95 TTFT is 6 s, and queue depth grows throughout the run.
Your objective requires p95 TTFT below 1 s.

Choose an initial engine candidate, a comparison candidate, and an admission policy.
Explain why the second load point is not sustainable capacity evidence.
State what you trace without storing prompt contents.
Finally, a token bucket has capacity 100 credits, contains 20 credits, and refills at 10 credits/s.
How many credits remain after three idle seconds and admission of a 40-credit request?

<details markdown="1">
<summary>Solution</summary>

vLLM is a reasonable initial candidate because it supports the required model-serving interface and iteration-level batching.
SGLang is a useful comparison candidate because the workload exposes reusable prefixes.
TensorRT-LLM is also defensible if NVIDIA-specific optimization and the supported model path justify its operational cost.
Name the hardware/model compatibility checks rather than claiming an engine wins before measurement.

The concurrency-16 run increases throughput only 15% while violating the TTFT objective by a factor of six.
A growing queue means arrival work exceeds completed service over that interval.
Do not report 460 tokens/s as capacity that satisfies the objective.
Start with a bounded admission level near the compliant measured region, then refine it with more samples and workload distributions.

Apply a tenant token budget, a shared concurrency cap, and a finite queue with a documented overload response.
Also limit whole-agent tokens, tools, retries, and elapsed time.
Trace request, retrieval, generation, validation, and tool spans.
Record model version, status, usage, finish reason, and timing with redacted identifiers.
Collect engine telemetry to distinguish queueing from prefill rather than inferring both from a client span.

After three seconds, the bucket contains $\min(100,20+3\times10)=50$ credits.
The 40-credit request can enter and leaves $50-40=10$ credits.
In a replicated gateway, perform this update atomically in shared state or partition the global budget explicitly.

</details>

## Part B — Code

### B1 — Implement a byte-accurate capacity report ●●

Write a CPU program that calculates KV bytes/token and a batch ceiling.
It accepts positive integer cache dimensions, context length, and a nonnegative KV budget.
It rejects invalid dimensions instead of silently producing a negative capacity.
For the A3 budget, print the cache rate, per-sequence allocation, maximum batch, and remaining bytes.
Then explain why this calculator is not a substitute for a runtime memory profile.

<details markdown="1">
<summary>Solution</summary>

Use integer arithmetic so byte boundaries remain explicit.
The public function returns a report rather than changing an engine configuration.
Run this program with `uv run --no-project python capacity.py`.

```python
def capacity(layers, kv_heads, head_dim, scalar_bytes, context, kv_budget):
    dimensions = (layers, kv_heads, head_dim, scalar_bytes, context)
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in dimensions):
        raise ValueError("Cache dimensions must be positive integers")
    if not isinstance(kv_budget, int) or isinstance(kv_budget, bool) or kv_budget < 0:
        raise ValueError("KV budget must be a nonnegative integer")
    per_token = 2 * layers * kv_heads * head_dim * scalar_bytes
    per_sequence = per_token * context
    batch, remaining = divmod(kv_budget, per_sequence)
    return {
        "bytes_per_token": per_token,
        "bytes_per_sequence": per_sequence,
        "batch": batch,
        "remaining_bytes": remaining,
    }

if __name__ == "__main__":
    report = capacity(32, 8, 128, 2, 4096, 8_000_000_000)
    for key, value in report.items():
        print(f"{key}={value}")
    try:
        capacity(32, 0, 128, 2, 4096, 8_000_000_000)
    except ValueError as error:
        print(error)
```

Expected output:

```text
bytes_per_token=131072
bytes_per_sequence=536870912
batch=14
remaining_bytes=483807232
Cache dimensions must be positive integers
```

The report assumes a fixed byte cost per cached scalar and equal full sequence lengths.
It omits block-table overhead and assumes that the supplied KV budget already excludes all other memory.
Quantized KV scales, hybrid attention layouts, prefix sharing, and runtime workspaces require a more specific model.
Use the engine's memory profile before converting the result into an admission setting.

</details>

### B2 — Parse an OpenAI-style SSE stream ●●

Implement a CPU parser for the one-JSON-data-line-per-event contract used by the notes' FastAPI proxy.
Ignore comments and empty lines.
Skip role-only chunks and usage-only chunks.
Concatenate nonempty content and require a terminal `[DONE]` marker.
Reject an explicit error event and a truncated stream.
Then run the notes' real endpoint and verify incremental output with curl.

<details markdown="1">
<summary>Solution</summary>

This parser consumes decoded lines, not arbitrary TCP byte fragments.
Use `httpx.Response.aiter_lines()` or another line decoder before applying this contract to network data.
It deliberately does not implement general multi-line SSE data fields.
Run the standalone reference with `uv run --no-project python sse_decode.py`.

```python
import json

def decode_chat(lines):
    pieces = []
    for line in lines:
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            return "".join(pieces)
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(event["error"]["message"])
        for choice in event.get("choices", []):
            content = choice.get("delta", {}).get("content")
            if content:
                pieces.append(content)
    raise RuntimeError("Stream ends without [DONE]")

sample = [
    ": keepalive",
    'data: {"choices":[{"delta":{"role":"assistant"}}]}',
    "",
    'data: {"choices":[{"delta":{"content":"KV "}}]}',
    "",
    'data: {"choices":[{"delta":{"content":"cache"}}]}',
    "",
    'data: {"choices":[],"usage":{"completion_tokens":3}}',
    "",
    "data: [DONE]",
    "",
]
print(decode_chat(sample))
try:
    decode_chat(sample[:-3])
except RuntimeError as error:
    print(error)
try:
    decode_chat(['data: {"error":{"message":"Model stream interrupted"}}'])
except RuntimeError as error:
    print(error)
```

Expected output:

```text
KV cache
Stream ends without [DONE]
Model stream interrupted
```

For the network portion, use the full [FastAPI reference endpoint](notes.md#runnable-snippet-5-fastapi-streaming-proxy).
Its asynchronous generator provides the reference solution for actual streaming and upstream cleanup.
Run the [curl consumer](notes.md#runnable-snippet-6-consume-the-stream).
Observe at least one nonempty content delta, a finish-reason event, and a `[DONE]` marker on success.
Cancel a second request and inspect the engine's active-request count until the generation is released.
Do not claim the CPU sample alone verifies cancellation across the HTTP boundary.

</details>

### B3 — Export a nested trace without user text ●●

Use OpenTelemetry to emit a request span and a child generation span.
Attach model identity and integer input/output token counts.
Do not attach prompt or answer text.
Use a console exporter so the exercise runs without a GPU, a paid API, or a telemetry account.
Then replace the synthetic work with the real model call from the notes when hardware is available.

<details markdown="1">
<summary>Solution</summary>

Install the existing public SDK packages in your lab environment:

```bash
uv pip install opentelemetry-api opentelemetry-sdk
```

The simulated usage values exercise instrumentation only.
They are labeled as synthetic and must not enter a production usage report.
Save this reference as `trace_cpu.py` and run it with that environment's Python.

```python
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider(resource=Resource.create({"service.name": "branch07-cpu"}))
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("branch07.exercise")

try:
    with tracer.start_as_current_span("agent.request"):
        with tracer.start_as_current_span("llm.generate") as generation:
            generation.set_attribute("lab.synthetic", True)
            generation.set_attribute("lab.model", "cpu-demonstration")
            generation.set_attribute("lab.input_tokens", 12)
            generation.set_attribute("lab.output_tokens", 8)
    print("Trace exported")
finally:
    provider.shutdown()
```

Expected observable behavior:

- The console exporter prints two span objects and then `Trace exported`.
- Both span objects contain the same trace ID.
- The child span's `parent_id` equals the request span's span ID.
- The generation span contains integer token counts 12 and 8 and `lab.synthetic: true`.
- Neither span includes a prompt or an answer field.

IDs and timestamps vary, so do not compare them against fixed literals.
The child span ends and exports before the parent because its context manager exits first.
Use the [real OpenTelemetry model-call reference](notes.md#runnable-snippet-7-opentelemetry-console-trace) for the hardware extension.
Use the [Langfuse generation reference](notes.md#runnable-snippet-8-langfuse-generation-observation) if your backend is configured.

</details>

### B4 — Summarize a benchmark without counting chunks as tokens ●●●

Write a CPU report function for successful requests with submission, first-content, last-content, completion, and output-token values.
Use seconds for input timestamps.
Calculate nearest-rank p95 TTFT, mean per-request TPOT, total output throughput, and successful request throughput.
Exclude requests with one output token from the TPOT mean, but include their tokens in throughput.
Given the sample below, reproduce the exact formatted report.
Then run the notes' real load generator at concurrency 1, 4, and 8.

<details markdown="1">
<summary>Solution</summary>

The reference uses server-reported output-token counts rather than the number of SSE chunks.
It defines the workload interval explicitly as 0 to 4 seconds.
It does not infer a server queue depth from client timings.
Run it with `uv run --no-project python report.py`.

```python
from math import ceil

def summarize(rows, start, end):
    if not rows or end <= start:
        raise ValueError("Provide requests and a positive workload interval")
    ttfts = []
    tpots = []
    tokens = 0
    for row in rows:
        submitted, first, last, completed, count = row
        if not (start <= submitted <= first <= last <= completed <= end):
            raise ValueError("Invalid timestamp order or workload boundary")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("Output token count must be a positive integer")
        ttfts.append(first - submitted)
        if count > 1:
            tpots.append((last - first) / (count - 1))
        tokens += count
    ttfts.sort()
    p95 = ttfts[ceil(0.95 * len(ttfts)) - 1]
    mean_tpot = sum(tpots) / len(tpots) if tpots else None
    return p95, mean_tpot, tokens / (end - start), len(rows) / (end - start)

rows = [
    (0.0, 0.3, 2.3, 2.34, 101),
    (0.5, 0.9, 2.9, 2.95, 101),
    (1.0, 1.2, 1.2, 1.25, 1),
]
p95, mean_tpot, token_rate, request_rate = summarize(rows, 0.0, 4.0)
print(f"p95_ttft_ms={p95 * 1000:.1f}")
print(f"mean_tpot_ms={mean_tpot * 1000:.1f}")
print(f"output_tokens_s={token_rate:.2f}")
print(f"requests_s={request_rate:.2f}")
```

Expected output:

```text
p95_ttft_ms=400.0
mean_tpot_ms=20.0
output_tokens_s=50.75
requests_s=0.75
```

The sorted TTFT values are 0.2, 0.3, and 0.4 seconds.
Nearest-rank p95 selects rank $\lceil0.95\times3\rceil=3$, or 0.4 seconds.
The two multi-token requests each have $2/100=0.02$ seconds TPOT.
The single-token request has no inter-token interval.
Throughput is $(101+101+1)/4=50.75$ output tokens/s and $3/4=0.75$ requests/s.

For the hardware extension, use the full [bounded load generator](notes.md#runnable-snippet-9-bounded-load-generator) as the reference implementation.
Expected behavior is three measured rows, not fixed performance numbers.
Record runtime versions, successful requests, output counts, errors, and the warmup policy.
Read queue depth from the engine's metrics during the run and attach those observations separately.
If a stream fails or lacks usage, report that failure instead of inventing a token count.

</details>

## Review your work

Recompute A1, A3, A5, and A7 on a blank page.
Explain every factor in the KV formula aloud or in a short written paragraph.
Run all four CPU references before comparing your output with the solutions.
Run the hardware extensions when a supported GPU is available and retain the actual measured report.
Do not invent GPU benchmark numbers when you complete only the CPU path.

Use the [exit criteria](index.md#exit-criteria) to decide whether you can proceed to Project 4.
