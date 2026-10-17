# Branch 5 problems — compute, adapters, and preferences

Attempt each problem before revealing its solution.
Use natural logarithms unless a problem specifies otherwise.
Report memory in decimal GB unless the question requests GiB.
The [notes](notes.md) define each objective and its assumptions.

Difficulty: **●** direct application, **●●** several connected steps, **●●●** diagnosis or implementation.
Part A contains seven problems.
Part B contains four coding problems with self-contained reference solutions.
The branch total is **11 problems**.

The CPU coding problems use Python and, where stated, PyTorch.
Use the existing environment or the setup in the notes.
No GPU training run is required for these exercises.

## Part A — Pen and paper

### A1 — Chinchilla compute budget ●●

You plan a dense decoder with $N=3\times10^9$ parameters.
Use $D=20N$ and $C\approx6ND$.

1. Calculate the token budget and training FLOPs.
2. A four-GPU cluster sustains $1.5\times10^{14}$ FLOP/s per GPU on the workload.
   Calculate its idealized wall-clock duration in days.
3. If the model grows to 6B parameters while retaining the twenty-token rule, by what factor does compute grow?
4. State two reasons why the calculated duration is not a delivery-date guarantee.

<details markdown="1">
<summary>Solution</summary>

First calculate tokens:

$$
D=20(3\times10^9)=60\times10^9.
$$

The dataset budget is 60 billion token occurrences.
This count does not guarantee that all occurrences contain unique information.

Then calculate compute:

$$
C=6(3\times10^9)(60\times10^9)
=1.08\times10^{21}\text{ FLOPs}.
$$

The cluster's aggregate sustained rate is

$$
4(1.5\times10^{14})=6\times10^{14}\text{ FLOP/s}.
$$

Divide operations by operations per second:

$$
t=\frac{1.08\times10^{21}}{6\times10^{14}}
=1.8\times10^6\text{ s}.
$$

Convert seconds to days:

$$
t=\frac{1.8\times10^6}{86400}\approx20.8333\text{ days}.
$$

At 6B parameters, the token budget also doubles to 120B.
The product $ND$ therefore grows by four.
Equivalently, $C\approx120N^2$ under this allocation rule.

The approximate formula omits some attention, optimizer, and recomputation costs.
Operational pauses such as checkpointing and restarts also increase elapsed time unless the measured rate already includes them.
Hardware failure and data-pipeline stalls add uncertainty.
Do not apply an extra utilization discount twice to a rate that already includes those losses.

</details>

### A2 — Curation and decoder changes ●●

Two normalized documents have shingle sets $A=\{a,b,c,d\}$ and $B=\{b,c,d,e\}$.

1. Calculate their Jaccard similarity.
2. With 100 independent MinHash components, what is the expected number of matching components?
3. Explain why splitting these documents randomly before near-deduplication can invalidate an evaluation.
4. A decoder has vocabulary size 32,000 and hidden width 2048.
   How many parameters does untying its output embedding add?
5. Explain whether replacing a standard attention kernel with FlashAttention changes the causal language-model objective.

<details markdown="1">
<summary>Solution</summary>

The intersection is $\{b,c,d\}$ and has size three.
The union is $\{a,b,c,d,e\}$ and has size five.
Therefore

$$
J(A,B)=3/5=0.6.
$$

Each ideal independent MinHash component matches with probability 0.6.
Linearity of expectation gives $100\times0.6=60$ matching components.
The observed number varies; the expectation is not a guaranteed exact match count.

A random split can place one document in training and its near duplicate in evaluation.
The model can then appear to generalize while benefiting from duplicated content.
Cluster near duplicates and split at the cluster or source level.
Protect evaluation benchmarks with a separate contamination check.

An independent output matrix adds $Vd$ parameters relative to a tied table:

$$
32{,}000\times2048=65{,}536{,}000.
$$

FlashAttention changes the execution strategy and memory traffic.
With the same causal mask and inputs, it computes the same attention operation up to floating-point effects.
It does not replace next-token cross-entropy with a different objective.
Its dense attention arithmetic still depends quadratically on sequence length.

</details>

### A3 — LoRA parameters with grouped-query attention ●●

A model has 32 decoder layers and hidden width 4096.
Each layer has the following projections in output-by-input notation:

- $W_q$: $4096\times4096$.
- $W_k$: $1024\times4096$.
- $W_v$: $1024\times4096$.
- $W_o$: $4096\times4096$.

Attach rank-16 LoRA to all four projections.
Use $\alpha=32$ and train no biases or extra heads.

1. Calculate adapter parameters for each projection and for the full model.
2. Compare the count with the notes' all-square example.
3. State the update scale and explain whether doubling alpha changes the parameter count.
4. Explain why initializing both LoRA factors to zero prevents learning.

<details markdown="1">
<summary>Solution</summary>

For a matrix with input width $d_i$ and output width $d_o$, the count is

$$
P=r(d_i+d_o).
$$

The query and output projections each use

$$
16(4096+4096)=131{,}072.
$$

The key and value projections each use

$$
16(4096+1024)=81{,}920.
$$

One decoder layer therefore uses

$$
2(131{,}072)+2(81{,}920)=425{,}984.
$$

Across 32 layers:

$$
32(425{,}984)=13{,}631{,}488.
$$

The all-square example has 16,777,216 adapter parameters.
The difference is

$$
16{,}777{,}216-13{,}631{,}488=3{,}145{,}728.
$$

Grouped-query key/value projections reduce the adapter count by 18.75% in this stated comparison.
The scale is $\alpha/r=32/16=2$.
Doubling alpha changes the scale to four but creates no new parameters.

The correction is $BA$.
Its gradient with respect to $A$ contains $B$, and its gradient with respect to $B$ contains $A$.
If both factors are zero, both factor gradients are zero.
Standard initialization uses random $A$ and zero $B$, preserving the initial function while allowing $B$ to learn first.

</details>

### A4 — QLoRA memory accounting ●●

Consider an eight-billion-parameter base model.
Use these explicit assumptions:

- Full fine-tuning uses BF16 weights and gradients, FP32 master weights, and two FP32 Adam moments.
- The total parameter-state accounting is sixteen bytes per full-model parameter.
- The QLoRA base weight codes use four bits per parameter.
- The adapter has 16,777,216 trainable parameters with the same sixteen-byte training-state accounting.
- Ignore quantization metadata, activations, temporary buffers, and allocator overhead in both calculations.

1. Calculate full fine-tuning parameter-state memory.
2. Calculate the quantized base weight floor and the adapter-state memory.
3. Calculate the approximate ratio of the first total to the combined QLoRA floor.
4. Explain why this ratio does not predict whether an 8B model trains on a 6 GB GPU.

<details markdown="1">
<summary>Solution</summary>

First verify the sixteen-byte assumption:

$$
2+2+4+4+4=16\text{ bytes per parameter}.
$$

Full fine-tuning parameter states occupy

$$
8\times10^9\times16=128\times10^9\text{ bytes}=128\text{ GB}.
$$

Four bits equal half a byte, so the quantized base weight floor is

$$
8\times10^9\times0.5=4\times10^9\text{ bytes}=4\text{ GB}.
$$

The adapter states occupy

$$
16{,}777{,}216\times16
=268{,}435{,}456\text{ bytes}
=0.268435456\text{ GB}.
$$

The combined floor is

$$
4+0.268435456=4.268435456\text{ GB}.
$$

The ratio is

$$
128/4.268435456\approx29.9876.
$$

Using only base weight codes gives the larger $128/4=32$ ratio, but that comparison omits trainable adapter states.
Both calculations omit scale metadata, activation tensors, logits, and optimizer or kernel temporary buffers.
Sequence length and batch size can make those omitted allocations substantial.
A 6 GB card therefore does not follow from a 4.268 GB lower accounting floor.
Some full-training implementations use a different master-weight or gradient policy, which changes the baseline as well.

</details>

### A5 — One DPO loss and its directions ●●

For one prompt, the preferred and rejected completion log-probabilities are:

| Model | Preferred | Rejected |
|---|---:|---:|
| Trainable policy | -2.0 | -3.0 |
| Frozen reference | -2.5 | -2.7 |

Use $\beta=0.2$.

1. Calculate the reference-relative margin, the scaled DPO logit, and the loss.
2. Compare the loss with the equal-margin baseline.
3. Determine the gradient sign for each policy log-probability.
4. Explain why arbitrary token averaging changes the standard sequence-level objective.

<details markdown="1">
<summary>Solution</summary>

The preferred reference-relative score is

$$
-2.0-(-2.5)=0.5.
$$

The rejected reference-relative score is

$$
-3.0-(-2.7)=-0.3.
$$

Subtract the second score from the first:

$$
m=0.5-(-0.3)=0.8.
$$

Scale the margin:

$$
z=0.2(0.8)=0.16.
$$

The negative log-sigmoid loss is

$$
\mathcal L=\log(1+e^{-0.16})\approx0.616344.
$$

The zero-margin baseline is $\log2\approx0.693147$.
This pair has a positive margin and a lower loss.

Differentiate with respect to the logit:

$$
\frac{\partial\mathcal L}{\partial z}=\sigma(z)-1.
$$

At $z=0.16$, $\sigma(z)\approx0.539915$.
The preferred log-probability gradient is

$$
0.2(0.539915-1)\approx-0.092017.
$$

The rejected log-probability gradient has the opposite sign, about $+0.092017$.
Gradient descent therefore favors the preferred response relative to the rejected one.
The neural policy's normalized probabilities remain coupled; the two scalar inputs are a diagnostic abstraction.

Standard DPO uses completion sequence log-probabilities, which sum token log-probabilities.
Replacing each sum with an average divides by a response-dependent length.
That changes margins and preference weighting when lengths differ.
Use a named length-normalized variant if you intend that change.

</details>

### A6 — Bradley–Terry and KL-regularized reward ●●

A reward model scores two responses as 1.2 and 0.2.
A policy assigns probabilities $(0.8,0.2)$, and its reference assigns $(0.5,0.5)$.

1. Calculate the Bradley–Terry probability that the first response wins and the corresponding negative log-likelihood.
2. Calculate the exact forward KL divergence from the policy to the reference.
3. With $\beta=0.1$ and expected reward 0.7, calculate the regularized objective.
4. For a single sampled second response, calculate the log-ratio and explain why it can be negative.

<details markdown="1">
<summary>Solution</summary>

The reward difference is $1.2-0.2=1$.
The Bradley–Terry preference probability is

$$
\sigma(1)=\frac{1}{1+e^{-1}}\approx0.731059.
$$

Its negative log-likelihood is

$$
-\log(0.731059)\approx0.313262.
$$

For the exact KL, sum both outcomes under the policy distribution:

$$
\begin{aligned}
D_{\mathrm{KL}}
&=0.8\log(1.6)+0.2\log(0.4)\\
&=0.8(0.470003629)+0.2(-0.916290732)\\
&\approx0.192744757.
\end{aligned}
$$

Multiply by the penalty coefficient:

$$
\beta D_{\mathrm{KL}}=0.019274476.
$$

Subtract that penalty from expected reward:

$$
J=0.7-0.019274476\approx0.680725524.
$$

For the second response alone, the sampled log-ratio is

$$
\log(0.2/0.5)=\log0.4\approx-0.916291.
$$

A single signed log-ratio is not the full KL divergence.
The exact KL averages over the policy distribution and is nonnegative.
An estimator based on a finite set of samples can differ from that expectation.

</details>

### A7 — Match the feedback contract ●●●

Choose a starting method and justify its machinery for each scenario.
Give one evaluation risk per scenario.

1. You have 20,000 clean chosen/rejected pairs from the same prompts and a stable SFT baseline.
2. You have individual user thumbs-up/down labels but no matched rejected response for most prompts.
3. You want one joint imitation/preference stage and cannot retain a frozen reference model.
4. You have a validated reward signal, need new on-policy samples, and can afford rollout infrastructure.
5. In scenario 1, an engineer replaces the reference model with a different tokenizer revision.
   Explain whether precomputed reference scores remain valid.

<details markdown="1">
<summary>Solution</summary>

For scenario 1, start with DPO.
It consumes same-prompt preference pairs and compares the trainable policy with the frozen SFT reference.
You can cache reference completion scores for a fixed dataset and input contract.
Inspect preference coverage and length bias on held-out prompts.

For scenario 2, consider KTO.
It uses binary desirability labels without requiring pairs for the same prompt.
It still uses reference-relative scores and a KL-related reference point.
Inspect class imbalance and the bias of users who submit feedback.

For scenario 3, consider ORPO.
It combines chosen-response likelihood with an odds-ratio preference term and does not need a frozen reference model.
It still requires preference pairs.
Inspect the imitation/preference weighting and capability drift.

For scenario 4, consider PPO-based RLHF.
Use the reward signal, on-policy rollouts, a frozen reference for the KL term, and any value-model machinery the implementation requires.
Inspect reward hacking and instability rather than assuming higher reward means higher task success.
If the reward is programmatically verifiable rather than human-derived, call the broader procedure RL with that reward rather than claiming all such runs are RLHF.

For scenario 5, the cached values do not describe the new tokenizer/model contract automatically.
Tokenization determines the sequence events whose log-probabilities you sum.
Keep the same tokenization and template for both policy and reference, or explicitly rebuild the experiment under a validated common contract.
Do not subtract incompatible sequence scores and call the result standard DPO.

</details>

## Part B — Code

### B1 — Format records and preserve a response mask ●●

Write a standard-library-only script that:

1. Converts the ShareGPT-style sample below to the exact ChatML string from the notes.
2. Constructs labels for input IDs `[10, 11, 12, 20, 21, 2, 0]` with prompt length three and final-token padding.
3. Reports the supervised token count.
4. Checks that the first response token remains supervised and padding remains ignored.

Use explicit token IDs for the mask exercise.
Do not pretend that character offsets and token offsets are interchangeable.

<details markdown="1">
<summary>Solution</summary>

The formatting function maps stored role names to ChatML roles.
The mask keeps every input token but selects only response labels for loss.
Both operations are independent of any large-model download.

```python
def render_chatml(record):
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    turns = record["conversations"]
    if not turns:
        raise ValueError("At least one turn is required")
    result = []
    for turn in turns:
        role = roles[turn["from"]]
        content = turn["value"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Each turn needs nonempty text")
        result.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(result)


record = {"conversations": [
    {"from": "human", "value": "What is 2 + 2?"},
    {"from": "gpt", "value": "4."},
]}
text = render_chatml(record)
ids = [10, 11, 12, 20, 21, 2, 0]
attention = [1, 1, 1, 1, 1, 1, 0]
prompt_length = 3
labels = [
    token if i >= prompt_length and attention[i] else -100
    for i, token in enumerate(ids)
]
assert labels[3] == ids[3]
assert labels[-1] == -100
assert ids[:3] == [10, 11, 12]
print(text, end="")
print(labels)
print("supervised:", sum(label != -100 for label in labels))
```

Expected output:

```text
<|im_start|>user
What is 2 + 2?<|im_end|>
<|im_start|>assistant
4.<|im_end|>
[-100, -100, -100, 20, 21, 2, -100]
supervised: 3
```

A real tokenizer can represent special markers as single IDs or ordinary text depending on its configuration.
Use the target tokenizer's template to obtain real input IDs.
Inspect the resulting assistant mask after truncation and packing.

</details>

### B2 — Implement and merge a tiny LoRA layer ●●●

Use PyTorch on a CPU.
Implement a LoRA wrapper around a bias-free `Linear(4, 3)` layer at rank two and alpha four.

1. Freeze the base parameters.
2. Initialize random $A$ and zero $B$.
3. Check exact initial output identity and count fourteen trainable parameters.
4. Take one SGD step on a nontrivial target.
5. Merge $sBA$ into a plain linear layer and check that its output matches the adapted layer within floating-point tolerance.

<details markdown="1">
<summary>Solution</summary>

The merged weight has the base matrix's `[3, 4]` shape.
The factor product `B @ A` has that same shape because `B` is `[3, 2]` and `A` is `[2, 4]`.
The output check compares the model's visible function, not merely a stored matrix name.

```python
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)


torch.manual_seed(7)
base = nn.Linear(4, 3, bias=False)
adapted = LoRALinear(base, rank=2, alpha=4)
x = torch.randn(5, 4)
target = torch.ones(5, 3)
torch.testing.assert_close(adapted(x), base(x), rtol=0, atol=0)
trainable = [p for p in adapted.parameters() if p.requires_grad]
count = sum(p.numel() for p in trainable)
assert count == 14
optimizer = torch.optim.SGD(trainable, lr=0.1)
optimizer.zero_grad()
F.mse_loss(adapted(x), target).backward()
assert base.weight.grad is None
assert adapted.B.grad.abs().sum().item() > 0
optimizer.step()
assert not torch.allclose(adapted(x), base(x))
merged = nn.Linear(4, 3, bias=False)
with torch.no_grad():
    merged.weight.copy_(base.weight + adapted.scale * (adapted.B @ adapted.A))
torch.testing.assert_close(merged(x), adapted(x), rtol=1e-5, atol=1e-6)
print(f"trainable={count}")
print("initial_identity=True base_frozen=True merged_match=True")
```

Expected output:

```text
trainable=14
initial_identity=True base_frozen=True merged_match=True
```

This non-quantized linear example permits direct merging.
A quantized large-model merge has additional support and requantization constraints.
Do not infer those guarantees from this dense FP32 exercise.

</details>

### B3 — Implement DPO and check the chosen direction ●●

Use PyTorch tensors of shape `[batch]`.
Implement a numerically stable DPO loss over completion log-probabilities.

1. Reproduce A5 to six decimal places.
2. Check that the preferred scalar receives a negative gradient and the rejected scalar a positive gradient.
3. Raise the preferred log-probability by 0.1 while holding the other diagnostic inputs fixed.
   Show that the loss decreases.
4. Keep the reference detached from autograd.

<details markdown="1">
<summary>Solution</summary>

The reference supplies a fixed baseline, not a second trainable policy.
The `logsigmoid` operation implements the pair objective stably.
This example checks a scalar pair; a real training loop must first gather masked completion-token probabilities.

```python
import torch
from torch.nn import functional as F


def dpo_loss(pc, pr, rc, rr, beta=0.2):
    if beta <= 0:
        raise ValueError("beta must be positive")
    margin = (pc - rc.detach()) - (pr - rr.detach())
    return -F.logsigmoid(beta * margin).mean()


pc = torch.tensor([-2.0], dtype=torch.float64, requires_grad=True)
pr = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
rc = torch.tensor([-2.5], dtype=torch.float64)
rr = torch.tensor([-2.7], dtype=torch.float64)
loss = dpo_loss(pc, pr, rc, rr)
loss.backward()
assert pc.grad.item() < 0
assert pr.grad.item() > 0
improved = dpo_loss(pc.detach() + 0.1, pr.detach(), rc, rr)
assert improved.item() < loss.item()
print(f"loss={loss.item():.6f}")
print(f"chosen_grad={pc.grad.item():.6f}")
print(f"rejected_grad={pr.grad.item():.6f}")
print(f"improved_loss={improved.item():.6f}")
```

Expected output:

```text
loss=0.616344
chosen_grad=-0.092017
rejected_grad=0.092017
improved_loss=0.607192
```

The improved logit is $0.2(0.9)=0.18$.
Its loss is $\log(1+e^{-0.18})\approx0.607192$.
These diagnostic scalar changes do not claim that two full-model sequence probabilities move independently.

</details>

### B4 — Build a resource and exposure calculator ●●

Write a standard-library-only script that reports:

1. The Chinchilla tokens and FLOPs for a 3B dense model.
2. The rank-16 adapter count for A3's 32-layer grouped-query architecture.
3. The full-training parameter-state memory and QLoRA floor under A4's assumptions, using A4's adapter count.
4. The effective sequence batch for two examples per device, eight accumulation steps, and four devices.

Keep the grouped-query adapter count and A4's all-square adapter count separate.
Label every output with a unit or count meaning.

<details markdown="1">
<summary>Solution</summary>

The script uses the matrix-shape formula rather than assuming that every attention projection is square.
Memory remains an explicit accounting estimate, not a hardware-fit claim.
The two adapter counts answer different hypothetical configurations.

```python
def lora_parameters(projections, rank, layers):
    if rank <= 0 or layers <= 0:
        raise ValueError("rank and layers must be positive")
    return layers * sum(rank * (out_dim + in_dim) for out_dim, in_dim in projections)


n = 3_000_000_000
d = 20 * n
c = 6 * n * d
gqa = [(4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096)]
gqa_count = lora_parameters(gqa, rank=16, layers=32)
a4_count = 16_777_216
full_gb = 8_000_000_000 * 16 / 1e9
qlora_gb = (8_000_000_000 * 0.5 + a4_count * 16) / 1e9
effective_batch = 2 * 8 * 4
assert gqa_count == 13_631_488
print(f"tokens={d:,}")
print(f"training_flops={c:.2e}")
print(f"gqa_adapter_parameters={gqa_count:,}")
print(f"full_parameter_states_GB={full_gb:.3f}")
print(f"qlora_parameter_floor_GB={qlora_gb:.3f}")
print(f"effective_sequence_batch={effective_batch}")
```

Expected output:

```text
tokens=60,000,000,000
training_flops=1.08e+21
gqa_adapter_parameters=13,631,488
full_parameter_states_GB=128.000
qlora_parameter_floor_GB=4.268
effective_sequence_batch=64
```

The effective batch calculation assumes data parallelism and one optimizer update per accumulation window.
Variable-length examples and packing change supervised tokens per update.
Add target-token accounting before using this estimate to compare real runs.

</details>

## Self-assessment

For Part A, require at least five of the seven problems to be correct before progressing.
That exceeds the syllabus's 70% threshold.
For Part B, reproduce all four expected behaviors, including the identity, frozen-base, merge, and gradient-direction checks.
Recompute one quantitative answer on a blank page the next day.

Explain every assumption when presenting a memory or compute number.
A correct arithmetic result with an unstated or incompatible model is not a correct engineering estimate.
Return to the [exit criteria](index.md#exit-criteria) after completing the exercises.
