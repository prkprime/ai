# Branch 5 — LLM engineering and fine-tuning

Turn a pretrained decoder into a task-focused assistant.
You budget pretraining compute, build supervised datasets, inject low-rank adapters, and compare preference objectives.
The core budget is exactly **9 hours**.
Large GPU runs and the portfolio project sit outside that budget.

Read the [teaching notes](notes.md), then attempt the [problems](problems.md) before opening their solutions.
Use the [syllabus home](../index.md) to place this branch in the full path.

## Learning objectives

- **5.1:** Calculate tokens and approximate training FLOPs from a parameter budget, and justify a data-curation pipeline.
- **5.1:** Compare a modern decoder's normalization, feed-forward layers, embeddings, and attention kernels with the vanilla transformer.
- **5.2:** Convert instruction records into masked training examples and calculate LoRA trainable parameters.
- **5.2:** Implement a zero-update LoRA initialization and choose an SFT configuration that fits a stated memory budget.
- **5.3:** Derive the DPO logit from a KL-regularized reward objective and calculate a pairwise loss.
- **5.3:** Select PPO-based RLHF, DPO, ORPO, or KTO from the available feedback and operational constraints.

## Prerequisites

- [Branch 4, §4.1: tokenization and chat templates](../branch-04-transformers-nlp/notes.md).
  Explain how instruction data is tokenized and where role boundaries come from.
- [Branch 4, §4.2: attention and language-model objectives](../branch-04-transformers-nlp/notes.md).
  Trace causal attention and shifted next-token cross-entropy before starting.
- [Branch 3, §3.2: AdamW and mixed precision](../branch-03-deep-learning/notes.md).
  Explain optimizer states, gradient accumulation, and numerical precision.
- Python and PyTorch tensor operations from the earlier branches.
  Read tensor dimensions and run a small CPU script.

## Hour budget and subtopic map

| Subtopic | Hours | Notes | Observable result |
|---|---:|---|---|
| 5.1 Pre-training dynamics and architecture variants | 2 | [5.1](notes.md#51-pre-training-dynamics-and-architecture-variants) | Compute budget and decoder comparison |
| 5.2 Adaptation, SFT, and parameter-efficient fine-tuning | 3.5 | [5.2](notes.md#52-adaptation-sft-and-parameter-efficient-fine-tuning) | Masked dataset and LoRA calculation |
| 5.3 Alignment and preference optimization | 3.5 | [5.3](notes.md#53-alignment-and-preference-optimization) | DPO loss and objective-selection rationale |
| **Total** | **9** | | **2 + 3.5 + 3.5 = 9** |

### Session schedule

The exercises and selected video excerpts fit inside these sessions.
The full videos form a reference shelf, not extra mandatory viewing.
Use the named chapter in the video player when timestamps differ by version.

| Elapsed time | Hours | Session | Work |
|---|---:|---|---|
| 0–1 | 1 | 5.1 Compute and data | Read scaling laws; calculate A1; watch 10 minutes on pretraining in *Intro to Large Language Models* |
| 1–2 | 1 | 5.1 Architecture and curation | Trace the training stack; solve A2; inspect the modern-decoder comparison |
| 2–3 | 1 | 5.2 Dataset contract | Render formats; implement B1; inspect loss masks |
| 3–4 | 1 | 5.2 Adapter mechanics | Derive LoRA counts; solve A3; run B2 |
| 4–5 | 1 | 5.2 Memory and configuration | Solve A4; complete B4; inspect SFTTrainer and QLoRA configuration |
| 5–5.5 | 0.5 | 5.2 Evaluation and deployment | Select hyperparameters; diagnose forgetting |
| 5.5–6.5 | 1 | 5.3 RLHF | Fit a pairwise reward conceptually; solve A6; watch 10 minutes on post-training in *Deep Dive into LLMs like ChatGPT* |
| 6.5–7.5 | 1 | 5.3 DPO | Follow the derivation; solve A5; run B3 |
| 7.5–8.5 | 1 | 5.3 Alternatives and evaluation | Compare ORPO/KTO; solve A7; design held-out preference evaluation |
| 8.5–9 | 0.5 | Consolidation | Recompute one result without notes; complete the exit checklist |
| **Total** | **9** | | |

## Video playlist

Durations describe the full videos.
Only the excerpts allocated above count toward the core schedule.
The two coding videos support optional refreshers when Branch 4 feels incomplete.
All four links return HTTP 200 and expose matching YouTube oEmbed titles on 2026-10-16.

| Title | Channel | Duration | Link | Why it earns study time |
|---|---|---|---|---|
| [1hr Talk] Intro to Large Language Models | Andrej Karpathy | 59:48 | [Watch](https://www.youtube.com/watch?v=zjkBMFhNj_g) | Separates pretraining from assistant post-training; use the pretraining segment in session 1. |
| Let's build GPT: from scratch, in code, spelled out. | Andrej Karpathy | 1:56:20 | [Watch](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Makes the causal language-model loss concrete before you add completion masks. |
| Let's reproduce GPT-2 (124M) | Andrej Karpathy | 4:01:26 | [Watch](https://www.youtube.com/watch?v=l8pRSuU81PU) | Connects token throughput, effective batch size, and implementation details to a compute budget. |
| Deep Dive into LLMs like ChatGPT | Andrej Karpathy | 3:31:23 | [Watch](https://www.youtube.com/watch?v=7xTGNNLPyMI) | Distinguishes supervised imitation from reward-driven post-training; use 10 minutes of the post-training chapters. |

## Embedded playlist

<iframe width="560" height="315" src="https://www.youtube.com/embed/zjkBMFhNj_g" title="Intro to Large Language Models — Andrej Karpathy" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/kCc8FmEb1nY" title="Let's build GPT: from scratch, in code, spelled out. — Andrej Karpathy" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/l8pRSuU81PU" title="Let's reproduce GPT-2 (124M) — Andrej Karpathy" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/7xTGNNLPyMI" title="Deep Dive into LLMs like ChatGPT — Andrej Karpathy" frameborder="0" allowfullscreen></iframe>

## Key sources

Read the indicated sections, not every linked document in full.
Papers supply the objective definitions; official documentation supplies the library contracts.
The access date records live URL checks, not an assertion that every package release stays API-compatible.

| Author or organization | Title | Type | Link | Covers | Accessed |
|---|---|---|---|---|---|
| Kaplan et al., OpenAI | Scaling Laws for Neural Language Models (2020) | Paper | [arXiv](https://arxiv.org/abs/2001.08361) | 5.1 Power laws and compute allocation | accessed 2026-10-16 |
| Hoffmann et al., DeepMind | Training Compute-Optimal Large Language Models (2022) | Paper | [arXiv](https://arxiv.org/abs/2203.15556) | 5.1 Chinchilla allocation | accessed 2026-10-16 |
| Dubey et al., Meta | The Llama 3 Herd of Models (2024) | Paper | [arXiv](https://arxiv.org/abs/2407.21783) | 5.1 Decoder architecture, data, and training | accessed 2026-10-16 |
| Dao | FlashAttention-2 (2023) | Paper | [arXiv](https://arxiv.org/abs/2307.08691) | 5.1 Attention kernel parallelism | accessed 2026-10-16 |
| Shah et al. | FlashAttention-3 (2024) | Paper | [arXiv](https://arxiv.org/abs/2407.08608) | 5.1 Hopper-specific attention kernels | accessed 2026-10-16 |
| Penedo et al., Hugging Face | The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale (2024) | Paper | [arXiv](https://arxiv.org/abs/2406.17557) | 5.1 Filtering and deduplication | accessed 2026-10-16 |
| Andrej Karpathy | nanoGPT | Source code | [Repository](https://github.com/karpathy/nanoGPT) | 5.1 Minimal pretraining implementation | accessed 2026-10-16 |
| Andrej Karpathy and contributors | llm.c | Source code | [Repository](https://github.com/karpathy/llm.c) | 5.1 Low-level training systems | accessed 2026-10-16 |
| Hu et al., Microsoft | LoRA: Low-Rank Adaptation of Large Language Models (2021) | Paper | [arXiv](https://arxiv.org/abs/2106.09685) | 5.2 Low-rank updates | accessed 2026-10-16 |
| Dettmers et al. | QLoRA: Efficient Finetuning of Quantized LLMs (2023) | Paper | [arXiv](https://arxiv.org/abs/2305.14314) | 5.2 NF4, double quantization, paged optimizers | accessed 2026-10-16 |
| Hugging Face | PEFT documentation | Official documentation | [Docs](https://huggingface.co/docs/peft) | 5.2 Adapter configuration and merging | accessed 2026-10-16 |
| Hugging Face | TRL documentation | Official documentation | [Docs](https://huggingface.co/docs/trl) | 5.2 SFT and 5.3 preference trainers | accessed 2026-10-16 |
| Hugging Face | Transformers documentation | Official documentation | [Docs](https://huggingface.co/docs/transformers) | 5.2 Tokenizers and model loading | accessed 2026-10-16 |
| Unsloth | Unsloth documentation | Official documentation | [Docs](https://docs.unsloth.ai/) | 5.2 Optional optimized fine-tuning workflow | accessed 2026-10-16 |
| Ouyang et al., OpenAI | Training language models to follow instructions with human feedback (2022) | Paper | [arXiv](https://arxiv.org/abs/2203.02155) | 5.3 Reward models and PPO-based RLHF | accessed 2026-10-16 |
| Rafailov et al. | Direct Preference Optimization: Your Language Model is Secretly a Reward Model (2023) | Paper | [arXiv](https://arxiv.org/abs/2305.18290) | 5.3 Implicit rewards and DPO | accessed 2026-10-16 |
| Hong, Lee, and Thorne | ORPO: Monolithic Preference Optimization without Reference Model (2024) | Paper | [arXiv](https://arxiv.org/abs/2403.07691) | 5.3 Reference-free odds ratio | accessed 2026-10-16 |
| Ethayarajh et al. | KTO: Model Alignment as Prospect Theoretic Optimization (2024) | Paper | [arXiv](https://arxiv.org/abs/2402.01306) | 5.3 Utility-based binary feedback | accessed 2026-10-16 |

## Exit criteria

- [ ] Calculate a Chinchilla token budget and state the assumptions behind $C \approx 6ND$.
- [ ] Explain why an inference-heavy product can rationally train beyond 20 tokens per parameter.
- [ ] Name four modern decoder changes and distinguish an architectural change from a kernel change.
- [ ] Describe MinHash deduplication, split contamination, and synthetic-data provenance risks.
- [ ] Render an instruction record and identify every supervised target token.
- [ ] Calculate LoRA parameters for square and grouped-query projections.
- [ ] Verify that a fresh adapter leaves a frozen linear layer's output unchanged.
- [ ] Separate QLoRA weight storage from total training memory.
- [ ] Derive the DPO pair logit and calculate its loss without confusing token and sequence probabilities.
- [ ] Calculate the reference KL term for a two-outcome policy.
- [ ] Choose among PPO-based RLHF, DPO, ORPO, and KTO using a stated feedback contract.
- [ ] Score at least 70% on Part A and reproduce all four Part B reference behaviors.

## After the branch

Use [Project 2](../index.md#project-2-domain-specific-peft-fine-tuning) as an optional portfolio extension.
Preserve data provenance, adapter configuration, and held-out evaluation artifacts.
Continue to [Branch 6](../branch-06-rag-agents/index.md) for retrieval and tool workflows.
