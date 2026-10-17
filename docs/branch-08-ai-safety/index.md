# Branch 8 — AI safety, red teaming & governance

Build a deployment decision from evidence, not from a model's fluent answers.
This final branch connects alignment, retrieval, tool permissions, evaluation, and incident response.
You spend exactly **4 core hours**: 2 hours on security and 2 hours on evaluation and interpretability.
You finish with a small safety case: a documented argument that your deployment meets explicit, limited release conditions.

Read the [teaching notes](notes.md), attempt the [problems](problems.md), and return to the exit criteria.
All attack demonstrations use synthetic documents and harmless canaries.
Run them only against systems you own or have permission to assess.

## Learning objectives

After 8.1, you can:

- Trace a direct or indirect injection across a named trust boundary.
- Classify roleplay, encoding, many-shot, and adversarial-suffix jailbreaks by the mechanism they exploit.
- Implement a layered gate with explicit deny, allow, and review outcomes.
- Compare a content-safety classifier with an authorization check and explain why neither replaces the other.
- Redact synthetic personal data and identify the residual leakage paths.
- Calculate a toy watermark detection statistic and explain its assumptions.

After 8.2, you can:

- Distinguish a predictive probe, an attention pattern, and a causal intervention.
- Explain what a sparse autoencoder feature dictionary adds to neuron-level inspection.
- Select MMLU, GSM8K, HumanEval, or preference comparisons for a stated measurement goal.
- Compute an Elo update, gatekeeper confusion metrics, and expected calibration error.
- Diagnose contamination, unequal subgroup errors, and weak construct validity.
- Produce a release recommendation with owners, thresholds, monitoring, and a rollback trigger.

## Prerequisites

- Review [Branch 5 notes, subtopic 5.3: alignment and preference optimization](../branch-05-llm-fine-tuning/notes.md).
  Explain supervised fine-tuning, preference labels, and the limits of alignment as an application security boundary.
- Review [Branch 6 notes, subtopic 6.3: function calling, tool use, and agentic loops](../branch-06-rag-agents/notes.md).
  Draw the path from retrieved text to an assistant answer and from a proposed tool call to its executor.
- Bring a synthetic retrieval example and a tool schema from your earlier work.
  Replace all real credentials and personal records with fake values before this branch.

## Hour budget

| Subtopic | Core hours | Required evidence |
|---|---:|---|
| 8.1 Security & prompt hardening | 2 | Attack trace, redaction output, gatekeeper metrics, layered control map |
| 8.2 Bias, interpretability & evaluation benchmarks | 2 | Benchmark selection, ECE calculation, Elo calculation, release decision |
| **Total** | **4** | **2 + 2 = 4 hours** |

### Four-hour study schedule

Videos replace the indicated reading minutes; they do not add core hours.
The full playlist and full source papers are reference material beyond the core excerpts.
Model downloads and optional package installation happen before the four-hour clock.

| Session | Subtopic | Minutes | Work |
|---|---|---:|---|
| Hour 1 | 8.1 | 60 | 5 min video excerpt; 25 min threat model and injection notes; 15 min Problems A1–A2; 15 min B1 |
| Hour 2 | 8.1 | 60 | 5 min video excerpt; 20 min guardrails, PII, watermark notes; 15 min A3–A4; 20 min B2 |
| Hour 3 | 8.2 | 60 | 5 min video excerpt; 25 min interpretability and benchmark notes; 15 min A5–A6; 15 min B3 |
| Hour 4 | 8.2 | 60 | 5 min video excerpt; 15 min calibration and bias notes; 20 min B4; 20 min release checklist and review |
| **Total** | | **240** | **240 minutes ÷ 60 = 4 hours** |

## Subtopic map

| Subtopic | Notes | Practice |
|---|---|---|
| 8.1 Security & prompt hardening | [Threat model and attacks](notes.md#81-security-and-prompt-hardening) | [Part A, A1–A4; Part B, B1–B2](problems.md) |
| 8.1 Layered controls | [Defense in depth](notes.md#defense-in-depth), [privacy](notes.md#data-leakage-and-pii-redaction), [watermarks](notes.md#model-watermarking) | Attack trace, gatekeeper metrics, redaction |
| 8.2 Bias, interpretability & evaluation benchmarks | [Representations and evidence](notes.md#82-bias-interpretability-and-evaluation-benchmarks) | [Part A, A5–A6; Part B, B3–B4](problems.md) |
| 8.2 Deployment decisions | [Calibration](notes.md#hallucination-detection-and-calibration), [governance](notes.md#close-the-loop-on-responsible-deployment) | ECE report, red-team report, release gate |

## Video playlist

These are individual videos, not required full-length viewing within four hours.
Titles and channels match YouTube metadata.
Durations come from the video's reported running time, rounded to the nearest second.
Links and metadata are checked on 2026-10-16.

| Video title | Channel | Duration | Core excerpt and why it earns study time |
|---|---|---:|---|
| [[1hr Talk] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) | Andrej Karpathy | 59:48 | Hour 1: choose 5 minutes from the security discussion; connect prompt injection to untrusted text. |
| [State of GPT \| BRK216HFS](https://www.youtube.com/watch?v=bZQun8Y4L2A) | Microsoft Developer | 42:40 | Hour 2: choose 5 minutes on assistant training and application use; distinguish learned behavior from enforced permissions. |
| [How might LLMs store facts \| Deep Learning Chapter 7](https://www.youtube.com/watch?v=9-Jl0dxWQs8) | 3Blue1Brown | 22:42 | Hour 3: watch 5 minutes on distributed representations; prepare for sparse feature dictionaries and their limits. |
| [Transformers, the tech behind LLMs \| Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) | 3Blue1Brown | 27:14 | Hour 4: review 5 minutes on token probabilities; distinguish next-token prediction from calibrated factual correctness. |

## Embedded playlist

Use the linked table if your Markdown viewer removes embedded players.
The MkDocs site displays these players inline.

<iframe width="560" height="315" src="https://www.youtube.com/embed/zjkBMFhNj_g" title="[1hr Talk] Intro to Large Language Models — Andrej Karpathy" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/bZQun8Y4L2A" title="State of GPT | BRK216HFS — Microsoft Developer" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/9-Jl0dxWQs8" title="How might LLMs store facts | Deep Learning Chapter 7 — 3Blue1Brown" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/wjZofJX0v4M" title="Transformers, the tech behind LLMs | Deep Learning Chapter 5 — 3Blue1Brown" frameborder="0" allowfullscreen></iframe>

## Key sources

Each row gives a primary paper, an official model card, or an official project resource.
Use the cited version's taxonomy and protocol rather than assuming the newest model shares them.
The Arena methodology paper replaces the landing page because the landing page fails automated access during verification.

| Author or organization | Title | Type | Link | Covers | Access date |
|---|---|---|---|---|---|
| OWASP | Top 10 for LLM applications | Official security guidance | [Top 10](https://genai.owasp.org/llm-top-10/) | 8.1 prompt injection, disclosure, excessive agency | accessed 2026-10-16 |
| Hines et al. | Defending against indirect prompt injection attacks with spotlighting | Research paper | [arXiv:2403.14720](https://arxiv.org/abs/2403.14720) | 8.1 lower-trust content marking | accessed 2026-10-16 |
| Wallace et al. | The instruction hierarchy: Training LLMs to prioritize privileged instructions | Research paper | [arXiv:2404.13208](https://arxiv.org/abs/2404.13208) | 8.1 instruction hierarchy | accessed 2026-10-16 |
| NVIDIA | NeMo Guardrails overview | Official documentation | [NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/latest/index.html) | 8.1 orchestration and rails | accessed 2026-10-16 |
| Inan et al. | Llama Guard: LLM-based input-output safeguard for human-AI conversations | Research paper | [arXiv:2312.06674](https://arxiv.org/abs/2312.06674) | 8.1 safety classification | accessed 2026-10-16 |
| Meta | Llama Guard 3-1B | Official model card | [Model and taxonomy](https://huggingface.co/meta-llama/Llama-Guard-3-1B) | 8.1 versioned labels and inference | accessed 2026-10-16 |
| Explosion | Named entity recognition | Official documentation | [spaCy entities](https://spacy.io/usage/linguistic-features#named-entities) | 8.1 NER-based redaction | accessed 2026-10-16 |
| Kirchenbauer et al. | A watermark for large language models | Research paper | [arXiv:2301.10226](https://arxiv.org/abs/2301.10226) | 8.1 logit bias and detection | accessed 2026-10-16 |
| Alain and Bengio | Understanding intermediate layers using linear classifier probes | Research paper | [arXiv:1610.01644](https://arxiv.org/abs/1610.01644) | 8.2 decodable representations | accessed 2026-10-16 |
| Anthropic and collaborators | Transformer Circuits publications | Research collection | [Publications](https://transformer-circuits.pub/) | 8.2 mechanistic interpretability | accessed 2026-10-16 |
| Olsson et al. | In-context learning and induction heads | Research article | [Induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | 8.2 attention circuits and interventions | accessed 2026-10-16 |
| Bricken et al. | Towards monosemanticity: Decomposing language models with dictionary learning | Research article | [Feature dictionaries](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | 8.2 sparse autoencoders | accessed 2026-10-16 |
| Hendrycks et al. | Measuring massive multitask language understanding | Benchmark paper | [MMLU](https://arxiv.org/abs/2009.03300) | 8.2 multiple-choice knowledge | accessed 2026-10-16 |
| Cobbe et al. | Training verifiers to solve math word problems | Benchmark paper | [GSM8K](https://arxiv.org/abs/2110.14168) | 8.2 grade-school math | accessed 2026-10-16 |
| Chen et al. | Evaluating large language models trained on code | Benchmark paper | [HumanEval](https://arxiv.org/abs/2107.03374) | 8.2 executable code correctness | accessed 2026-10-16 |
| Chiang et al. | Chatbot Arena: An open platform for evaluating LLMs by human preference | Methodology paper | [arXiv:2403.04132](https://arxiv.org/abs/2403.04132) | 8.2 pairwise preferences and rating uncertainty | accessed 2026-10-16 |
| Wang et al. | Self-consistency improves chain of thought reasoning in language models | Research paper | [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) | 8.2 sampled answer agreement | accessed 2026-10-16 |
| Guo et al. | On calibration of modern neural networks | Research paper | [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) | 8.2 ECE and calibration | accessed 2026-10-16 |
| Vectara | Hallucination leaderboard | Official evaluation repository | [Repository](https://github.com/vectara/hallucination-leaderboard) | 8.2 grounded summarization evaluation and protocol limits | accessed 2026-10-16 |
| NIST | AI Risk Management Framework | Official framework | [AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) | 8.1 and 8.2 Govern, Map, Measure, Manage | accessed 2026-10-16 |

## Exit criteria

Treat every unchecked item as a reason to continue studying, not as evidence to hide.

- [ ] I complete at least 70% of Part A without revealing solutions.
- [ ] I reproduce all four coding exercises' reference behavior with synthetic data.
- [ ] I trace an indirect injection from retrieved document to proposed assistant behavior.
- [ ] I explain one false positive and one false negative in the gatekeeper exercise.
- [ ] I separate content moderation from tool authorization and retrieval access control.
- [ ] I identify at least two privacy leaks that output regex redaction cannot prevent.
- [ ] I explain why watermark detection does not establish truth, ownership, or individual authorship.
- [ ] I distinguish probing, attention analysis, SAE features, and causal evidence.
- [ ] I compute the ECE and Elo examples on paper and name their assumptions.
- [ ] I select benchmarks for a target use case and document contamination and subgroup risks.
- [ ] I write a release gate with an accountable owner, review path, and rollback condition.
- [ ] I document the limits of the evidence rather than claim universal safety.

## Final handoff

Attach your branch artifacts to the retrieval or agent project from Branch 6.
Connect its release gate to the serving and monitoring work from Branch 7.
Use the [responsible-deployment checklist](notes.md#close-the-loop-on-responsible-deployment) to make the decision auditable.
Return to the [full syllabus](../index.md) when you complete the branch.
