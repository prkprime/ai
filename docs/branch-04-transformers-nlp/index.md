# Branch 4 — Transformers & Modern NLP

Turn text into token IDs, then derive the transformer that processes those IDs.
This nine-hour branch connects the mathematics in Branch 1 to the training machinery in Branch 3.
It supplies the architecture vocabulary for [Branch 5, LLM engineering](../branch-05-llm-fine-tuning/index.md) and [Branch 6, retrieval and agents](../branch-06-rag-agents/index.md).
Treat tensor shapes, masks, and tokenizer configuration as executable contracts.

## Learning objectives

- **4.1:** Trace BPE merges, compare WordPiece and Unigram, and train a byte-covering tokenizer with explicit special tokens.
- **4.1:** Derive the skip-gram negative-sampling and GloVe objectives and distinguish static from contextual representations.
- **4.2:** Compute scaled dot-product attention by hand and implement causal attention with annotated tensor shapes.
- **4.2:** Derive sinusoidal and rotary position encodings and test their relative-position properties.
- **4.2:** Count transformer parameters and compare MHA, MQA, and GQA cache costs.
- **4.2:** Select encoder-only, decoder-only, or encoder-decoder architectures from their attention masks and training objectives.
- **4.2:** Explain which memory transfers FlashAttention removes without claiming linear-time exact attention.

## Prerequisites

- Review [Branch 1 notes, subtopic 1.1: matrix algebra and projections](../branch-01-math-foundations/notes.md).
  Multiply matrices, track dimensions, and distinguish a dot product from elementwise multiplication.
- Review [Branch 1 notes, subtopic 1.2: probability and cross-entropy](../branch-01-math-foundations/notes.md).
  Interpret a normalized categorical distribution and negative log-likelihood.
- Review [Branch 3 notes, subtopic 3.1: autograd, softmax, and LayerNorm](../branch-03-deep-learning/notes.md).
  Differentiate through a tensor program and explain why normalization changes optimization.
- Review [Branch 3 notes, subtopic 3.2: training stability](../branch-03-deep-learning/notes.md).
  Explain gradients, initialization, and residual connections.

## Time budget and subtopic map

| Subtopic | Hours | Notes | Evidence you produce |
|---|---:|---|---|
| 4.1 Tokenization & linguistic representation | 2 | [Tokenization notes](notes.md#tokenization) | Merge trace, tokenizer round trip, objective comparison |
| 4.2 The transformer architecture deep dive | 7 | [Architecture notes](notes.md#architecture) | Attention implementation, position calculations, parameter count |
| **Total** | **9** | | **2 + 7 = 9** |

The links above use explicit anchors declared in this branch's notes.
Cross-branch links target files because generated heading slugs differ across renderers.

## Nine one-hour sessions

| Hour | Subtopic | Study and practice | Minutes |
|---|---|---|---:|
| 1 | 4.1 | Read tokenization notes; trace BPE; attempt A1 | 60 |
| 2 | 4.1 | Compare objectives and special tokens; run tokenizer snippet; attempt A2 and B1 | 60 |
| 3 | 4.2 | Derive attention; watch the attention video; attempt A3 | 60 |
| 4 | 4.2 | Track MHA shapes; implement causal masking; attempt A4 and B2 | 60 |
| 5 | 4.2 | Derive sinusoidal encodings and RoPE; attempt A5 and B3 | 60 |
| 6 | 4.2 | Compare position schemes, MQA, and GQA; attempt A6 | 60 |
| 7 | 4.2 | Compare architecture families and training targets; attempt A7 | 60 |
| 8 | 4.2 | Count parameters; explain attention complexity; attempt A8 and B4 | 60 |
| 9 | 4.2 | Recompute examples without solutions; debug all snippets; assess exit criteria | 60 |
| **Total** | **4.1: 120 min; 4.2: 420 min** | **540 minutes** | **540** |

Read [the notes](notes.md) in order.
Attempt [the problems](problems.md) before opening each solution.
The playlist is a resource menu, not an additional required nine hours.
Use selected segments within the study time above; watch full long-form builds as optional reinforcement.

## Video playlist

Titles, channels, and availability were checked through YouTube oEmbed on 2026-10-16.
Durations use the video page's media metadata, rounded to the nearest second.

| Title | Channel | Duration | Why it earns study time |
|---|---|---:|---|
| [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) | Andrej Karpathy | 2:13:34 | Relates byte-level BPE, Unicode, and special-token boundaries to real model behavior; select the BPE section for hour 1. |
| [Transformers, the tech behind LLMs \| Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) | 3Blue1Brown | 27:14 | Establishes the residual-stream and next-token-prediction picture before the architecture derivation. |
| [Attention in transformers, step-by-step \| Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) | 3Blue1Brown | 26:09 | Visualizes query–key matching and weighted value aggregation for hour 3. |
| [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Andrej Karpathy | 1:56:20 | Connects causal masks, multi-head projections, and language-model training; use selected implementation segments. |

## Embedded playlist

Play selected segments here when you use the built MkDocs site.
Use the linked table on GitHub, which removes inline video players.

<iframe width="560" height="315" src="https://www.youtube.com/embed/zduSFxRajkE" title="Let's build the GPT Tokenizer — Andrej Karpathy" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/wjZofJX0v4M" title="Transformers, the tech behind LLMs | Deep Learning Chapter 5 — 3Blue1Brown" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/eMlx5fFNoYc" title="Attention in transformers, step-by-step | Deep Learning Chapter 6 — 3Blue1Brown" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/kCc8FmEb1nY" title="Let's build GPT: from scratch, in code, spelled out. — Andrej Karpathy" frameborder="0" allowfullscreen></iframe>

## Key sources

Every link below passed a live `curl` status check.
The Hugging Face course moved from the NLP-course path to the LLM-course path; use the canonical links below.
Primary papers support the mathematical claims; illustrated guides support first-pass intuition.

| Author or organization | Title | Type | Link | Covers | Accessed |
|---|---|---|---|---|---|
| Sennrich, Haddow, Birch | Neural Machine Translation of Rare Words with Subword Units | Primary paper | [arXiv 1508.07909](https://arxiv.org/abs/1508.07909) | 4.1 BPE | accessed 2026-10-16 |
| Wu et al. | Google's Neural Machine Translation System | Primary paper | [arXiv 1609.08144](https://arxiv.org/abs/1609.08144) | 4.1 WordPiece | accessed 2026-10-16 |
| Kudo | Subword Regularization | Primary paper | [arXiv 1804.10959](https://arxiv.org/abs/1804.10959) | 4.1 Unigram and segmentation sampling | accessed 2026-10-16 |
| Kudo and Richardson | SentencePiece | Primary paper | [arXiv 1808.06226](https://arxiv.org/abs/1808.06226) | 4.1 raw-text tokenization | accessed 2026-10-16 |
| Mikolov et al. | Distributed Representations of Words and Phrases and their Compositionality | Primary paper | [arXiv 1310.4546](https://arxiv.org/abs/1310.4546) | 4.1 negative sampling | accessed 2026-10-16 |
| Pennington, Socher, Manning | GloVe: Global Vectors for Word Representation | Paper and project | [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) | 4.1 co-occurrence objective | accessed 2026-10-16 |
| Hugging Face | Tokenizers; the tokenizers library; BPE; WordPiece; Unigram | Official course | [Chapter 2](https://huggingface.co/learn/llm-course/chapter2/4), [chapter 6 overview](https://huggingface.co/learn/llm-course/chapter6/1), [BPE](https://huggingface.co/learn/llm-course/chapter6/5), [WordPiece](https://huggingface.co/learn/llm-course/chapter6/6), [Unigram](https://huggingface.co/learn/llm-course/chapter6/7) | 4.1 implementations | accessed 2026-10-16 |
| Hugging Face | Chat templates | Official documentation | [Chat templating](https://huggingface.co/docs/transformers/chat_templating) | 4.1 roles and serialization | accessed 2026-10-16 |
| Vaswani et al. | Attention Is All You Need | Primary paper | [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) | 4.2 attention, sinusoidal positions, architecture | accessed 2026-10-16 |
| Su et al. | RoFormer: Enhanced Transformer with Rotary Position Embedding | Primary paper | [arXiv 2104.09864](https://arxiv.org/abs/2104.09864) | 4.2 rotary position derivation | accessed 2026-10-16 |
| Ainslie et al. | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | Primary paper | [arXiv 2305.13245](https://arxiv.org/abs/2305.13245) | 4.2 grouped keys and values | accessed 2026-10-16 |
| Dao et al. | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Primary paper | [arXiv 2205.14135](https://arxiv.org/abs/2205.14135) | 4.2 tiled exact attention | accessed 2026-10-16 |
| Devlin et al. | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Primary paper | [arXiv 1810.04805](https://arxiv.org/abs/1810.04805) | 4.2 encoder and MLM | accessed 2026-10-16 |
| Radford et al. | Language Models are Unsupervised Multitask Learners | Primary report | [GPT-2 report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 4.2 decoder and causal LM | accessed 2026-10-16 |
| Raffel et al. | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | Primary paper | [arXiv 1910.10683](https://arxiv.org/abs/1910.10683) | 4.2 T5 and relative bias | accessed 2026-10-16 |
| Peng et al. | YaRN: Efficient Context Window Extension of Large Language Models | Primary paper | [arXiv 2309.00071](https://arxiv.org/abs/2309.00071) | 4.2 context extension | accessed 2026-10-16 |
| Press, Smith, Lewis | Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation | Primary paper | [arXiv 2108.12409](https://arxiv.org/abs/2108.12409) | 4.2 ALiBi | accessed 2026-10-16 |
| Jay Alammar | The Illustrated Transformer; The Illustrated GPT-2 | Illustrated guides | [Transformer](https://jalammar.github.io/illustrated-transformer/), [GPT-2](https://jalammar.github.io/illustrated-gpt2/) | 4.2 visual walkthroughs | accessed 2026-10-16 |
| Stanford | CS224N: Natural Language Processing with Deep Learning | University course | [Course](https://web.stanford.edu/class/cs224n/) | 4.1–4.2 lectures and exercises | accessed 2026-10-16 |
| Jurafsky and Martin | Speech and Language Processing, third-edition draft | Textbook | [Official draft](https://web.stanford.edu/~jurafsky/slp3/) | 4.1–4.2 linguistic and architectural context | accessed 2026-10-16 |

## Exit criteria

- [ ] Trace three BPE merges with weighted pair counts and a stated tie rule.
- [ ] Explain why byte coverage prevents unknown byte sequences but does not make token counts language-neutral.
- [ ] Serialize a two-turn chat without confusing role metadata with trusted instructions.
- [ ] Compute every entry of a two-token attention result without a library attention function.
- [ ] Track every tensor in the MHA shapes table and identify the softmax axis.
- [ ] Derive the score-variance scaling under explicit independence assumptions.
- [ ] Show that future value changes do not affect earlier causal outputs.
- [ ] Rotate a two-dimensional vector and derive RoPE's relative-position identity.
- [ ] Compare sinusoidal, learned, relative-bias, RoPE, and ALiBi positions.
- [ ] Count MHA and feed-forward weights with the bias convention stated.
- [ ] Compute MHA, GQA, and MQA cache sizes for a specified configuration.
- [ ] Distinguish MLM, causal LM, and span-corruption targets and attention masks.
- [ ] Explain why FlashAttention saves score-matrix memory but not quadratic dense-attention arithmetic.
- [ ] Score at least 70% on the eight pen-and-paper problems and reproduce all four coding outputs.

Continue to [Branch 5](../branch-05-llm-fine-tuning/index.md) for training and adaptation.
Use [Branch 6](../branch-06-rag-agents/index.md) to connect contextual representations to retrieval.
Keep serving implementation details for [Branch 7](../branch-07-llmops-serving/index.md).
