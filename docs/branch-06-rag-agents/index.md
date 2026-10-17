# Branch 6 — Retrieval-augmented generation and agentic workflows

Build a retrieval system that supplies evidence to a language model.
Then connect that model to validated tools through a bounded workflow.
Retrieval-augmented generation (RAG) combines retrieved material with generation instead of relying only on model parameters.
An agentic workflow lets model outputs influence which tool or workflow step runs next.
Neither technique removes the need for evaluation, access control, or explicit failure handling.

This branch takes exactly **8 core hours**.
Its three sessions combine selected reading, video excerpts, worked examples, and exercises.
The source tables support deeper study; reading every linked source falls outside the core budget.
The production RAG milestone is an extension beyond the core hours.

## Learning objectives

After this branch, you can:

- **Derive** cosine scores, contrastive retrieval losses, and vector-memory estimates for subtopic 6.1.
- **Select** a chunking strategy, embedding dimension, and approximate index using measured recall and latency for subtopic 6.1.
- **Compute** BM25 and reciprocal rank fusion scores by hand for subtopic 6.2.
- **Diagnose** retrieval, grounding, and answer-relevance failures with separate measurements for subtopic 6.2.
- **Implement** validated tool dispatch and a checkpointed state graph for subtopic 6.3.
- **Diagnose** a wrong-tool trace and place a human approval boundary before a consequential action for subtopic 6.3.

## Prerequisites

- [Branch 4 notes, subtopic 4.1: tokenization and embeddings](../branch-04-transformers-nlp/notes.md).
  Explain token embeddings and distinguish token vectors from pooled sentence vectors.
- [Branch 4 notes, subtopic 4.2: attention](../branch-04-transformers-nlp/notes.md).
  Trace how attention combines token information before you compare bi-encoders and cross-encoders.
- [Branch 5 notes, subtopic 5.2: instruction tuning and fine-tuned assistants](../branch-05-llm-fine-tuning/notes.md).
  Apply an instruction template and distinguish assistant output from trusted application instructions.
- [Branch 5 notes, subtopics 5.2 and 5.3: adaptation objectives and controlled evaluation](../branch-05-llm-fine-tuning/notes.md).
  Separate a prompt change from a parameter update.
- Use Python functions, dictionaries, NumPy arrays, and virtual environments without additional scaffolding.

## Hour budget

| Subtopic | Core hours | Result |
|---|---:|---|
| 6.1 Dense retrieval, embeddings and vector stores | 2.5 | A documented index and chunking decision |
| 6.2 Advanced RAG architectures | 3.0 | A hybrid retrieval calculation and evaluation plan |
| 6.3 Function calling, tool use and agentic loops | 2.5 | A validated, checkpointed tool workflow |
| **Total** | **8.0** | **2.5 + 3.0 + 2.5 = 8.0** |

## Session schedule

Elapsed times start when you begin the branch.
Video time sits inside the listed activity budgets.
Use the notes as the core text; consult sources only for the named sections during these sessions.

| Elapsed time | Hours | Subtopic | Activity |
|---|---:|---|---|
| 0:00–0:45 | 0.75 | 6.1 | Study bi-encoders, contrastive learning, MRL, and chunking; watch the vector-database video |
| 0:45–1:45 | 1.00 | 6.1 | Trace flat, IVF, HNSW, and PQ; complete A1 and A2 |
| 1:45–2:30 | 0.75 | 6.1 | Run cosine retrieval; complete B1; compare stores |
| 2:30–3:30 | 1.00 | 6.2 | Watch the RAG video; derive BM25 and RRF; complete A3 and A4 |
| 3:30–4:30 | 1.00 | 6.2 | Study reranking and query transformation; complete A5 and B2 |
| 4:30–5:30 | 1.00 | 6.2 | Map the RAG triad to metrics; complete A6 and inspect the Ragas example |
| 5:30–6:30 | 1.00 | 6.3 | Watch the agentic-workflows talk; trace ReAct and structured decoding; complete A7 and B3 |
| 6:30–7:30 | 1.00 | 6.3 | Build the LangGraph example; complete A8 and B4 |
| 7:30–8:00 | 0.50 | 6.3 | Review authorization, checkpoints, and exit criteria; record Branch 7 handoff |
| **Total** | **8.00** | | **480 minutes** |

The transformer video is an optional prerequisite refresher.
Do not add its full duration to the eight-hour core schedule.
If you need extra practice, extend your study time rather than skipping verification steps.

## Subtopic map

| Subtopic | Teaching notes | Practice |
|---|---|---|
| 6.1 | [Dense retrieval, embeddings and vector stores](notes.md#61-dense-retrieval-embeddings-and-vector-stores) | [A1–A2 and B1](problems.md) |
| 6.2 | [Advanced RAG architectures](notes.md#62-advanced-rag-architectures) | [A3–A6 and B2](problems.md) |
| 6.3 | [Function calling, tool use and agentic loops](notes.md#63-function-calling-tool-use-and-agentic-loops) | [A7–A8 and B3–B4](problems.md) |

## Video playlist

Titles and channels match the publishers' video metadata.
Durations round the published player duration to the nearest second.
All four watch URLs return HTTP 200 and all four YouTube oEmbed responses contain a title on 2026-10-16.

| Video | Channel | Duration | Why it earns study time |
|---|---|---:|---|
| [What is a Vector Database?](https://www.youtube.com/watch?v=t9IDoenf-lo) | IBM Technology | 8:11 | Connect vector representations, similarity search, and database responsibilities before choosing an index. |
| [What is Retrieval-Augmented Generation (RAG)?](https://www.youtube.com/watch?v=T-D1OfcDW1M) | IBM Technology | 6:35 | Identify the retrieval boundary and explain why evidence changes the generation input. |
| [What's next for AI agentic workflows ft. Andrew Ng of AI Fund](https://www.youtube.com/watch?v=sal78ACtGTc) | Sequoia Capital | 13:40 | Compare reflection, planning, tool use, and multi-agent designs with a fixed one-pass prompt. |
| [Transformers, the tech behind LLMs \| Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) | 3Blue1Brown | 27:14 | Refresh embeddings and token prediction if the Branch 4 prerequisites need reinforcement. |

## Embedded playlist

Play the same four verified videos below when you use the built site.
Use the link table if your Markdown viewer removes embedded players.

<iframe width="560" height="315" src="https://www.youtube.com/embed/t9IDoenf-lo" title="What is a Vector Database? — IBM Technology" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/T-D1OfcDW1M" title="What is Retrieval-Augmented Generation (RAG)? — IBM Technology" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/sal78ACtGTc" title="What's next for AI agentic workflows ft. Andrew Ng of AI Fund — Sequoia Capital" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/wjZofJX0v4M" title="Transformers, the tech behind LLMs — 3Blue1Brown" frameborder="0" allowfullscreen></iframe>

## Key sources

Every row records **accessed 2026-10-16**.
Prefer the paper for an algorithm and the official implementation documentation for an API.
The learning center and course supply supporting explanations, not independent benchmark evidence.

| Author or organization | Title | Type | Link | Covers | Access date |
|---|---|---|---|---|---|
| Lewis et al. | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Paper, 2020 | [RAG](https://arxiv.org/abs/2005.11401) | 6.2, retrieval-conditioned generation | accessed 2026-10-16 |
| FlagOpen / BAAI | FlagEmbedding | Official implementation | [BGE models and rerankers](https://github.com/FlagOpen/FlagEmbedding) | 6.1, 6.2, embedding and reranking families | accessed 2026-10-16 |
| Wang et al. | Text Embeddings by Weakly-Supervised Contrastive Pre-training | Paper, 2022 | [E5](https://arxiv.org/abs/2212.03533) | 6.1, contrastive embedding training | accessed 2026-10-16 |
| Kusupati et al. | Matryoshka Representation Learning | Paper, 2022 | [MRL](https://arxiv.org/abs/2205.13147) | 6.1, nested embedding dimensions | accessed 2026-10-16 |
| MTEB contributors | MTEB leaderboard | Benchmark | [Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) | 6.1, task-specific embedding comparisons | accessed 2026-10-16 |
| Malkov and Yashunin | Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs | Paper, preprint 2016 | [HNSW](https://arxiv.org/abs/1603.09320) | 6.1, graph search and index parameters | accessed 2026-10-16 |
| Faiss contributors | Faiss documentation | Official documentation | [Faiss](https://faiss.ai/) | 6.1, flat, IVF, and product quantization indices | accessed 2026-10-16 |
| Qdrant | Qdrant documentation | Official documentation | [Qdrant](https://qdrant.tech/documentation/) | 6.1, filtering and vector storage | accessed 2026-10-16 |
| Milvus contributors | Milvus | Official implementation | [Milvus](https://github.com/milvus-io/milvus) | 6.1, distributed vector search | accessed 2026-10-16 |
| pgvector contributors | pgvector | Official implementation | [pgvector](https://github.com/pgvector/pgvector) | 6.1, PostgreSQL vector indices | accessed 2026-10-16 |
| Chroma | Chroma documentation | Official documentation | [Chroma](https://docs.trychroma.com/docs/overview/introduction) | 6.1, collections and deployment choices | accessed 2026-10-16 |
| Cormack, Clarke, and Buettcher | Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods | Paper, SIGIR 2009 | [RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) | 6.2, rank-based fusion | accessed 2026-10-16 |
| Gao et al. | Precise Zero-Shot Dense Retrieval without Relevance Labels | Paper, preprint 2022 | [HyDE](https://arxiv.org/abs/2212.10496) | 6.2, hypothetical document embeddings | accessed 2026-10-16 |
| Cohere | Rerank overview | Official documentation | [Rerank](https://docs.cohere.com/docs/rerank-overview) | 6.2, cross-encoder-style reranking service | accessed 2026-10-16 |
| LlamaIndex | Python framework documentation | Official documentation | [LlamaIndex](https://developers.llamaindex.ai/python/framework/) | 6.1, 6.2, ingestion and retrieval components | accessed 2026-10-16 |
| Ragas | Available metrics and faithfulness | Official documentation | [Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/), [faithfulness API](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) | 6.2, context metrics and collections-based evaluation | accessed 2026-10-16 |
| TruLens | The RAG triad | Official documentation | [RAG triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/) | 6.2, context relevance, groundedness, answer relevance | accessed 2026-10-16 |
| Yao et al. | ReAct: Synergizing Reasoning and Acting in Language Models | Paper, 2022 | [ReAct](https://arxiv.org/abs/2210.03629) | 6.3, alternating reasoning and tool actions | accessed 2026-10-16 |
| Wang et al. | Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models | Paper, 2023 | [Plan-and-solve](https://arxiv.org/abs/2305.04091) | 6.3, explicit task decomposition | accessed 2026-10-16 |
| LangChain | LangChain overview; LangGraph overview and persistence | Official documentation | [LangChain](https://docs.langchain.com/oss/python/langchain/overview), [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), [persistence](https://docs.langchain.com/oss/python/langgraph/persistence) | 6.3, tool integration, state, checkpoints | accessed 2026-10-16 |
| Instructor contributors | Instructor documentation | Official documentation | [Instructor](https://python.useinstructor.com/) | 6.3, typed output validation and retry integration | accessed 2026-10-16 |
| dottxt | Outlines documentation | Official documentation | [Outlines](https://dottxt-ai.github.io/outlines/latest/) | 6.3, structured generation and constrained decoding | accessed 2026-10-16 |
| Pinecone | Learning center | Supporting tutorials | [Learning center](https://www.pinecone.io/learn/) | 6.1, 6.2, retrieval explanations | accessed 2026-10-16 |
| DeepLearning.AI | Building Systems with the ChatGPT API | Supporting video course | [Course](https://www.deeplearning.ai/courses/chatgpt-building-system) | 6.2, 6.3, multi-step application design | accessed 2026-10-16 |

## Exit criteria

- [ ] Recompute cosine rankings and MRL prefix normalization without copying the solution.
- [ ] Explain why hard negatives help and why a false negative harms contrastive training.
- [ ] Compare flat, IVF-Flat, HNSW, and PQ using recall, memory, and latency.
- [ ] Recompute memory for one million 768-dimensional vectors in fp32, int8, and PQ codes.
- [ ] Compute both BM25 document scores and all RRF scores in the paper problems.
- [ ] Explain why a reranker cannot recover a document missing from its candidates.
- [ ] Distinguish a grounded but irrelevant answer from an ungrounded answer.
- [ ] Reproduce the four coding exercises and their specified outputs.
- [ ] Reject unknown tool names, invalid arguments, and unauthorized side effects before dispatch.
- [ ] Explain checkpoint scope, replay hazards, and the difference between thread state and shared memory.
- [ ] Correct at least 70% of the paper problems before revealing their solutions.
- [ ] Record unresolved evaluation failures instead of treating a high average score as proof of safety.

## Handoff to Branch 7

Carry your retrieval contract, validated tool schemas, and graph state schema into [Branch 7](../branch-07-llmops-serving/index.md).
Record the retrieval latency budget separately from model-serving latency.
The agentic-service milestone adds serving, concurrency measurements, tracing, and persistent infrastructure.
It reuses the tool boundaries and approval design from this branch.
The milestone does not fit inside these eight core hours.

Continue with the [teaching notes](notes.md), then attempt the [problems](problems.md).
