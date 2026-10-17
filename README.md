# 60-Hour AI/LLM Engineering Syllabus

Source-backed study path from math foundations to production LLM systems.
Eight branches, 60 core hours, four milestone projects, reviewed against [ACCEPTANCE.md](ACCEPTANCE.md) by an independent model family.

## Start here

- [Syllabus home](docs/index.md) — master schedule, dependency map, projects.
- [Branch 1 — Mathematical Foundations & Production Engineering](docs/branch-01-math-foundations/index.md)
- [Branch 2 — Classical Statistical & Machine Learning](docs/branch-02-classical-ml/index.md)
- [Branch 3 — Deep Learning Foundations & Representation Learning](docs/branch-03-deep-learning/index.md)
- [Branch 4 — Transformers & Modern NLP](docs/branch-04-transformers-nlp/index.md)
- [Branch 5 — LLM Engineering & Fine-Tuning](docs/branch-05-llm-fine-tuning/index.md)
- [Branch 6 — RAG & Agentic Workflows](docs/branch-06-rag-agents/index.md)
- [Branch 7 — LLMOps, High-Throughput Serving & Systems](docs/branch-07-llmops-serving/index.md)
- [Branch 8 — AI Safety, Red Teaming & Governance](docs/branch-08-ai-safety/index.md)
- [Consolidated references](docs/references.md)

## Build the site

The docs build with MkDocs Material:

```sh
uvx --with mkdocs-material mkdocs serve
```

Then open http://127.0.0.1:8000.

Math renders via MathJax, diagrams via Mermaid, and video sections embed as iframes in the built site.
On GitHub's Markdown renderer, iframes do not render; video links and titles still appear as normal links.

## Conventions

- One folder per branch under `docs/`, each with `index.md` (overview, schedule, sources, videos), `notes.md` (teaching notes), and `problems.md` (problems with collapsible solutions).
- Math uses `$...$` and `$$...$$` delimiters.
- Diagrams use Mermaid code fences or clearly formatted ASCII.
- All cited URLs were link-checked at authoring time (2026-10-16).
