# Review log

Independent review loop for the 60-hour AI/LLM engineering syllabus.
Reviewers apply [ACCEPTANCE.md](ACCEPTANCE.md) verbatim and are a different model family than the authors.

- Round 1 reviewer: GLM-5.3 (`zai/glm-5.3`)
- Round 2+ reviewer (on failures / confirmation): GPT-6.1 (`openai/gpt-6.1-sol`)
- Author sessions: OpenCode coder agents; fixes continue in the author session where possible.
- Verdicts below are reviewer-reported, with parent-session reconciliation notes where a fix landed after the verdict.

| Branch | Round | Reviewer | Verdict | Blocking issues summary | Resolution |
|--------|-------|----------|---------|--------------------------|------------|
| 1 | 1 | GLM-5.3 (`zai/glm-5.3`) | **PASS** | none — all criteria pass; 10 formulas recomputed, 9 snippets executed, 18+ sources fetched | no fix needed |
| 2 | 1 | GLM-5.3 (`zai/glm-5.3`) | **PASS** | none blocking; noted root-table count (already fixed to 12) | no branch fix needed |
| 3 | 1 | GLM-5.3 (`zai/glm-5.3`) | **PASS** | none — all A/B/C criteria pass; 6 derivations recomputed, 13 snippets executed, 12 sources + 4 videos live-verified | no fix needed |
| 4 | 1 | GLM-5.3 (`zai/glm-5.3`) | **PASS** | none — 39 checks, 10 snippets executed, 13 papers verified; #architecture INFO is a validator false-positive (anchor present in built HTML) | no fix needed |
| 5 | 1 | GLM-5.3 (`zai/glm-5.3`) | **PASS** | C2 WARN only: one dead anchor pair (GitHub-style slug) | fixed by parent (single-dash MkDocs anchor) |
| 6 | 1 | GLM-5.3 (`zai/glm-5.3`) | **PASS** | non-blocking: (a) flagged URL false positive (file cites working singular slug; verified by parent + GPT-6.1); (b) optional ragas install failed at import — real, fixed with verified pin | ragas pin `langchain-community<0.4` applied to notes.md (reproduced failure + verified fix) |
| 6 | 2 | GPT-6.1 (`openai/gpt-6.1-sol`) | **PASS** | none blocking; C2 WARN (prereq links name sections without anchors — site convention, non-blocking) | no fix needed |
| 7 | 1 | GLM-5.3 (`zai/glm-5.3`) | **PASS** | none blocking; root-table count discrepancy (10 vs actual 12) found outside branch | root table fixed by parent (10→12) |
| 8 | 1 | GLM-5.3 (`zai/glm-5.3`) | **PASS** | none blocking; 2 minor notes (4 GB→4.5 GB wording; duplicate heading anchors, no links) | 4.5 GB wording fixed by parent |

## Verdict details

Full verdict blocks from each reviewer round are appended below as the loop progresses.

### Branch 1 — Round 1 (GLM-5.3) — PASS

```
BRANCH: 1 — Mathematical Foundations & Production Engineering
VERDICT: PASS
A1-A7: PASS — 3 prose-heavy subtopic sections; hours 3+2+3=8; verb-first objectives; explicit no-prerequisite note with refresher links; 8 one-hour sessions; every subtopic mapped; problems Part A/B with 10 <details markdown="1"> rendering in build.
B1-B7: PASS — 10 formulas recomputed (projection, Bernoulli MLE, Beta MAP, Gaussian MLE/MAP, entropy/KL, softmax-CE gradient, eigen/SVD, dense backprop); 9 snippets executed on numpy 2.5.4, polars 1.44.2, numba 0.68.0, pydantic 2.14.1, pytest 9.1.1 — outputs match; 4 Mermaid diagrams valid; 4 videos verified; 18+ sources fetched, no fabrications; numeric examples (8ND, Numba, perplexity, views) all recomputed correct.
C1-C4: PASS — style, links, structure, study flow.
ISSUES: None.
COUNT VERIFICATION: hours 8; problems 10 (6+4); snippets 15; diagrams 4; videos 4 (+4 embeds); sources 18; all match.
```

### Branch 3 — Round 1 (GLM-5.3) — PASS

```
BRANCH: 3 — Deep Learning Foundations & Representation Learning
VERDICT: PASS
A1-A7: PASS — all subtopics have substantive prose; hours 3+2.5+2.5=8; 8 verb-first objectives; prerequisites linked; 9-session schedule; every subtopic mapped to live primary sources; problems.md has Part A/B with 10 <details markdown="1"> solutions rendering in the build.
B1-B7: PASS — 6 derivations independently recomputed and matched; 10 problems stamped; all 13 snippets executed on torch 2.14.1 with matching outputs; 4 Mermaid diagrams valid and consistent; 4 videos oEmbed-verified with durations and embeds; 12 sources + 4 videos live-fetched, no fabrications; conv/ViT/RMSNorm/FP16 arithmetic recomputed correct.
C1-C4: PASS — style, links, structure, and study flow all pass.
ISSUES: None.
COUNT VERIFICATION: hours 8; problems 10 (6+4); snippets 13; diagrams 4; videos 4 (+4 embeds); sources 12; all match author claims.
```

### Branch 5 — Round 1 (GLM-5.3) — PASS

```
BRANCH: 5 — LLM Engineering & Fine-Tuning
VERDICT: PASS
A1-A7: PASS — all three subtopics substantive; hours 2+3.5+3.5=9; 6 verb-first objectives; prerequisites linked to B4 §4.2 and B3 §3.2; 10-session schedule; per-subtopic sources; 11 problems in Part A/B with <details markdown="1">.
B1-B7: PASS — 8B compute budget, GQA LoRA counts, DPO loss/gradient, BT loss, QLoRA memory (29.99x ratio), Kaplan exponents all recomputed correct; TRL v1.16.0/PEFT APIs current with stale-arg warnings; 8 of 11 snippets executed matching outputs; 3 Mermaid diagrams valid; 4 videos verified (durations exact); 12 arXiv citations + 8 docs fetched live.
C1-C4: PASS except C2 warned the Project 2 appendix anchor pair (GitHub-style slug); fixed by parent post-verdict (single-dash MkDocs anchor in index.md and notes.md).
ISSUES: None blocking.
COUNT VERIFICATION: hours 9; problems 11 (7+4); snippet blocks 11; diagrams 3; videos 4 (+4 embeds); sources 18; all match.
```

### Branch 8 — Round 1 (GLM-5.3) — PASS

```
BRANCH: 8 — AI Safety, Red Teaming & Governance
VERDICT: PASS
A1-A7: PASS — both subtopics substantive; hours 2+2=4; verb-first objectives; prerequisites linked to B5 §5.3 and B6 §6.3; 4-part schedule summing to 240 min; sources per subtopic; 10 problems with <details markdown="1">.
B1-B7: PASS — Elo update, ECE, watermark z-score, confusion matrix, pass@k, subgroup FPR all recomputed correct; all 4 reference solutions executed byte-for-byte; dtype= confirmed current (transformers >= 4.56); 3 Mermaid diagrams valid; 4 videos verified; 13 sources fetched live (Llama Guard S1-S13, spotlighting, HumanEval 164, MMLU 57, NIST RMF, watermark paper) — no fabrications.
C1-C4: PASS — canary-only safety posture explicit; one-H1 check passes (other # lines are code comments).
ISSUES: None blocking. Two minor notes: "roughly 4 GB" wording understated 1.12B x fp32 (~4.5 GB) — fixed by parent; duplicate "Common pitfalls"/"Checklist" headings deduplicate safely in MkDocs with no inbound links.
COUNT VERIFICATION: hours 4; problems 10 (6+4); snippets 10; diagrams 3; videos 4 (+4 embeds); sources 20; all match.
```

### Branch 7 — Round 1 (GLM-5.3) — PASS

```
BRANCH: 7 — LLMOps, High-Throughput Serving & Systems
VERDICT: PASS
A1-A7: PASS — 7.1/7.2 substantive; hours 4+3=7; 9 verb-first objectives; prerequisites linked to B4 §4.2 and B5 §5.2; 7-session schedule; 20 sources verified 200; 12 problems in Part A/B with <details markdown="1">.
B1-B7: PASS — ~15 numeric examples recomputed (8B KV 131,072 B/token; 70B INT4 32.60 GiB; speculative E[N]=3.3616; batch ceilings; scale overhead 3.125%; token bucket) all correct; all 4 CPU reference programs executed with byte-for-byte outputs; live API traces against vLLM/openai/Langfuse sources current; 3 Mermaid diagrams valid; 4 videos oEmbed+durations verified; 14 sources fetched; PagedAttention "20.4%" quote context-checked.
C1-C4: PASS.
ISSUES: None blocking. Parent-table discrepancy: docs/index.md said 10 problems for Branch 7; branch verified at 12 — root table corrected post-verdict.
COUNT VERIFICATION: hours 7; problems 12 (8+4); snippets 14; diagrams 3; videos 4 (+4 embeds); sources 20; all match.
```

### Branch 2 — Round 1 (GLM-5.3) — PASS

```
BRANCH: 2 — Classical Statistical & Machine Learning
VERDICT: PASS
A1-A7: PASS — 2.1/2.2 prose-heavy; hours 4+3=7; 6 verb-first objectives; prerequisites link B1 1.1/1.2; 7-row schedule summing 420 min; per-subtopic sources; 12 problems in Part A/B with <details markdown="1"> rendering in build.
B1-B7: PASS — 12 items recomputed (Ridge closed form, logistic grad, Gini/entropy gains, XGB leaf weights/gain, Brier decomposition both ways, PCA eigen, Lloyd, momentum, bias-variance, PR/F1 variants, soft-threshold); XGBoost early-stopping API current; 10 snippets executed with exact outputs; 3 Mermaid diagrams valid; 4 StatQuest videos oEmbed-verified; 15 sources fetched and content-checked; no fabrications.
C1-C4: PASS — (minor nit: DBSCAN acronym not expanded at first use; non-blocking).
ISSUES: None blocking. Root-table count noted; already corrected by parent (10→12).
COUNT VERIFICATION: hours 7; problems 12 (8+4); solution blocks 12; snippets 10; diagrams 3; videos 4 (+4 embeds); sources 15; all match.
```

### Branch 4 — Round 1 (GLM-5.3) — PASS

```
BRANCH: 4 — Transformers & Modern NLP
VERDICT: PASS
A1-A7: PASS — all subtopics substantive (BPE trace, Unigram EM, RoPE/NTK/YaRN/ALiBi, MHA/MQA/GQA with KV-cache formula, three architecture families, parameter counting, FlashAttention online softmax); hours 2+7=9; 7 verb-first objectives; prerequisites link B1 1.1/1.2 and B3 3.1/3.2; 9-session schedule = 540 min; every subtopic mapped to fetched sources; 12 problems with <details markdown="1">.
B1-B7: PASS — 39 checks, 0 failures (two-token attention by hand, own BPE reimplementation reproducing the merge trace, RoPE rotation + dot identity in float64, Unigram posteriors, SGNS/GloVe values, d_k variance 64→1, sinusoidal PE properties, NTK-aware scaling; KV cache 384 MiB / GQA 96 MiB / MQA 24 MiB; parameter counts; score tensor 512 MiB); all 10 python blocks executed with exact outputs; 3 Mermaid diagrams valid; 4 videos verified; 13 arXiv citations verified via API; ALiBi/Vaswani claims checked.
C1-C4: PASS — the #architecture sub-anchor is present in built HTML (validator INFO only); no branch warnings.
ISSUES: None.
COUNT VERIFICATION: hours 9; problems 12 (8+4); solution blocks 12; snippets 10; diagrams 3; videos 4 (+4 embeds); sources 20; all match.
```

### Branch 6 — Round 2 (GPT-6.1) — PASS

```
BRANCH: 6 — RAG & Agentic Workflows
VERDICT: PASS
A1-A7: PASS — all three subtopics substantive; hours 2.5+3+2.5=8 matching root table; 2 measurable objectives per subtopic; prerequisites link B4 4.1/4.2 and B5 5.2; 9-session schedule = 480 min; sources mapped per subtopic; 12 problems in Part A/B with <details markdown="1">.
B1-B7: PASS — BM25 recomputed (0.2373416716 / 0.1985680322); RRF recomputed (B/A/D/C order match); PQ and memory math recomputed (fp32 3,072,000,000 B; PQ codes 96,000,000 B; 32x payload compression); exactly 2 snippets traced against live Ragas/LangGraph APIs; 5 Mermaid diagrams valid; 4 videos verified (durations exact; 2 also oEmbed-checked); 5 sources fetched live; course URL confirmed 200 without redirect; documented optional `uv pip install ragas openai` dry-run resolves today (92 packages, ragas 0.4.3 + openai 3.3.0).
C1-C4: PASS — since C2 WARN: prerequisite labels name sections but omit anchors (matches site convention; non-blocking).
ISSUES: None blocking.
COUNT VERIFICATION: hours 8; problems 12 (8+4); solution blocks 12; snippets 11 (10 Python + 1 setup); diagrams 5; videos 4 (+4 embeds); sources 24; all match.
NOTE: Round 1 GLM-5.3 session ran extensive verification (versions, snippet execution, source checks) but did not terminate its final message; superseded by this Round 2 verdict.
```

### Branch 6 — Round 1 (GLM-5.3) — PASS (completed after Round 2; double confirmation)

```
BRANCH: 6 — RAG & Agentic Workflows
VERDICT: PASS
A1-A7: PASS — subtopics substantive; hours 2.5+3+2.5=8; 6 verb-first objectives; prerequisites verified; schedule = 480 min; sources mapped; 12/12 solutions in <details markdown="1"> verified rendering in built HTML.
B1-B7: PASS — 6 derivations recomputed (BM25 0.237342/0.198568; RRF order; PQ 96 B/vector, 32x; MRL; contrastive losses; masked softmax); all 10 Python snippets executed in a fresh venv at the exact claimed versions (NumPy 2.4.6, rank-bm25 0.2.2, Pydantic 2.14.1, LangGraph 1.2.15) with matching outputs; rank-bm25 IDF convention verified against installed source; 5 Mermaid diagrams parse under mermaid@10; 4 videos live-verified; 20+ citations fetched.
ISSUES (non-blocking):
1. Claimed dead plural deeplearning.ai slug — FALSE POSITIVE: the file cites the singular slug (index.md:134) which returns 200; verified independently by parent curl and Round 2 reviewer.
2. notes.md:47 optional `uv pip install ragas openai` resolved but failed at `import ragas` (langchain-community 0.4.x removed chat_models.vertexai) — REPRODUCED by parent; fixed by pinning 'langchain-community<0.4' (import verified OK with ragas 0.4.3 + langchain-community 0.3.31).
C1-C4: PASS — C4 WARN (above) resolved by the pin.
COUNT VERIFICATION: hours 8; problems 12 (8+4); solutions 12; snippets 11; diagrams 5; videos 4 (+4 embeds); sources 24; all match.
```

## Post-review rendering QA and deployment (2026-10-17)

- Visual QA (headless Chromium screenshots, parent) found Branch 1 solutions rendered raw `$$...$$`
  math because `pymdownx.arithmatex` only converts `$$` blocks starting at column 0, and Python-Markdown
  split numbered steps around indented math (step numbers also restarted).
- Fix (commit 34423fa, `branch-01: Fix display math rendering in solutions`): move display math to top
  level, convert step numbers to bold labels, dedent continuation prose; fenced code blocks verified
  untouched (indentation preserved in rendered HTML).
- Independent rendered-HTML review (GPT-6.1, screenshots + DOM/CSS across 8 routes): **PASS**.
  R1-R7 pass (code scroll without page overspill; no raw math; Mermaid renders as SVG; players inside
  content column at 560x315; solutions closed by default and render math/code when opened; mobile
  390px without horizontal scroll; 5 internal destinations 200).
  Non-blocking issues: two wide Mermaid flows render very small (B6 retrieval pipeline, B2 boosting
  pipeline) and B6 HNSW subgraph titles overlap.
- Deployment: repo published at https://github.com/prkprime/ai (public; Pages requires public on the
  free plan). GitHub Pages with Actions build type; deploy runs 38988725940 (site) green.
  Live at **https://prkprime.dev/ai/** (custom domain inherited from the user site; verified 200,
  fresh build, 98 MathJax elements on the B1 problems page).

## Cross-cutting gap and richness reviews (2026-10-17)

Two independent reviewers (GLM-5.3) covered branches 1-4 and 5-8 for correctness spot-checks, richness
against the hour budget, and knowledge-gap continuity of the prerequisite chain.

- Branches 1-4: **PASS**. Spot-checks recomputed (projection/MAP/perplexity/dense backward, Ridge/XGB
  gain/Brier decomposition, backprop/Xavier/AdamW/conv arithmetic, attention/RoPE/KV-cache/parameter
  counts). No contradictions; dependency chain B1->B4 sound; softmax-derivative repetition in B3 judged
  spaced repetition.
- Branches 5-8: **PASS**. 20+ recomputations (Chinchilla budget, LoRA counts, DPO derivation, BM25/RRF/PQ,
  KV memory scales, speculative decoding, Elo/ECE/watermark). Seams B4->B5, KV cache (B4/B7), quantization
  (B5/B7) verified from both sides.

Non-blocking gaps fixed after the reviews:

- B4 4.2: added "From logits to text: decoding modes" (greedy, temperature formula, top-k/top-p,
  determinism caveat) - previously used in B5/B6/B7 without introduction.
- B3 3.2: added a dropout definition (train-time zeroing with inverted scaling, disabled at eval) -
  previously used in B4/B5 without introduction.
- B2: defined the Lipschitz constant before the eta<=1/L rule; disambiguated t-SNE perplexity from
  language-model perplexity; introduced the Lagrange multiplier at first use in the PCA derivation.
- B5: defined reward hacking at first use; expanded mixture-of-experts beyond a caveat sentence.
- B5 prerequisites: added Branch 4 4.1 (tokenization and chat templates).
- B6: defined MRR and nDCG; relabeled the Branch 5 prerequisite bullet; fixed the two Mermaid diagrams
  flagged by the rendered-HTML review (retrieval pipeline and boosting pipeline now vertical flows;
  HNSW subgraph titles shortened so they no longer overlap).
- B1 1.1: forward pointer from the attention-backward section to Branch 4.
- B4 4.1: bridge paragraph connecting subword segmentation to morphology and word boundaries.

Accepted as-is (recorded decisions): B8 8.2 pacing (interpretability breadth against a 2-hour budget),
B5 5.1 FlashAttention recap as deliberate seam reinforcement, B1/B3 softmax-derivative repetition as
spaced repetition.

## Second-pass richness and rendering pass (2026-10-17)

Two independent reviewers (GLM-5.3) re-audited branches 1-4 and 5-8 for sub-section richness,
logic/flow, and Markdown rendering risks. Both returned **PASS** with concrete enrichment targets;
all applied:

- B1: thread/process timing snippet, typed kernel/adapter sketch, density-vs-probability example,
  corrected exercise pointer.
- B2: GBDT gain instantiation, UMAP membership formula, hierarchical merge trace, L1/L2 geometry
  example, schedule ordering fix.
- B3: dropout expansion (expectation check, mask example, `F.dropout` snippet, JMLR citation),
  ConvNeXt parameter contrast, zero-shot snippet, schedule mention.
- B4: decoding arithmetic example, WordPiece scoring example, SentencePiece marker example, SGNS
  loss value, KV-cache forward references, milestone link, prerequisite label normalization.
- B5: MoE routing and Mixtral counts, Kaplan exponent arithmetic, packing utilization, PPO clip
  objective, ORPO/KTO worked values, LoRA first-step gradient wording, training-stack diagram
  split, session schedule fix.
- B6: RAG objective paragraph, MRR/nDCG formulas with a worked ranking.
- B7: Branch 6 6.3 prerequisite, affine quantizer worked error, k-quant bits-per-weight, token
  bucket arithmetic, activation-outlier sentence.
- B8: induction-head attention table, SAE toy loss; evidence diagram split into two.

Full rendered-site audit (GPT-6.1; all 26 pages at 1280px and 390px) returned **FAIL** on two
blocking classes; both fixed and re-verified:

- Mobile horizontal overflow: wide inline and display math (plus one inline variance equation in
  B1). Fixed by converting wide inline expressions to display math, splitting wide display equations
  into aligned lines, enabling MathJax display line breaking, and adding container-level horizontal
  scroll for display math. Re-measured: all eight branch notes pages are exactly 390/390 at a 390px
  viewport (was 412px on B1).
- Mermaid labels scaling below readable size: 12 wide horizontal flows converted to vertical, two
  wide diagrams split, long node labels shortened on three diagrams, diagrams render at natural size
  with local scrolling, assistive MathML clipped, table scroll wrappers constrained. All 29 diagrams
  still render as SVG and labels are readable at desktop and mobile widths.
- Long SSE lines (209 chars) accepted: code blocks scroll inside their own container and do not
  widen the page.

## Final validation (parent)

- URL link check across all docs (2026-10-16, post-fixes): **215 checked, 215 ok, 0 broken**.
- MkDocs build `uvx --with mkdocs-material mkdocs build --site-dir <tmp>`: **zero WARNING/ERROR**; single benign INFO for `notes.md#architecture` (raw-HTML anchor present in built page, verified `id="architecture"` in built HTML).
- Independent structural counts (count_check.py): every branch 6-8 pen-and-paper + 4 coding problems, 3-5 Mermaid diagrams, 4 iframe embeds — all ≥ acceptance minimums; root syllabus table corrected to actual counts (B2/B6/B7 = 12).
- Video metadata: durations/oEmbed independently verified by reviewers for all 8 branches.
- Review verdicts: 8/8 PASS (GLM-5.3 rounds 1 on all branches + GPT-6.1 round 2 confirmation on Branch 6).







