# Branch 6 problems — Retrieval and controlled tool use

Use the [notes](notes.md) as your formula reference on the first attempt.
On a second attempt, work without opening them.
Use natural logarithms and one-based ranks unless a problem says otherwise.
Report decimal MB and GB unless you explicitly convert to binary units.

Difficulty: ● foundational, ●● applied, ●●● integrative.
There are **eight paper problems and four coding problems**.
The reference solutions sit inside collapsible blocks so that you can attempt each problem first.
The paper problems include intermediate steps, not only final values.

## Part A — Pen and paper

### A1 — Vector payload and PQ compression ●●

You store $N=1{,}000{,}000$ vectors with $d=768$ coordinates.
Ignore IDs, graph links, metadata, replicas, and allocator overhead at first.
Compare fp32 coordinates, ideal int8 coordinates, and PQ with $m=96$ subspaces and $b=8$ bits per subspace code.

1. Compute bytes per vector and total payload bytes for all three formats.
2. Compute the PQ subvector dimension and number of centroids per subspace.
3. Compute fp32 codebook storage for shared subspace codebooks.
4. Compute the payload-only compression ratios relative to fp32.
5. Explain why these numbers do not determine a database server's RAM requirement.

<details markdown="1">
<summary>Solution</summary>

**Step 1: count coordinate payloads.**
An fp32 coordinate takes four bytes.
Each fp32 vector takes $768\times4=3072$ bytes.
The collection takes $1{,}000{,}000\times3072=3{,}072{,}000{,}000$ bytes.
That is 3.072 GB in decimal units.

An ideal int8 coordinate takes one byte.
Each int8 vector takes 768 bytes.
The collection takes 768,000,000 bytes, or 768 MB.
Quantization scales and zero points are excluded by the stated idealization.

**Step 2: count PQ codes.**
Each vector stores $mb=96\times8=768$ bits.
Divide by eight to get 96 bytes per vector.
The collection's codes take 96,000,000 bytes, or 96 MB.
Each subvector has $768/96=8$ coordinates.
Each subspace codebook has $2^8=256$ centroids.

**Step 3: count shared codebook values.**
There are $96\times256\times8=196{,}608$ fp32 values.
They take $196{,}608\times4=786{,}432$ bytes.
Codes plus these codebooks take 96,786,432 bytes.
This still excludes any coarse IVF centroids or graph structure.

**Step 4: compute compression ratios.**
Int8 reduces the coordinate payload by $3072/768=4$ times.
PQ reduces the per-vector payload by $3072/96=32$ times.
Including the shared codebooks makes the collection-level PQ ratio slightly smaller than 32.
Do not count the codebook once per vector.

**Step 5: restore the deployment context.**
The server also stores IDs, metadata, index structures, temporary buffers, and process state.
Replicas multiply some persistent and resident data.
Some deployments retain original vectors for reranking or reconstruction.
The computed payload is a lower-level component, not a capacity plan.

</details>

### A2 — MRL truncation and HNSW selection ●●

An MRL-trained vector is $z=(3,4,12)$.
You select its trained two-coordinate prefix for cosine retrieval.
A separate HNSW experiment fixes `M` and `efConstruction` and measures the following query settings.

| `efSearch` | Recall@10 against flat search | p95 latency |
|---:|---:|---:|
| 16 | 0.83 | 4 ms |
| 64 | 0.94 | 8 ms |
| 128 | 0.97 | 13 ms |
| 256 | 0.98 | 23 ms |

1. Normalize the full vector and the selected prefix.
2. Explain why slicing the normalized full vector is not enough for a dot-product cosine implementation.
3. Select the lowest-latency setting with recall at least 0.95 and p95 at most 15 ms.
4. Explain what changing `M`, `efConstruction`, and `efSearch` changes.
5. State whether recall against flat search proves that users receive relevant evidence.

<details markdown="1">
<summary>Solution</summary>

**Step 1: compute norms.**
The full norm is $\sqrt{9+16+144}=13$.
The full unit vector is $(3/13,4/13,12/13)$.
The prefix norm is $\sqrt{9+16}=5$.
The prefix unit vector is $(3/5,4/5)=(0.6,0.8)$.

**Step 2: inspect the sliced vector.**
The first two coordinates of the full unit vector are $(3/13,4/13)$.
Their norm is $5/13$, not 1.
Renormalizing them yields $(0.6,0.8)$.
If you use their dot product as cosine without renormalization, coordinate-prefix norms distort scores.
Both query and document vectors need the same selected dimension and normalization rule.

**Step 3: apply both constraints.**
Settings 16 and 64 fail the recall threshold.
Setting 128 achieves 0.97 recall and 13 ms p95, so it passes both.
Setting 256 passes recall but fails the 15 ms latency threshold.
Choose `efSearch=128` from these measured options.

**Step 4: distinguish build and query controls.**
`M` controls graph connectivity and influences graph memory and construction work.
`efConstruction` controls insertion search breadth and affects the built graph's quality.
`efSearch` controls query-time search breadth and does not rebuild the graph.
Increasing either build parameter usually requires rebuilding or reconstructing the relevant index configuration.
Larger settings are not a guarantee of a fixed recall gain on every dataset.

**Step 5: separate two definitions of recall.**
Agreement with flat search measures nearest-neighbor approximation.
Flat search can retrieve semantically wrong documents if the embedding model is unsuitable.
Measure labeled evidence relevance as well as ANN recall.

</details>

### A3 — Recompute two BM25 scores ●●

Document A is `cat cat sat`.
Document B is `cat dog`.
The query is `cat`.
Use $k_1=1.2$, $b=0.75$, and

$$
\operatorname{IDF}(t)=\ln\left(1+\frac{N-n_t+0.5}{n_t+0.5}\right).
$$

1. Compute average document length and the IDF of `cat`.
2. Compute each document's length-normalization factor.
3. Compute both BM25 scores to six decimal places.
4. Explain why the longer document wins here.
5. Explain why `rank_bm25.BM25Okapi` can produce different raw scores on this exact corpus.

<details markdown="1">
<summary>Solution</summary>

**Step 1: count the corpus.**
There are $N=2$ documents.
Their lengths are 3 and 2, so average length is $(3+2)/2=2.5$.
Both contain `cat`, giving $n_{cat}=2$.
The specified IDF is $\ln(1+(2-2+0.5)/(2+0.5))=\ln(1.2)$.
Its value is approximately 0.1823215568.

**Step 2: compute length factors.**
For A, $1-b+b|A|/avgdl=0.25+0.75(3/2.5)=1.15$.
For B, $1-b+b|B|/avgdl=0.25+0.75(2/2.5)=0.85$.

**Step 3: substitute term frequencies.**
For A, $f(cat,A)=2$.
The numerator is $2(1.2+1)=4.4$.
The denominator is $2+1.2(1.15)=3.38$.
Thus $s_A=0.1823215568\times4.4/3.38\approx0.2373416716$.

For B, $f(cat,B)=1$.
The numerator is $1(1.2+1)=2.2$.
The denominator is $1+1.2(0.85)=2.02$.
Thus $s_B=0.1823215568\times2.2/2.02\approx0.1985680322$.

Rounded to six decimal places, the scores are **A: 0.237342; B: 0.198568**.
The ranking is A before B.

**Step 4: interpret the result.**
A pays a length penalty, but its second `cat` occurrence still adds enough saturated frequency weight to win.
BM25 does not simply prefer either the shortest document or the most repeated term.
Both factors interact through the denominator.

**Step 5: compare implementations honestly.**
`rank_bm25.BM25Okapi` starts from the IDF without the leading `1+` inside the logarithm.
It also handles negative IDF values with an epsilon-scaled average-IDF floor.
The problem explicitly chooses a different positive-IDF convention.
Match the convention before comparing numeric outputs.

</details>

### A4 — Fuse two retrieval lists ●

Lexical retrieval returns `[A, B, C]`.
Dense retrieval returns `[B, D, A]`.
Use RRF with one-based ranks and damping constant $k=60$.

1. Write each document's contribution from each list.
2. Compute all four fused scores to eight decimal places.
3. Give the final order.
4. Explain why a missing document contributes zero and why raw cosine scores do not enter this calculation.

<details markdown="1">
<summary>Solution</summary>

**Step 1: create rank pairs.**
A has ranks 1 and 3.
B has ranks 2 and 1.
C has lexical rank 3 and is absent from the dense list.
D is absent from the lexical list and has dense rank 2.

**Step 2: compute contributions.**
Rank 1 contributes $1/(60+1)=1/61\approx0.0163934426$.
Rank 2 contributes $1/62\approx0.0161290323$.
Rank 3 contributes $1/63\approx0.0158730159$.

**Step 3: add per-document scores.**
A receives $1/61+1/63=0.0322664585$.
B receives $1/62+1/61=0.0325224749$.
C receives only $1/63=0.0158730159$.
D receives only $1/62=0.0161290323$.

| Document | Lexical contribution | Dense contribution | Fused score, rounded |
|---|---:|---:|---:|
| A | $1/61$ | $1/63$ | 0.03226646 |
| B | $1/62$ | $1/61$ | 0.03252247 |
| C | $1/63$ | 0 | 0.01587302 |
| D | 0 | $1/62$ | 0.01612903 |

**Step 4: sort.**
The final order is **B, A, D, C**.
RRF uses only ranks, which avoids treating unrelated lexical and dense score scales as comparable.
An absent document has no observed rank in that list.
Assigning it a made-up last rank would give every absent document unsupported credit.

</details>

### A5 — Choose a query transformation and evidence boundary ●●

A user asks, “Which of plans A and B costs less after the annual discount, and which includes audit logs?”
The corpus has separate pricing and feature pages.
The retriever misses “annual” because the pricing page says “yearly subscription.”

1. Propose a query expansion that preserves the original question.
2. Decompose the question into independent and dependent steps.
3. Explain how HyDE could help and how it could harm.
4. State which text may appear as cited evidence in the final answer.
5. Explain when a reranker can and cannot fix the failure.

<details markdown="1">
<summary>Solution</summary>

**Step 1: expand terminology without replacing intent.**
Keep the original query as one retrieval branch.
Add a branch containing “yearly subscription discount pricing plan A plan B.”
Use the same plan identifiers and do not invent a discount percentage.
Fuse the branch results by stable evidence IDs.

**Step 2: identify independent lookups.**
Retrieve plan A's base annual price and discount rules.
Retrieve plan B's base annual price and discount rules.
Retrieve each plan's audit-log feature description.
Those lookups can run concurrently if the plan identifiers are already resolved.

**Step 3: identify dependent computation.**
Normalize prices to the same currency, billing period, seat count, and tax assumptions.
Apply only discounts that the source states are eligible.
Then compare effective costs and join the feature results.
The arithmetic depends on the retrieved values and cannot precede them.

**Step 4: evaluate HyDE.**
A hypothetical pricing explanation may use wording close to the real pricing page.
Its embedding may therefore retrieve the missing yearly-subscription section.
It can also invent a discount or feature and direct retrieval toward the wrong plan.
Treat the generated passage only as a retrieval probe.

**Step 5: preserve evidence provenance.**
Cite actual retrieved pricing and feature documents, including their versions and conditions.
Do not cite the generated HyDE passage.
Show the calculation separately from quoted source facts.
If eligibility remains unclear, qualify the comparison or ask a question.

**Step 6: locate the reranker's limit.**
A reranker can promote a pricing page already in its candidate set.
It cannot promote a page that no retrieval branch returns.
Fix candidate recall before claiming that a stronger reranker solves the problem.

</details>

### A6 — Diagnose the RAG triad ●●

The question is “How many hours before departure must I cancel?”
The retrieved context says, “Policy v3 requires cancellation at least 48 hours before departure.”
Consider three answers:

- A: “You must cancel at least 48 hours before departure.”
- B: “You must cancel 24 hours before departure.”
- C: “Policy v3 discusses cancellation.”

1. Assess context relevance, groundedness, and answer relevance for each answer.
2. A separate answer contains four extracted factual claims; the context supports three.
   Compute a simple claim-support fraction.
3. Explain why that fraction is not a guarantee of factual truth.
4. Propose a human calibration check for an automated judge.

<details markdown="1">
<summary>Solution</summary>

**Step 1: hold context relevance separate.**
The retrieved context directly addresses the cancellation threshold.
Its context relevance is high for all three cases because the question and retrieved passage stay the same.
Changing the answer does not retroactively change what the retriever returned.

**Step 2: assess A.**
A repeats the supported threshold and directly answers the question.
It has high groundedness and high answer relevance.
Its source still needs a version and authority check in a real system.

**Step 3: assess B.**
B addresses the requested quantity but contradicts the retrieved threshold.
Its answer relevance can be high while its groundedness is low.
Relevance does not mean correctness.

**Step 4: assess C.**
C makes a supported but weak claim about the policy's topic.
It is grounded, but its answer relevance or completeness is low because it omits the requested number.
Different evaluation rubrics can split relevance and completeness into separate metrics.
State the rubric rather than assuming one judge label captures both.

**Step 5: compute claim support.**
The fraction is $3/4=0.75$.
This assumes the extracted claims have equal weight and that all four claims are evaluable.
Real metrics need explicit handling for empty answers and unsupported judge outputs.

**Step 6: limit the interpretation.**
The source could be obsolete or false.
Claim extraction could omit a harmful qualification.
The judge could misclassify entailment.
Therefore, 0.75 is an estimate under a particular evaluation process, not a factual-truth probability.

**Step 7: calibrate.**
Have a human label a small held-out set of supported, contradicted, and irrelevant answers.
Compare judge decisions with those labels, inspect disagreements, and record the judge model and prompt.
Include cases where the source contains negation, exceptions, or conflicting versions.
Do not tune the judge on every example you later use as final evidence of quality.

</details>

### A7 — Diagnose a wrong-tool ReAct trace ●●●

The user asks, “Does policy v3 allow cancellation 36 hours before departure?”
Available tools are `search_policy(query)`, `weather(city)`, and `delete_booking(id)`.
The model produces this trace:

| Step | Proposal or observation |
|---|---|
| 1 | `weather({"city":"Paris"})` |
| 2 | Observation: `{"temperature_c":18}` |
| 3 | `weather({"city":"Paris"})` |
| 4 | Observation: `{"temperature_c":18}` |
| 5 | Answer: “Yes, cancellation is allowed because the weather is mild.” |

1. Identify the first failure and explain why schema validation does not catch it.
2. Identify the retry and final-answer failures.
3. Write a corrected bounded trace using the fact that policy v3 requires at least 48 hours.
4. State whether `delete_booking` should execute if the user asks only about policy.
5. Name the minimum control fields you record without storing hidden reasoning.

<details markdown="1">
<summary>Solution</summary>

**Step 1: locate semantic misrouting.**
The first weather call does not retrieve the policy threshold.
Its JSON can be syntactically and structurally valid while its action remains irrelevant.
Schema validation checks permitted fields and types, not whether weather answers a cancellation question.
A tool-routing evaluator or explicit policy-lookup route can catch this mismatch.

**Step 2: distinguish the loop failure.**
The second identical weather call adds no relevant information.
An iteration cap bounds the damage but does not repair the route.
The controller should detect lack of progress, correct the tool selection, or stop.

**Step 3: identify unsupported generation.**
The mild-weather observation contains no evidence about cancellation rules.
The final answer invents a policy relationship.
It fails groundedness and factual correctness in this scenario.

**Step 4: produce a corrected trace.**
First call `search_policy({"query":"policy v3 cancellation deadline"})`.
Observe `policy-v3#cancel`: “Cancel at least 48 hours before departure.”
Compare the provided 36-hour interval with the 48-hour threshold.
Return “No under this rule: 36 hours is below the 48-hour minimum,” with the evidence ID.
Stop after this successful lookup and comparison.
If retrieval fails within the budget, report the missing evidence instead of guessing.

**Step 5: preserve the action boundary.**
The user asks for information, not a cancellation action.
Do not call `delete_booking`.
Even an explicit cancellation request needs authorization, a precise target, and any required approval.

**Step 6: record observable control data.**
Record request ID, caller scope, selected tool, validated arguments, call ID, timing, result status, evidence IDs, attempt count, and stop reason.
Redact secrets and unnecessary personal data.
Use short decision summaries rather than private chain-of-thought text.

</details>

### A8 — Constrained decoding and checkpointed approval ●●●

At one generation step, the next-token logits are 2 for token A, 1 for token B, and 0 for token C.
A grammar permits only B and C.
Later, a graph proposes deleting resource `R7` and pauses for human approval.
The graph uses a persistent checkpointer.

1. Compute the masked probability of each token.
2. Explain what the mask guarantees and what it does not.
3. List the fields the human approval must bind to.
4. Explain why persistent checkpoints do not guarantee exactly-once deletion.
5. Place the deletion relative to the interrupt and describe an idempotency strategy.

<details markdown="1">
<summary>Solution</summary>

**Step 1: apply the grammar mask.**
Replace A's logit with $-\infty$.
Keep B's logit at 1 and C's logit at 0.
The denominator is $e^1+e^0=e+1$.
Thus $p(A)=0$, $p(B)=e/(e+1)\approx0.73105858$, and $p(C)=1/(e+1)\approx0.26894142$.

**Step 2: separate syntax from intent.**
The decoder cannot choose A at this step under the implemented grammar.
It does not prove that B is factually correct or authorized.
The final output still needs parsing, validation, authorization, and error handling.
An unsupported schema feature may require a separate application check.

**Step 3: bind approval precisely.**
Bind approval to the request ID, action name, resource `R7`, validated arguments or payload hash, approver identity, and approval time.
Include any action preconditions or source version that affects the decision.
If the target or arguments change, require approval again.
Check that the approver has permission to delete the resource.

**Step 4: examine the crash window.**
The deletion can succeed before the process records its result in a later checkpoint.
After restart, replay can reach the deletion again.
A persistent state snapshot alone cannot atomically control an unrelated external service.

**Step 5: place and protect the side effect.**
Interrupt before deletion, not after it.
On resume, validate the decision and recheck authorization and action preconditions.
Use a stable operation ID as an idempotency key at the target service.
If a timeout leaves the outcome unknown, query that operation's status before retrying.
Store the confirmed outcome and stop reason in the graph state.

</details>

## Part B — Code

Run each reference solution as its own script in the environment described in the [notes](notes.md).
All four coding problems run locally without provider credentials.
The Ragas example in the notes is optional when you lack an authorized evaluation endpoint.
Expected outputs describe public behavior, not private library internals.

### B1 — Cosine retrieval with a normalized MRL prefix ●●

Implement `top_k` over a NumPy matrix of document vectors.
Accept a nonzero query and a valid k between one and the number of documents.
Reject zero vectors instead of returning NaN scores.
Keep original row order when scores tie.
Then normalize the trained two-coordinate prefix of `(3, 4, 12)`.

Use these inputs:

- Document rows: `(1, 0, 0)`, `(1, 1, 0)`, `(0, 1, 0)`.
- Query: `(1, 0, 0)`.
- `k=2`.

<details markdown="1">
<summary>Reference solution</summary>

```python
import numpy as np

def top_k(documents: np.ndarray, query: np.ndarray, k: int):
    documents = np.asarray(documents, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if documents.ndim != 2 or query.ndim != 1:
        raise ValueError("expected a matrix and a vector")
    if documents.shape[1] != query.shape[0]:
        raise ValueError("dimension mismatch")
    if not 1 <= k <= documents.shape[0]:
        raise ValueError("invalid k")
    document_norms = np.linalg.norm(documents, axis=1)
    query_norm = np.linalg.norm(query)
    if np.any(document_norms == 0) or query_norm == 0:
        raise ValueError("cosine needs nonzero vectors")
    scores = (documents @ query) / (document_norms * query_norm)
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), round(float(scores[i]), 4)) for i in order]

documents = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]])
print(top_k(documents, np.array([1, 0, 0]), k=2))
prefix = np.array([3, 4, 12], dtype=np.float64)[:2]
print((prefix / np.linalg.norm(prefix)).tolist())
try:
    top_k(documents, np.zeros(3), k=2)
except ValueError as error:
    print(str(error))
```

Expected output:

```text
[(0, 1.0), (1, 0.7071)]
[0.6, 0.8]
cosine needs nonzero vectors
```

The matrix-vector multiplication produces one score per document row.
The denominator has the same row count because each document has its own norm.
The query norm is a scalar.
The stable sort preserves row order for equal scores.
Prefix normalization demonstrates the arithmetic only; arbitrary vectors do not prove an MRL retrieval guarantee.

</details>

### B2 — BM25 arithmetic and RRF fusion ●●

Implement the positive-IDF BM25 formula from A3 using the Python standard library.
Do not substitute a package with a different IDF convention.
Print the two scores rounded to six decimal places.
Then fuse the two A4 rankings with one-based RRF ranks and print document IDs in fused order.
Remove duplicates within each list before assigning ranks.

<details markdown="1">
<summary>Reference solution</summary>

```python
import math
from collections import Counter, defaultdict

def bm25_scores(corpus: list[str], query: str, k1=1.2, b=0.75):
    documents = [text.split() for text in corpus]
    average_length = sum(map(len, documents)) / len(documents)
    query_terms = set(query.split())
    scores = []
    for document in documents:
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            document_frequency = sum(term in item for item in documents)
            idf = math.log1p(
                (len(documents) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            frequency = frequencies[term]
            length_factor = 1 - b + b * len(document) / average_length
            score += idf * frequency * (k1 + 1) / (
                frequency + k1 * length_factor
            )
        scores.append(score)
    return scores

def fuse(rankings: list[list[str]], k=60):
    if k <= 0:
        raise ValueError("k must be positive")
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, document_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[document_id] += 1 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))

scores = bm25_scores(["cat cat sat", "cat dog"], "cat")
print([round(score, 6) for score in scores])
print(fuse([["A", "B", "C"], ["B", "D", "A"]]))
print(fuse([["A", "A", "B"], ["B"]]))
```

Expected output:

```text
[0.237342, 0.198568]
['B', 'A', 'D', 'C']
['B', 'A']
```

The last call shows that repeating A inside one list does not grant it extra votes.
This exercise assumes a nonempty corpus with positive average length and the valid parameter values given in A3.
A reusable search service also needs an explicit empty-corpus and invalid-parameter policy.
The implementation favors visible arithmetic over indexing efficiency.
Do not scan all documents to recompute document frequency inside every online query in production.

</details>

### B3 — Reject malformed and unauthorized tool proposals ●●●

Build a bounded dispatcher for a read-only `search_policy` tool.
Require a query of length 1–100, an integer limit of 1–3, and a tenant argument.
Use strict Pydantic validation and reject unknown fields or tool names.
The authenticated caller belongs to tenant `acme`.
Do not execute a call that names another tenant.

Process three proposals in order:

1. An `acme` call with limit 9.
2. An `other` tenant call with limit 1.
3. An `acme` call with limit 1.

For this exercise, continue through the fixed proposal fixture after each rejection.
A live system must not retry an authorization denial by asking the model to guess another identity.
Execute at most one valid authorized call and stop after at most three proposals.

<details markdown="1">
<summary>Reference solution</summary>

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError

class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tenant: str
    query: str = Field(min_length=1, max_length=100)
    limit: int = Field(ge=1, le=3)

class Call(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: Literal["search_policy"]
    arguments: Arguments

def search_policy(arguments: Arguments):
    return [f"{arguments.tenant}:policy-v3"][:arguments.limit]

authenticated_tenant = "acme"
dispatch = {"search_policy": search_policy}
proposals = [
    '{"name":"search_policy","arguments":{"tenant":"acme","query":"cancel","limit":9}}',
    '{"name":"search_policy","arguments":{"tenant":"other","query":"cancel","limit":1}}',
    '{"name":"search_policy","arguments":{"tenant":"acme","query":"cancel","limit":1}}',
]
for attempt, raw in enumerate(proposals[:3], start=1):
    try:
        call = Call.model_validate_json(raw)
    except ValidationError:
        print(f"{attempt}: invalid")
        continue
    if call.arguments.tenant != authenticated_tenant:
        print(f"{attempt}: denied")
        continue
    print(f"{attempt}: {dispatch[call.name](call.arguments)}")
    break
else:
    print("stopped")
```

Expected output:

```text
1: invalid
2: denied
3: ['acme:policy-v3']
```

The invalid limit fails before dispatch.
The second proposal passes structural validation but fails application authorization.
The third proposal reaches the allowlisted function.
The tenant identity comes from authenticated application context, not from model assertion.
In a real retrieval service, pass the authenticated tenant into the storage filter even after this comparison.
Keep the model from choosing a credential or bypassing row-level access policy.

</details>

### B4 — Interrupt, approve, and resume a LangGraph workflow ●●●

Create a graph with an approval node and an execution node.
Use `InMemorySaver` so the exercise remains local.
The proposed action is `archive:report-7`.
Execution only returns a demonstration string; it must not touch a filesystem or external service.

Run two isolated threads.
Pause each thread with `interrupt`, then resume the first with `True` and the second with `False`.
Only the first thread may reach the execution node.
Print the final result for each thread.
Explain why this exercise does not demonstrate durable recovery or exactly-once side effects.

<details markdown="1">
<summary>Reference solution</summary>

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

class State(TypedDict):
    action: str
    approved: bool
    result: str

def approve(state: State):
    decision = interrupt({"action": state["action"], "question": "Approve?"})
    if decision is not True and decision is not False:
        raise ValueError("approval must be a boolean")
    return {"approved": decision}

def route(state: State):
    return "execute" if state["approved"] else "stop"

def execute(state: State):
    return {"result": f"demo executed {state['action']}"}

builder = StateGraph(State)
builder.add_node("approve", approve)
builder.add_node("execute", execute)
builder.add_edge(START, "approve")
builder.add_conditional_edges(
    "approve", route, {"execute": "execute", "stop": END}
)
builder.add_edge("execute", END)
graph = builder.compile(checkpointer=InMemorySaver())

for thread_id, decision in [("accept", True), ("deny", False)]:
    config = {"configurable": {"thread_id": thread_id}}
    paused = graph.invoke(
        {"action": "archive:report-7", "approved": False, "result": "not executed"},
        config,
    )
    print(f"{thread_id}: paused={bool(paused.get('__interrupt__'))}")
    final = graph.invoke(Command(resume=decision), config)
    print(f"{thread_id}: {final['result']}")
```

Expected output:

```text
accept: paused=True
accept: demo executed archive:report-7
deny: paused=True
deny: not executed
```

The same compiled graph serves both threads, but separate thread IDs isolate their checkpointed state.
Each resume uses the corresponding thread's configuration.
The route sends a denial directly to the end state.
The execution node returns a string only, so the exercise performs no real archival operation.

`InMemorySaver` loses every checkpoint when the process exits.
A persistent saver would preserve state but still would not atomically commit an external archive action with a graph checkpoint.
A real action needs authenticated approval, target authorization, immutable action binding, and an idempotency key.
Do not place the external operation before `interrupt`, because the interrupted node can rerun from its beginning.

</details>

## Review your work

Recompute A1, A3, and A4 independently after a break.
Those problems expose unit mistakes, scoring-convention mistakes, and rank-index mistakes.
Run B1–B4 and compare their observable outputs with the stated outputs.
Explain one rejected tool proposal and one denied approval in your own words.

Use the [exit criteria](index.md#exit-criteria) before starting the next branch.
Keep your evidence and tool contracts for the [Branch 7 agentic-service work](../branch-07-llmops-serving/index.md).
