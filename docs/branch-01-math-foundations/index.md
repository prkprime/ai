# Branch 1 — Mathematical Foundations & Production Engineering

Use this eight-hour branch to connect matrix calculus, statistical objectives, and reliable Python execution.
You derive the gradients that later branches use, interpret a language model's loss, and build a small validated computation pipeline.
Study the [notes](notes.md), then attempt the [problems](problems.md) without opening their solutions.
Return to the [syllabus overview](../index.md) for the full learning path.

## Learning objectives

- Derive projections, a small singular value decomposition, dense-layer gradients, and attention gradient shapes for 1.1.
- Compute Bernoulli and Gaussian maximum-likelihood estimates and distinguish them from maximum a posteriori estimates for 1.2.
- Derive cross-entropy as entropy plus KL divergence and convert mean negative log-likelihood to perplexity for 1.2.
- Select an execution model for I/O-bound and CPU-bound Python workloads for 1.3.
- Implement and profile a vectorized numerical kernel, validate its boundary with Pydantic v2, and check its contract with pytest and CI for 1.3.

## Prerequisites

This is the first branch.
It has no earlier-branch prerequisites.
Complete prerequisite refreshers before starting the eight-hour clock.

- Write Python functions, classes, imports, and context managers; use the [official Python documentation](https://docs.python.org/3/library/asyncio.html) to locate unfamiliar asynchronous constructs.
- Multiply small matrices and differentiate a polynomial; review chapters 2 and 5 of [Mathematics for Machine Learning](https://mml-book.github.io/) if needed.
- Create a virtual environment and run a Python file from a terminal.
- Read NumPy shapes such as `(batch, features)` and distinguish an array view from a copy using the [NumPy guide](https://numpy.org/doc/stable/user/basics.copies.html).

## Time budget

| Subtopic | Hours | Evidence you produce |
|---|---:|---|
| 1.1 Vector calculus & linear algebra | 3 | Projection, SVD, and gradient derivations |
| 1.2 Probability, statistics & information theory | 2 | Estimates, information measures, and perplexity |
| 1.3 High-performance Python & software engineering | 3 | Executable profiling and validation lab |
| **Total** | **8** | **3 + 2 + 3 = 8** |

### Session schedule

Each row covers one hour, including practice.
Read the indicated notes, not the linked books in full.
Videos fit within these hours; use the remaining time for the written derivations.
Set up the environment before session 1.

| Hour | Subtopic | Study and practice |
|---|---|---|
| 1 | 1.1 | Vectors, rank, projections; first two videos; problem A1 |
| 2 | 1.1 | Eigen-decomposition, SVD, gradients; last two videos; problem A2 |
| 3 | 1.1 | Dense and attention backpropagation; problems A3 and B1 |
| 4 | 1.2 | Distributions, moments, MLE and MAP; problem A4 |
| 5 | 1.2 | Entropy, KL, cross-entropy, perplexity; problems A5 and A6 |
| 6 | 1.3 | Asyncio, threads, processes, and the GIL; problem B2 |
| 7 | 1.3 | NumPy memory, Polars, Numba; problem B3 |
| 8 | 1.3 | Architecture, typing, Pydantic v2, pytest, CI; problem B4 and exit check |
| **Total** | | **8 hours** |

## Subtopic map

| Subtopic | Notes | Practice | Primary-source route |
|---|---|---|---|
| 1.1 | [Vector calculus & linear algebra](notes.md#11-vector-calculus-linear-algebra) | [Part A](problems.md#part-a-pen-and-paper), A1–A3; B1 | MML chapters 2–5; Stanford linear algebra review |
| 1.2 | [Probability, statistics & information theory](notes.md#12-probability-statistics-information-theory) | A4–A6 | MML chapter 6; Murphy; Stanford probability review |
| 1.3 | [High-performance Python & software engineering](notes.md#13-high-performance-python-software-engineering) | [Part B](problems.md#part-b-code), B2–B4 | Python, NumPy, Polars, Numba, Pydantic, pytest, GitHub docs |

## Video playlist

Watch these four videos in order.
Titles and channels match YouTube's oEmbed metadata.
Durations come from each video's player metadata.
All links resolve on 2026-10-16.

| Title | Channel | Duration | Link | Why it earns study time |
|---|---|---:|---|---|
| Vectors — Chapter 1, Essence of linear algebra | 3Blue1Brown | 9:51 | [Watch](https://www.youtube.com/watch?v=fNk_zzaMoSs) | Relate coordinates to geometric vectors before manipulating tensors. |
| Linear combinations, span, and basis vectors — Chapter 2, Essence of linear algebra | 3Blue1Brown | 9:59 | [Watch](https://www.youtube.com/watch?v=k7RM-ot2NWY) | Connect rank to the dimension of a span. |
| Linear transformations and matrices — Chapter 3, Essence of linear algebra | 3Blue1Brown | 10:58 | [Watch](https://www.youtube.com/watch?v=kYB8IZa5AuE) | Interpret matrix columns as transformed basis vectors. |
| Eigenvectors and eigenvalues — Chapter 14, Essence of linear algebra | 3Blue1Brown | 17:15 | [Watch](https://www.youtube.com/watch?v=PFDu9oVAE-g) | Visualize invariant directions before computing an eigen-decomposition. |

Total viewing time is 48 minutes and 3 seconds.
The [creator's linear algebra collection](https://www.3blue1brown.com/topics/linear-algebra) provides optional context outside the core schedule.

## Embedded playlist

Play the same four videos below when you use the built site.
Use the verified links above when your Markdown viewer removes embedded players.

<iframe width="560" height="315" src="https://www.youtube.com/embed/fNk_zzaMoSs" title="Vectors — Chapter 1, Essence of linear algebra — 3Blue1Brown" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/k7RM-ot2NWY" title="Linear combinations, span, and basis vectors — Chapter 2, Essence of linear algebra — 3Blue1Brown" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/kYB8IZa5AuE" title="Linear transformations and matrices — Chapter 3, Essence of linear algebra — 3Blue1Brown" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/PFDu9oVAE-g" title="Eigenvectors and eigenvalues — Chapter 14, Essence of linear algebra — 3Blue1Brown" frameborder="0" allowfullscreen></iframe>

## Key sources

Every source below is accessed 2026-10-16.
Use the specific chapters or documentation topics listed rather than treating the full collection as required reading.
The official documentation replaces secondary tutorials when those tutorials do not pass the live URL check.

| Author or organization | Title | Type | Link | What it covers | Access date |
|---|---|---|---|---|---|
| Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong | Mathematics for Machine Learning | Open textbook | [Book](https://mml-book.github.io/) | 1.1: chapters 2–5; 1.2: chapter 6 | accessed 2026-10-16 |
| Stanford CS229 | Linear algebra review and reference | Course notes | [PDF](https://cs229.stanford.edu/section/cs229-linalg.pdf) | 1.1: rank, eigenvectors, matrix calculus | accessed 2026-10-16 |
| Stanford CS229 | Probability theory review | Course notes | [PDF](https://cs229.stanford.edu/section/cs229-prob.pdf) | 1.2: distributions, expectations, covariance | accessed 2026-10-16 |
| Kevin P. Murphy | Probabilistic Machine Learning: An Introduction | Textbook portal | [Book](https://probml.github.io/pml-book/) | 1.2: likelihood, Bayesian estimation, information theory | accessed 2026-10-16 |
| Joseph K. Blitzstein, Jessica Hwang | Introduction to Probability | Author-provided book | [Book](https://probabilitybook.net/) | 1.2: distributions, conditioning, moments; alternative to the blocked Stat 110 site | accessed 2026-10-16 |
| Grant Sanderson / 3Blue1Brown | Essence of linear algebra | Video collection | [Collection](https://www.3blue1brown.com/topics/linear-algebra) | 1.1: geometric intuition | accessed 2026-10-16 |
| Luciano Ramalho | Fluent Python, second edition | Companion site | [Site](https://www.fluentpython.com/) | 1.3: Python data model and concurrency context | accessed 2026-10-16 |
| Python Software Foundation | asyncio | Official documentation | [Docs](https://docs.python.org/3/library/asyncio.html) | 1.3: cooperative concurrency and tasks | accessed 2026-10-16 |
| Python Software Foundation | threading | Official documentation | [Docs](https://docs.python.org/3/library/threading.html) | 1.3: threads and CPython GIL qualifications | accessed 2026-10-16 |
| Python Software Foundation | concurrent.futures | Official documentation | [Docs](https://docs.python.org/3/library/concurrent.futures.html) | 1.3: thread and process executors | accessed 2026-10-16 |
| Python Software Foundation | tracemalloc | Official documentation | [Docs](https://docs.python.org/3/library/tracemalloc.html) | 1.3: traced allocation peaks and limitations | accessed 2026-10-16 |
| NumPy contributors | Copies and views | Official documentation | [Docs](https://numpy.org/doc/stable/user/basics.copies.html) | 1.3: buffers, views, and copies | accessed 2026-10-16 |
| NumPy contributors | numpy.linalg.svd | API reference | [Docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) | 1.1: reduced SVD and return shapes | accessed 2026-10-16 |
| Polars contributors | Polars user guide | Official documentation | [Docs](https://docs.pola.rs/) | 1.3: expressions and lazy execution | accessed 2026-10-16 |
| Numba contributors | Performance tips | Official documentation | [Docs](https://numba.readthedocs.io/en/stable/user/performance-tips.html) | 1.3: compiled loops and numerical tradeoffs | accessed 2026-10-16 |
| Pydantic contributors | Pydantic validation documentation | Official documentation | [Docs](https://docs.pydantic.dev/latest/) | 1.3: v2 models, strict fields, validation | accessed 2026-10-16 |
| pytest contributors | pytest documentation | Official documentation | [Docs](https://docs.pytest.org/en/stable/) | 1.3: fixtures, assertions, exceptions | accessed 2026-10-16 |
| GitHub | GitHub Actions documentation | Official documentation | [Docs](https://docs.github.com/en/actions) | 1.3: CI jobs, permissions, and workflow execution | accessed 2026-10-16 |

## Exit criteria

- [ ] Solve at least five of the six Part A problems without reading the solutions first.
- [ ] Reconstruct the small SVD and verify the orthogonality of its factors.
- [ ] Derive softmax-cross-entropy and state every attention gradient shape.
- [ ] Explain how a prior changes an MLE into a MAP estimate.
- [ ] Convert mean token NLL to perplexity with the same logarithm base.
- [ ] Run all four coding exercises and match their observable contracts.
- [ ] Report a warmed-up timing and array payload size without promising a speedup.
- [ ] Reject invalid prediction input and run the supplied pytest contract checks.
- [ ] Explain why a CI check does not by itself authorize a model deployment.

Continue through the [global branch map](../index.md#the-60-hours-at-a-glance) after you meet these criteria.
