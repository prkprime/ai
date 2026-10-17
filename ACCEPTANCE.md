# Acceptance Criteria — review checklist

Reviewers use this checklist verbatim.
Report every criterion as PASS or FAIL with file-and-quote evidence.
A FAIL in Section A or Section B blocks the branch.
Section C produces warnings only, but list them.
The reviewer must independently verify correctness; reading the author's self-assessment is not evidence.

## Section A — Coverage and structure (blocking)

| ID | Criterion | Evidence required |
|----|-----------|-------------------|
| A1 | Every subtopic from the branch outline appears in `notes.md` with substantive prose coverage (not only bullet lists, not only definitions). | Quote or anchor per subtopic. |
| A2 | Subtopic hour estimates in `index.md` sum exactly to the branch's allotted hours, and each subtopic in the outline has an hour estimate. | Show the arithmetic. |
| A3 | `index.md` states measurable learning objectives (verb-first: derive, implement, diagnose, ...), at least one per subtopic. | Quote objectives; flag non-measurable verbs ("understand", "know"). |
| A4 | `index.md` lists prerequisites with links to the specific earlier branch sections that provide them. | Quote the prerequisite list. |
| A5 | `index.md` contains an hour-by-hour or session-by-session schedule table covering the branch hours. | Show the table rows summing to the allotment. |
| A6 | Every subtopic maps to at least one primary source with a working URL, in `notes.md` or the "Key sources" table in `index.md`. | Count sources per subtopic. |
| A7 | `problems.md` exists with Part A (pen-and-paper) and Part B (coding), and solutions live in collapsible `<details markdown="1">` blocks (the `markdown="1"` attribute is required so MkDocs renders the solution content; bare `<details>` renders raw under MkDocs). | Quote structure. |

## Section B — Content quality (blocking)

| ID | Criterion | Evidence required |
|----|-----------|-------------------|
| B1 | Mathematical statements are correct. The reviewer must independently recompute at least three formulas or solutions (state which). | Show the check. |
| B2 | At least 10 problems: ≥ 6 pen-and-paper with complete step-by-step solutions, ≥ 4 coding with reference solutions and expected output or observable behavior. | Count; spot-check two solutions end to end. |
| B3 | Code is correct and current (APIs as of 2026). The reviewer must trace at least two snippet-level code blocks line by line (shapes, dtypes, function signatures) and flag anything stale or wrong. Running code is allowed but not required. | Name the snippets reviewed. |
| B4 | At least 3 diagrams, each valid Mermaid or clearly-structured ASCII, and each consistent with the surrounding text. Reviewer checks Mermaid syntax (labels quoted where needed, no reserved characters). | Name each diagram and the matching text claim. |
| B5 | At least 4 video resources in `index.md`, each with correct title, channel, duration (or "playlist"), a working link, and one line stating why it earns study time. Each video also has an inline iframe embed (`https://www.youtube.com/embed/...`, title attribute set) in an "Embedded playlist" subsection; GitHub may strip iframes — the MkDocs build must show them. Reviewer verifies at least 2 links resolve. | List videos and verification results. |
| B6 | No fabricated citations. The reviewer spot-checks at least 5 sources or factual claims against the live source (fetch the URL or verify the paper/authors/year exist). | List what was checked. |
| B7 | Numeric worked examples are arithmetically correct (memory calculations, dimensional analysis, score computations). Reviewer recomputes at least two numbers. | Show recomputation. |

## Section C — Style and navigation (warnings only)

| ID | Criterion |
|----|-----------|
| C1 | Style: active voice, present tense, short sentences, jargon defined at first use, no exclamation marks, sentence-case headings. |
| C2 | Relative links between branches resolve (`../branch-NN-slug/file.md` targets exist). |
| C3 | Consistent structure across the branch's three files (frontmatter-free plain Markdown is fine; heading hierarchy sane: one `#` per file). |
| C4 | Reading flow: a motivated learner can study the branch start to finish without external scaffolding beyond the cited sources. |

## Verdict format

Reviewers must reply exactly in this shape (one block per branch):

```
BRANCH: <number and name>
FILES REVIEWED: <paths + line counts>
ROUND: <n>
VERDICT: PASS | FAIL
A1..A7: PASS/FAIL — <one-line evidence>
B1..B7: PASS/FAIL — <one-line evidence>
C1..C4: PASS/WARN — <one-line evidence>
ISSUES (only for FAILs; ordered by severity):
1. <file>: <quote or location> — <what is wrong> — <suggested fix>
2. ...
COUNT VERIFICATION: <recounted totals vs claimed totals>
```
