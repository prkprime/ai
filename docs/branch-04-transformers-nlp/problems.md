# Branch 4 problems — Transformers & Modern NLP

Attempt each problem before revealing its solution.
Use [the notes](notes.md) for notation and [the overview](index.md) for source links.
Keep a record of shapes, assumptions, and intermediate numbers.
Give each pen-and-paper problem equal weight and score at least 70% before progressing.
Reproduce all four coding exercises without importing an attention implementation.

Difficulty: ● foundational, ●● applied, ●●● integrative.
All code solutions run on a CPU.
Use `uv run --with numpy --with torch --with tokenizers python solution.py` for a saved solution block.
Each solution is self-contained; you do not need variables from another problem.

## Part A — Pen and paper

### A1. Trace weighted BPE merges · ●●

**Subtopic: 4.1.**
Train three BPE merges on `low` with frequency 2 and `lower` with frequency 1.
Start from characters and append `</w>` to each word.
Do not merge across words.
Break ties lexicographically on the first symbol, then the second.
List every initial adjacent pair and its weighted frequency.
Show the corpus after each merge and explain how inference uses the result.

<details markdown="1">
<summary>Solution</summary>

1. Start with `l o w </w>` at weight 2 and `l o w e r </w>` at weight 1.
2. Count pairs within each representation, then multiply by the representation's frequency.
3. Select and replace the highest-count pair, recomputing adjacency after every replacement.

| Initial pair | Contribution from `low` | Contribution from `lower` | Total |
|---|---:|---:|---:|
| `l o` | 2 | 1 | 3 |
| `o w` | 2 | 1 | 3 |
| `w </w>` | 2 | 0 | 2 |
| `w e` | 0 | 1 | 1 |
| `e r` | 0 | 1 | 1 |
| `r </w>` | 0 | 1 | 1 |

The tie rule chooses `l o` because `l` precedes `o`.
After merge 1, the corpus is `lo w </w>` with weight 2 and `lo w e r </w>` with weight 1.
The pair `lo w` now has weighted count 3.
The remaining counts are `w </w>`: 2, `w e`: 1, `e r`: 1, and `r </w>`: 1.

Merge 2 combines `lo w` into `low`.
The corpus becomes `low </w>` with weight 2 and `low e r </w>` with weight 1.
Now the pair `low </w>` has count 2.
The pairs `low e`, `e r`, and `r </w>` each have count 1.

Merge 3 combines `low </w>` into `low</w>`.
The final representations are `low</w>` and `low e r </w>`.
The learned ranked merge list is `(l,o)`, `(lo,w)`, `(low,</w>)`.
Inference applies those learned ranks to new inputs; it does not learn new frequencies from the input.
The marker and initial alphabet remain part of the tokenizer contract.

</details>

### A2. Compare segmentation and embedding objectives · ●●

**Subtopic: 4.1.**
A Unigram tokenizer segments `ab` as either `[ab]` or `[a,b]`.
Its piece probabilities are $p(ab)=0.4$, $p(a)=0.3$, and $p(b)=0.3$.
Compute the string probability, each posterior segmentation probability, and each expected piece count for one occurrence.
Then evaluate one skip-gram negative-sampling loss with one positive dot product of zero and two negative dot products of zero.
Finally compute one GloVe residual-squared term with $X_{ij}=4$, dot product zero, both biases zero, and weight $f(4)=1$.
Explain why a byte-covered tokenizer still does not guarantee short token sequences for every language.

<details markdown="1">
<summary>Solution</summary>

1. The one-piece path has probability $0.4$.
   The two-piece path has probability $0.3\cdot0.3=0.09$.
2. Marginalize over paths: $P(\text{ab})=0.4+0.09=0.49$.
3. Normalize the path probabilities to obtain posteriors.

$$
P([ab]\mid ab)=\frac{40}{49}\approx0.8163265,\qquad
P([a,b]\mid ab)=\frac{9}{49}\approx0.1836735.
$$

The expected count of `ab` is $40/49$.
The expected count of `a` is $9/49$, and the expected count of `b` is also $9/49$.
These counts need not sum to one because segmentations contain different numbers of pieces.
Their sum is $58/49\approx1.1836735$, the expected number of pieces.
EM accumulates such fractional counts across examples before normalizing the vocabulary probabilities.

For SGNS, $\sigma(0)=1/2$.
The positive term contributes $-\log(1/2)=\log2$.
Each of the two negatives contributes another $\log2$.
The total loss is $3\log2\approx2.0794415$.
The negative terms use $\sigma(-s)$, but the sign does not change a zero score.

The GloVe residual is $0+0+0-\log4=-\log4$.
Its weighted squared contribution is $(\log4)^2\approx1.9218121$.
This is a regression term against a log co-occurrence count, not a normalized token probability.

Byte coverage guarantees a representation for every input byte after the chosen preprocessing.
It does not guarantee that training learns frequent multi-byte merges for every script.
An uncommon Unicode character can therefore occupy several byte tokens.
Coverage, reversibility, and token efficiency are separate properties.

</details>

### A3. Compute a two-token attention result · ●

**Subtopic: 4.2.**
Use $d_k=1$ and

$$
Q=\begin{bmatrix}0\\\log3\end{bmatrix},\quad
K=\begin{bmatrix}0\\1\end{bmatrix},\quad
V=\begin{bmatrix}2&0\\0&4\end{bmatrix}.
$$

Compute the full $2\times2$ score matrix, row-wise softmax, and output.
Repeat after applying a lower-triangular causal mask.
State the shape of every matrix and the axis over which the probabilities sum to one.

<details markdown="1">
<summary>Solution</summary>

1. $Q$ and $K$ have shape $(2,1)$, while $V$ has shape $(2,2)$.
2. $K^T$ has shape $(1,2)$, so $QK^T$ has shape $(2,2)$.
3. The scale $1/\sqrt{d_k}$ is one.

$$
S=\begin{bmatrix}0\cdot0&0\cdot1\\(\log3)\cdot0&(\log3)\cdot1\end{bmatrix}
=\begin{bmatrix}0&0\\0&\log3\end{bmatrix}.
$$

Exponentiating gives first-row weights proportional to $(1,1)$ and second-row weights proportional to $(1,3)$.
Divide each row by its own sum:

$$
A=\begin{bmatrix}1/2&1/2\\1/4&3/4\end{bmatrix}.
$$

The key axis is the last axis, so each query row sums to one.
Multiply by values:

$$
AV=\begin{bmatrix}
(1/2)2+(1/2)0&(1/2)0+(1/2)4\\
(1/4)2+(3/4)0&(1/4)0+(3/4)4
\end{bmatrix}
=\begin{bmatrix}1&2\\0.5&3\end{bmatrix}.
$$

The causal mask adds $-\infty$ only to the upper-right score.
The first softmax row becomes $(1,0)$, while the second row stays $(1/4,3/4)$.
The causal output is therefore

$$
O_{causal}=\begin{bmatrix}2&0\\0.5&3\end{bmatrix}.
$$

Both outputs have shape $(2,2)$: one row per query and two value features.

</details>

### A4. Derive scaling and trace MHA shapes · ●●

**Subtopic: 4.2.**
Assume independent query and key coordinates with zero mean and unit variance.
Assume products from different coordinates are uncorrelated.
Derive the variance of $q^Tk$ before and after scaling for $d_k=64$.
Then trace projected queries, head-major queries, scores, head outputs, concatenation, and final output for $B=2,n=5,d=12,h=3$.
Explain why increasing the number of heads at fixed $d$ does not increase the standard MHA weight count.

<details markdown="1">
<summary>Solution</summary>

1. For coordinate $r$, independence gives $E[q_rk_r]=E[q_r]E[k_r]=0$.
2. Its second moment is $E[q_r^2k_r^2]=E[q_r^2]E[k_r^2]=1$.
3. Its variance is therefore one.
4. Sum 64 uncorrelated products: $Var(q^Tk)=64$.
5. The unscaled standard deviation is eight.
6. Divide the score by $\sqrt{64}=8$.
7. Variance scales by the square of that factor: $Var(q^Tk/8)=64/64=1$.

The result describes an initialization-scale argument under stated assumptions.
Trained query and key coordinates need not retain unit variance or independence.
The scaling still supplies a consistent architecture convention.

Each head width is $d/h=4$.
The tensors are:

| Tensor | Shape |
|---|---|
| Input | $(2,5,12)$ |
| Fused projected $Q,K,V$ | Each $(2,5,12)$ |
| Split queries before transpose | $(2,5,3,4)$ |
| Head-major queries, keys, values | Each $(2,3,5,4)$ |
| Transposed keys | $(2,3,4,5)$ |
| Scores and probabilities | Each $(2,3,5,5)$ |
| Per-head weighted values | $(2,3,5,4)$ |
| Transpose before concatenation | $(2,5,3,4)$ |
| Concatenated head features | $(2,5,12)$ |
| Output projection | $(12,12)$ |
| Final output | $(2,5,12)$ |

The four dense projection matrices each contain $d^2$ weights.
Splitting those features into more heads does not add weights when the total width stays fixed.
Thus standard MHA has $4d^2$ weights, excluding biases.

</details>

### A5. Rotate a vector and prove relative position dependence · ●●

**Subtopic: 4.2.**
Use $R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}$.
Rotate $x=(2,1)^T$ by $\pi/2$ and verify its squared norm.
Prove the RoPE dot-product identity for query position $p$ and key position $t$ at frequency $\omega$.
Then compute $PE(0)$ for sinusoidal width four and explain the linear-shift property of its sine–cosine pairs.

<details markdown="1">
<summary>Solution</summary>

At $\pi/2$, cosine is zero and sine is one.
The rotation matrix is $\begin{bmatrix}0&-1\\1&0\end{bmatrix}$.
Multiplication gives

$$
R(\pi/2)\begin{bmatrix}2\\1\end{bmatrix}
=\begin{bmatrix}-1\\2\end{bmatrix}.
$$

The original squared norm is $2^2+1^2=5$.
The rotated squared norm is $(-1)^2+2^2=5$.
The negative sign in the first coordinate follows from the chosen counterclockwise rotation convention.

For arbitrary vectors, transpose the rotated query:

$$
(R(p\omega)q)^T(R(t\omega)k)
=q^TR(p\omega)^TR(t\omega)k.
$$

Orthogonality gives $R(a)^T=R(-a)$.
Composition gives $R(a)R(b)=R(a+b)$.
Therefore the score equals $q^TR((t-p)\omega)k$.
Its position-dependent rotation uses the relative offset $t-p$.
This identity does not remove the content dependence of $q$ or $k$.

For sinusoidal width four, positions use the pairs $(\sin p,\cos p)$ and $(\sin(0.01p),\cos(0.01p))$.
At zero the vector is $(0,1,0,1)$.
For one frequency, angle-addition identities give

$$
\begin{bmatrix}\sin((p+\Delta)\omega)\\\cos((p+\Delta)\omega)\end{bmatrix}
=\begin{bmatrix}\cos(\Delta\omega)&\sin(\Delta\omega)\\-\sin(\Delta\omega)&\cos(\Delta\omega)\end{bmatrix}
\begin{bmatrix}\sin(p\omega)\\\cos(p\omega)\end{bmatrix}.
$$

For fixed $\Delta$, the same matrix shifts every position pair.
Unlike RoPE, ordinary additive sinusoidal encoding does not force learned content projections to preserve a purely relative score form.

</details>

### A6. Count the KV cache and compare position extensions · ●●●

**Subtopic: 4.2.**
A decoder has 24 layers, one sequence, a 4,096-token prefix, 16 query heads, and head width 64.
Keys and values each use two-byte elements.
Compute raw cache storage for MHA, four-KV-head GQA, and MQA, in bytes and MiB.
State the one-token score shape for GQA.
Then distinguish NTK-aware RoPE scaling, YaRN, and ALiBi without calling ALiBi a rotary method.

<details markdown="1">
<summary>Solution</summary>

1. Count keys and values separately with a factor of two.
2. Multiply by layers, batch, sequence length, KV heads, and head width.
3. Multiply scalar count by two bytes and divide by $2^{20}$ for MiB.

$$
bytes=2\cdot24\cdot1\cdot4096\cdot g\cdot64\cdot2.
$$

For MHA, $g=16$:

$$
bytes=402{,}653{,}184,\qquad MiB=402{,}653{,}184/1{,}048{,}576=384.
$$

For GQA, $g=4$, so divide the MHA result by four.
The result is 100,663,296 bytes, or 96 MiB.
For MQA, $g=1$, so divide MHA by sixteen.
The result is 25,165,824 bytes, or 24 MiB.
These figures omit allocator overhead, metadata, and quantization scales.

GQA stores each key and value tensor as $(1,4,4096,64)$ at one layer.
The new query has shape $(1,16,1,64)$.
There are still 16 attention distributions, so scores have shape $(1,16,1,4096)$.
KV sharing does not reduce the query-head count.

NTK-aware scaling changes RoPE frequencies, commonly by modifying the base to compress slow frequencies more than fast ones.
YaRN combines a frequency-dependent extension scheme with attention scaling and a training recipe.
ALiBi instead adds fixed head-specific linear distance penalties directly to logits.
ALiBi does not rotate feature pairs.
All three need evaluation at the proposed deployment length; evaluating a position formula beyond training length is not evidence of useful extrapolation.

</details>

### A7. Diagnose architecture and target leakage · ●●

**Subtopic: 4.2.**
Choose an attention pattern and learning objective for each case: bidirectional token tagging, open-ended next-token generation, and reconstructing deleted source spans as output text.
For causal training on `[bos, A, B, eos]`, list input/target pairs.
Explain why the diagonal of the causal mask stays visible.
Explain why masking padded keys alone does not remove padded targets from the loss.
Describe where T5's decoder gets its keys and values in cross-attention.

<details markdown="1">
<summary>Solution</summary>

Bidirectional token tagging fits an encoder-only architecture.
The encoder can use left and right context for each token label.
MLM can pretrain its representations, then a token-classification head can learn labels.
The downstream label loss and the pretraining MLM loss are different objectives.

Open-ended generation fits a decoder-only causal language model.
The model predicts the next token from its prefix and never observes future inputs through self-attention.
Span reconstruction fits T5-style encoder-decoder pretraining.
The encoder reads the corrupted source bidirectionally, and the decoder emits the missing spans with sentinel alignment.

For the given sequence, the aligned training pairs are:

| Position | Visible input token | Target token |
|---|---|---|
| 0 | bos | A |
| 1 | A | B |
| 2 | B | eos |

At position 1, the representation of `A` predicts `B`.
It can see `bos` and `A`, including its diagonal, but not the input `B` at position 2.
The target shift and the triangular mask work together to prevent leakage.
Using unshifted labels lets the model learn to copy its current input instead of predicting the next token.

A key-padding mask prevents valid queries from retrieving padded source positions.
It does not remove the loss term assigned to a padded query's target.
The training objective must separately ignore padded labels or apply a loss mask.
If pad and EOS share an ID, derive the label mask from valid positions rather than blindly discarding every occurrence of that ID.

T5's decoder forms cross-attention queries from its decoder hidden states.
It forms cross-attention keys and values from encoder outputs.
All valid source positions are available to each target query; the causal restriction applies to target self-attention, not to the already observed source.

</details>

### A8. Count parameters and explain FlashAttention · ●●●

**Subtopic: 4.2.**
Use $d=128,h=8,d_{head}=16,f=512$.
Count MHA weights, MHA biases, an ordinary two-matrix FFN with biases, and two affine LayerNorms.
Exclude embeddings, final normalization, output vocabulary head, and position parameters.
Then compute the size of a two-byte score tensor with 16 heads and sequence length 4,096 for one batch item.
Explain which part FlashAttention avoids storing and which asymptotic arithmetic cost remains.

<details markdown="1">
<summary>Solution</summary>

The head relation is $8\cdot16=128$, so each fused projection is square.
Query, key, value, and output weights contribute

$$
4\cdot128\cdot128=65{,}536.
$$

Each of four projections has a width-128 bias, adding 512.
MHA therefore contains 66,048 parameters.
The first FFN matrix has $128\cdot512=65{,}536$ weights.
The second has $512\cdot128=65{,}536$ weights.
The two FFN biases add $512+128=640$.
The FFN total is 131,712.

Each affine LayerNorm has a scale and bias of width 128.
Two such layers contribute $2\cdot2\cdot128=512$ parameters.
The block total is

$$
66{,}048+131{,}712+512=198{,}272.
$$

The score tensor has $16\cdot4096^2=268{,}435{,}456$ elements.
At two bytes per element it uses 536,870,912 bytes, or 512 MiB.
This is the size of one score tensor, not all training activations.

FlashAttention tiles exact attention and maintains an online softmax accumulator.
It avoids materializing the full quadratic score/probability matrix in GPU high-bandwidth memory.
It still computes dense query–key interactions and weighted values.
The arithmetic remains $O(n^2d)$, while memory traffic and intermediate-storage requirements improve.
It is not sparse attention or a linear-time approximation.

</details>

## Part B — Code

### B1. Train and audit a byte-covering tokenizer · ●●

**Subtopic: 4.1.**
Train BPE from the inline corpus below with all byte symbols and `<unk>`, `<bos>`, and `<eos>` reserved.
Use a vocabulary budget of 300.
Do not insert BOS or EOS automatically.
Round-trip a string that contains an emoji and a Devanagari word absent from training.
Report whether decoding matches, whether any unknown ID occurs, and whether the special IDs are distinct.
Do not hard-code the learned merge IDs.

<details markdown="1">
<summary>Solution</summary>

Seed the complete byte alphabet before training.
Use the ByteLevel pre-tokenizer and matching decoder, with no lossy normalizer.
The model can use byte pieces for unseen characters.

```python
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

corpus = ["low low lower", "models read text", "bytes cover scripts"]
tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=300,
    min_frequency=1,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<unk>", "<bos>", "<eos>"],
    show_progress=False,
)
tok.train_from_iterator(corpus, trainer=trainer)
text = "low 🧪 नमस्ते"
ids = tok.encode(text).ids  # Variable-length list of integer token IDs
round_trip = tok.decode(ids) == text
unknown = tok.token_to_id("<unk>") in ids
special_ids = [tok.token_to_id(s) for s in ("<unk>", "<bos>", "<eos>")]
distinct = all(i is not None for i in special_ids) and len(set(special_ids)) == 3
assert round_trip and not unknown and distinct
assert tok.token_to_id("<bos>") not in ids
assert tok.token_to_id("<eos>") not in ids
print("round-trip:", round_trip)
print("unknown:", unknown)
print("distinct special IDs:", distinct)
```

Expected output:

```text
round-trip: True
unknown: False
distinct special IDs: True
```

The exact token list can vary with merge tie ordering.
The observable contract does not depend on that ordering.
Adding special tokens to a trainer reserves IDs but does not by itself define a post-processing template.
That is why this example does not automatically prepend BOS or append EOS.

</details>

### B2. Implement causal attention and check future isolation · ●●

**Subtopic: 4.2.**
Implement stable scaled dot-product attention in NumPy for two-dimensional query, key, and value arrays.
Use an allowed-position Boolean mask with a lower-triangular causal pattern.
Reproduce A3's causal output.
Change only the last value vector, then show that the first output remains unchanged and the second output changes.
Reject any mask row that contains no allowed key.

<details markdown="1">
<summary>Solution</summary>

The first query sees only the first key, so it cannot retrieve the changed second value.
The second query sees both keys and assigns positive weight to the second value.
Mask before softmax and normalize along the key axis.

```python
import numpy as np

def attend(q, k, v, allowed):
    # q: (n, d_k), k: (m, d_k), v: (m, d_v), allowed: (n, m)
    assert q.ndim == k.ndim == v.ndim == 2
    assert q.shape[1] == k.shape[1] and k.shape[0] == v.shape[0]
    assert allowed.dtype == np.bool_
    assert allowed.shape == (q.shape[0], k.shape[0])
    if not allowed.any(axis=-1).all():
        raise ValueError("Each query needs at least one allowed key")
    scores = q @ k.T / np.sqrt(q.shape[1])
    scores = np.where(allowed, scores, -np.inf)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities @ v, probabilities

q = np.array([[0.0], [np.log(3.0)]])
k = np.array([[0.0], [1.0]])
v = np.array([[2.0, 0.0], [0.0, 4.0]])
allowed = np.tril(np.ones((2, 2), dtype=bool))
output, probabilities = attend(q, k, v, allowed)
np.testing.assert_allclose(output, [[2.0, 0.0], [0.5, 3.0]])
np.testing.assert_allclose(probabilities.sum(-1), np.ones(2))
changed_v = v.copy()
changed_v[-1] = [100.0, 100.0]
changed_output, _ = attend(q, k, changed_v, allowed)
np.testing.assert_allclose(output[0], changed_output[0])
assert not np.allclose(output[1], changed_output[1])
rejected = False
try:
    attend(q, k, v, np.zeros((2, 2), dtype=bool))
except ValueError:
    rejected = True
assert rejected
print(output)
print("earlier output unchanged:", np.allclose(output[0], changed_output[0]))
print("all-masked rows rejected:", rejected)
```

Expected output:

```text
[[2.  0. ]
 [0.5 3. ]]
earlier output unchanged: True
all-masked rows rejected: True
```

The future-isolation check changes values while holding scores fixed.
A complete causal transformer also prevents future inputs from affecting the earlier queries and keys in every layer.
That stronger guarantee requires causal masking throughout the stack, not only at the final layer.

</details>

### B3. Check the RoPE relative-position identity · ●●●

**Subtopic: 4.2.**
Implement adjacent-pair rotary transforms on a tensor whose last dimension is four.
Use float64 to isolate mathematical identity from low-precision rounding.
Verify the dot-product identity for $p=2$, $t=5$, and frequencies $(1,0.01)$.
Also rotate the vector $(2,1)$ by a right angle and verify norm preservation.
Print Boolean identity checks rather than unstable rounded residuals.

<details markdown="1">
<summary>Solution</summary>

Broadcast a position scalar against the two frequency pairs.
Rotate each pair with the same sign convention as A5.
The two sides should agree within float64 numerical tolerance.

```python
import torch

def rotate_pairs(x, position, frequencies):
    # x: (..., d), frequencies: (d/2,), position: scalar
    assert x.shape[-1] == 2 * frequencies.numel()
    angles = position * frequencies  # (d/2,)
    even, odd = x[..., 0::2], x[..., 1::2]  # (..., d/2)
    c, s = angles.cos(), angles.sin()
    pairs = torch.stack((even * c - odd * s, even * s + odd * c), dim=-1)
    return pairs.flatten(-2)  # (..., d)

q = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float64)
k = torch.tensor([0.5, -1.0, 2.0, 1.0], dtype=torch.float64)
freq = torch.tensor([1.0, 0.01], dtype=torch.float64)
p, t = 2.0, 5.0
left = rotate_pairs(q, p, freq) @ rotate_pairs(k, t, freq)
right = q @ rotate_pairs(k, t - p, freq)
relative = torch.allclose(left, right, atol=1e-12, rtol=1e-12)
norms = torch.allclose(rotate_pairs(q, p, freq).norm(), q.norm(), atol=1e-12)
x = torch.tensor([2.0, 1.0], dtype=torch.float64)
right_angle = rotate_pairs(x, torch.pi / 2, torch.ones(1, dtype=torch.float64))
expected = torch.tensor([-1.0, 2.0], dtype=torch.float64)
angle_ok = torch.allclose(right_angle, expected, atol=1e-12)
assert relative and norms and angle_ok
print("relative-position identity:", relative)
print("norm preserved:", norms)
print("right-angle result:", angle_ok)
```

Expected output:

```text
relative-position identity: True
norm preserved: True
right-angle result: True
```

The frequency tensor has two entries because a four-feature vector contains two rotation pairs.
Queries and keys need the same frequency schedule for this identity.
Applying the transform only to queries does not produce the same relative-position property.
Changing the pairing layout without changing the checkpoint's projections also changes the model.

</details>

### B4. Audit parameters against actual PyTorch modules · ●●

**Subtopic: 4.2.**
Build the ordinary pre-normalization block parameter set from four linear attention projections, two FFN projections, and two affine LayerNorms.
Use $d=128,h=8,d_{head}=16,f=512$ and include every listed bias.
Count actual trainable parameters by module group.
Compare them with A8 and compute the GQA weight-only count for four KV heads at the same widths.
Do not include embeddings, dropout state, or optimizer state.

<details markdown="1">
<summary>Solution</summary>

PyTorch `Linear(in_features, out_features)` stores its weight as `(out_features, in_features)`.
That orientation differs from the row-vector notation in the notes but has the same number of parameters.
Each LayerNorm contributes one scale and one bias.
The exercise audits parameter allocation; it does not claim to implement a forward pass.

```python
import torch.nn as nn

d, h, d_head, f = 128, 8, 16, 512
assert h * d_head == d
attention = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(4)])
ffn = nn.ModuleList([nn.Linear(d, f, bias=True), nn.Linear(f, d, bias=True)])
norms = nn.ModuleList([nn.LayerNorm(d), nn.LayerNorm(d)])

def count(module):
    return sum(parameter.numel() for parameter in module.parameters())

counts = {"attention": count(attention), "ffn": count(ffn), "norms": count(norms)}
counts["total"] = sum(counts.values())
expected = {"attention": 66048, "ffn": 131712, "norms": 512, "total": 198272}
assert counts == expected
g = 4
gqa_weights = 2 * d * d + 2 * d * g * d_head
gqa = nn.ModuleList([
    nn.Linear(d, d, bias=False),
    nn.Linear(d, g * d_head, bias=False),
    nn.Linear(d, g * d_head, bias=False),
    nn.Linear(d, d, bias=False),
])
assert count(gqa) == gqa_weights == 49152
print(counts)
print("GQA weights only:", gqa_weights)
```

Expected output:

```text
{'attention': 66048, 'ffn': 131712, 'norms': 512, 'total': 198272}
GQA weights only: 49152
```

The GQA query and output matrices each have 16,384 weights.
The key and value matrices each have $128\cdot64=8,192$ weights.
Their total is $32,768+16,384=49,152$.
The MHA figure includes biases, while the GQA figure explicitly excludes them.
Keep this convention visible whenever you compare counts.

</details>

## Review your work

Recompute A1, A3, A5, and A8 on a blank page the next day.
Check that each arithmetic result follows from its stated dimensions and conventions.
Run all four code solutions and compare their observable outputs.
Use [the exit criteria](index.md) to decide whether to continue to Branch 5 or revisit a derivation.
If a code result differs, inspect its inputs and masks before changing the expected output.
