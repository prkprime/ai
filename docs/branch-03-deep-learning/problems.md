# Branch 3 problems — Derivatives, stable training, and image representations

Attempt each problem before you open its solution.
Use [the notes](notes.md) only to resolve a definition you cannot reconstruct.
Difficulty uses ● for direct application, ●● for a short derivation, and ●●● for integration.
The set contains six pen-and-paper problems and four coding problems.
For Part A, award one point per completed numbered task within a problem and compute your percentage across all tasks.
Pass at 70% or higher and complete every code exercise.

## Part A — Pen and paper

### A1. Backpropagation through two hidden neurons — ●●

Use column vectors and the squared loss $L=(\hat y-y)^2/2$.
The network is $z=W_1x+b_1$, $a=\operatorname{ReLU}(z)$, and $\hat y=W_2a+b_2$.
Set:

$$
x=(1,2)^T,\quad
W_1=\begin{bmatrix}1&-1\\1&1\end{bmatrix},\quad
b_1=(0,0)^T,\quad W_2=(2,-1),\quad b_2=0,\quad y=1.
$$

1. Compute $z$, $a$, $\hat y$, and $L$.
2. Compute $\partial L/\partial W_2$ and $\partial L/\partial b_2$.
3. Compute $\partial L/\partial a$ and $\partial L/\partial z$.
4. Compute $\partial L/\partial W_1$, $\partial L/\partial b_1$, and $\partial L/\partial x$.
5. Explain why the first hidden weight row receives no gradient on this sample.

<details markdown="1">
<summary>Solution</summary>

**Step 1: evaluate the graph.**

$$
z=\begin{bmatrix}1(1)-1(2)\\1(1)+1(2)\end{bmatrix}
=\begin{bmatrix}-1\\3\end{bmatrix},\qquad
a=\begin{bmatrix}0\\3\end{bmatrix}.
$$

The prediction is $2(0)-1(3)=-3$.
The residual is $-3-1=-4$.
The loss is $16/2=8$.

**Step 2: differentiate the output.**

The loss derivative with respect to prediction is $-4$.
Therefore:

$$
\frac{\partial L}{\partial W_2}=(-4)(0,3)=(0,-12),\qquad
\frac{\partial L}{\partial b_2}=-4.
$$

**Step 3: pass through the hidden activation.**

$$
\frac{\partial L}{\partial a}=W_2^T(-4)=(-8,4)^T.
$$

ReLU's mask at $z=(-1,3)^T$ is $(0,1)^T$.
The preactivation derivative is therefore $(0,4)^T$.

**Step 4: differentiate the first affine map.**

$$
\frac{\partial L}{\partial W_1}
=\begin{bmatrix}0\\4\end{bmatrix}(1,2)
=\begin{bmatrix}0&0\\4&8\end{bmatrix}.
$$

The bias derivative is $(0,4)^T$.
The input derivative is:

$$
W_1^T\begin{bmatrix}0\\4\end{bmatrix}
=\begin{bmatrix}1&1\\-1&1\end{bmatrix}
\begin{bmatrix}0\\4\end{bmatrix}
=\begin{bmatrix}4\\4\end{bmatrix}.
$$

**Step 5: interpret the zero row.**

The first neuron has negative preactivation.
Its ReLU derivative is zero.
Its incoming activation derivative $-8$ does not pass through that gate.
This is a local statement about this sample, not proof that the neuron is inactive on every sample.

</details>

### A2. Derive initialization scales — ●●

Assume independent zero-mean weights with variance $s^2$.
Each input coordinate has second moment $q=1$.
Assume the preactivation distribution is symmetric.

1. Derive $\mathbb{E}[z^2]$ for fan-in $n$.
2. Derive the He fan-in variance for a ReLU layer.
3. Calculate weight variance and standard deviation at fan-in 128.
4. Calculate Xavier uniform's bound for fan-in 128 and fan-out 128.
5. Explain why the He calculation preserves a second moment rather than ReLU output variance exactly.
6. Explain why fully zero-initialized hidden weights do not break symmetry.

<details markdown="1">
<summary>Solution</summary>

**Step 1: expand the sum.**

Write $z=\sum_{i=1}^n w_ix_i$.
Zero-mean independent weights make cross terms vanish in the squared expectation.
Each diagonal term contributes $s^2q$.
Thus $\mathbb{E}[z^2]=ns^2q$.

**Step 2: include the ReLU gate.**

Symmetry places half the squared preactivation mass on either side of zero.
Therefore:

$$
\mathbb{E}[\operatorname{ReLU}(z)^2]=\tfrac12ns^2q.
$$

Set this equal to $q$ to obtain $s^2=2/n$.

**Step 3: substitute fan-in 128.**

The variance is $2/128=1/64=0.015625$.
The standard deviation is $\sqrt{1/64}=1/8=0.125$.
The preactivation second moment is $128(1/64)=2$.
After ReLU, the second moment is one.

**Step 4: derive the uniform bound.**

Xavier variance is $2/(128+128)=1/128$.
A uniform distribution on $[-a,a]$ has variance $a^2/3$.
Solve $a^2/3=1/128$ to get $a=\sqrt{3/128}\approx0.153093$.

**Step 5: separate variance from second moment.**

ReLU usually has positive mean.
Its variance equals its second moment minus its squared mean.
For Gaussian preactivation with variance two, the ReLU mean is $1/\sqrt{\pi}$.
Its output variance is $1-1/\pi\approx0.68169$, not one.
The second-moment argument remains the intended approximation for propagating signal scale.

**Step 6: explain symmetry.**

Identically initialized hidden neurons compute identical functions and receive identical derivatives.
Deterministic gradient updates preserve that equality.
Fully zero ReLU layers can also produce zero activations and zero incoming gradients.
Random weights break this symmetry; zero biases alone are not a problem.

</details>

### A3. Softmax Jacobian and classification gradient — ●●

Let $z=(\log2,0)^T$ and let the first class be correct, $t=(1,0)^T$.

1. Compute $p=\operatorname{softmax}(z)$.
2. Derive every entry of its $2\times2$ Jacobian.
3. Verify that the Jacobian kills the vector $(1,1)^T$.
4. Compute $\partial L/\partial z$ for cross-entropy and explain the sign of each entry.

<details markdown="1">
<summary>Solution</summary>

**Step 1: normalize exponentials.**

The exponentials are $(2,1)$ and their sum is three.
Thus $p=(2/3,1/3)^T$.

**Step 2: apply the quotient rule.**

For any indices, $J_{ij}=p_i(\delta_{ij}-p_j)$.
The first diagonal entry is $(2/3)(1/3)=2/9$.
The second diagonal entry is $(1/3)(2/3)=2/9$.
Each off-diagonal entry is $-(2/3)(1/3)=-2/9$.
Therefore:

$$
J=\frac19\begin{bmatrix}2&-2\\-2&2\end{bmatrix}.
$$

**Step 3: test the constant-shift direction.**

Multiplication by $(1,1)^T$ gives $(0,0)^T$.
Adding one shared infinitesimal shift to the logits has no effect on probabilities.
The finite version follows by cancelling a shared exponential factor.

**Step 4: differentiate the loss.**

For the first-class target, $L=-\log p_1$.
The probability gradient is $(-1/p_1,0)^T=(-3/2,0)^T$.
Multiply by $J^T$:

$$
\nabla_zL=(-1/3,1/3)^T=p-t.
$$

Gradient descent raises the correct logit because its derivative is negative.
It lowers the incorrect logit because its derivative is positive.

</details>

### A4. Receptive field with stride and dilation — ●●

Consider three $3\times3$ convolutions with dilation one.
Only the second layer uses stride two; the other strides are one.
Use enough padding to discuss an interior output location.

1. Derive the receptive-field and jump recurrences.
2. Compute the jump and receptive-field span after every layer.
3. Change the third layer's dilation to two and recompute its final span.
4. Explain why changing padding does not change this interior span formula.

<details markdown="1">
<summary>Solution</summary>

**Step 1: derive the recurrences.**

Let the previous jump be $j$ and receptive-field span be $r$.
A kernel samples $k$ positions separated by $d$ previous-layer locations.
Its first and last samples are $(k-1)d j$ input pixels apart.
The union spans $r+(k-1)d j$ pixels.
Output centers move by $s$ previous-layer positions, so the new jump is $js$.

**Step 2: propagate layer by layer.**

Start with $r_0=j_0=1$.

| Layer | Calculation for span | New span | New jump |
|---|---|---:|---:|
| 1 | $1+(3-1)(1)(1)$ | 3 | $1\times1=1$ |
| 2 | $3+(3-1)(1)(1)$ | 5 | $1\times2=2$ |
| 3 | $5+(3-1)(1)(2)$ | 9 | $2\times1=2$ |

Use the previous jump in the span calculation.
Using the new jump at layer two would incorrectly enlarge that layer's field.

**Step 3: change the final dilation.**

The final span becomes $5+(3-1)(2)(2)=13$.
The final jump remains two because dilation does not change output-center stride.

**Step 4: distinguish padding from kernel spacing.**

Padding shifts center alignment and affects the real pixels available at boundaries.
It does not change the spacing between kernel samples at an interior location.
The span recurrence describes theoretical extent, not the number of nonzero contributions at a padded edge.

</details>

### A5. One coordinate of Adam, AdamW, and Lion — ●●

Use $\theta_0=1$, $g_1=2$, zero optimizer state, and $\eta=0.001$.
Ignore $\epsilon$ only in this hand calculation.
For Adam, use $\beta_1=0.9$ and $\beta_2=0.999$.
For AdamW, also use $\lambda=0.1$.
For Lion, use $\beta_1=0.9$, $\beta_2=0.99$, and the same decay.

1. Compute Adam's moments, bias-corrected moments, and parameter value.
2. Compute AdamW's parameter value and separate the two update contributions.
3. Compute Lion's direction blend, parameter value, and new momentum state.
4. For $g=(3,4)$, compare global norm clipping at one with value clipping at one.

<details markdown="1">
<summary>Solution</summary>

**Step 1: compute Adam.**

The first moment is $m_1=0.9(0)+0.1(2)=0.2$.
The second moment is $v_1=0.999(0)+0.001(4)=0.004$.
Bias correction gives $\hat m_1=0.2/0.1=2$ and $\hat v_1=0.004/0.001=4$.
The adaptive direction is $2/\sqrt4=1$.
Therefore $\theta_1=1-0.001=0.999$.

**Step 2: decouple decay.**

The adaptive contribution remains $-0.001$.
The decay contribution is $-\eta\lambda\theta_0=-0.0001$.
Therefore AdamW gives $\theta_1=0.9989$.
Adding $\lambda\theta_0$ to the gradient would instead change the moments and would not reproduce decoupled decay generally.

**Step 3: compute Lion.**

The direction blend uses the old state:
$c_1=0.9(0)+0.1(2)=0.2$.
Its sign is positive one.
The new parameter is $(1-0.001\times0.1)(1)-0.001=0.9989$.
Only after forming the direction do you update the stored momentum:
$m_1=0.99(0)+0.01(2)=0.02$.
Matching first-step parameter values does not make Lion and AdamW equivalent across later steps.

**Step 4: compare clipping rules.**

The gradient norm is five.
Global norm clipping multiplies both entries by $1/5$, yielding $(0.6,0.8)$ with norm one.
Value clipping independently clamps them, yielding $(1,1)$ with norm $\sqrt2$.
The first operation preserves direction and the second changes it.

</details>

### A6. Convolution dimensions, parameters, and FLOPs — ●●

An input has shape `(1, 3, 32, 32)`.
A convolution has 16 output channels, kernel size three, stride one, padding one, dilation one, groups one, and a bias.

1. Derive the output height and width.
2. Count weights, biases, and total parameters separately.
3. Count MACs for one image and estimate FLOPs using two FLOPs per MAC, excluding bias.
4. Change only stride to two and recompute output shape and MACs.
5. Explain which counts change if dilation becomes two with padding two and stride one.

<details markdown="1">
<summary>Solution</summary>

**Step 1: compute the output dimensions.**

$$
H_{\mathrm{out}}=\left\lfloor\frac{32+2(1)-1(3-1)-1}{1}\right\rfloor+1
=32.
$$

Width follows the same calculation.
The output shape is `(1, 16, 32, 32)`.

**Step 2: count parameters.**

Each output channel has $3\times3\times3=27$ kernel weights.
There are $16\times27=432$ weights and 16 biases.
The total is 448 parameters.

**Step 3: count direct operations.**

There are $16\times32\times32=16{,}384$ output scalars.
Each requires 27 MACs.
The total is $16{,}384\times27=442{,}368$ MACs.
At two FLOPs per MAC, the estimate is $884{,}736$ FLOPs.
The convention excludes bias additions; adding one bias per output would add 16,384 scalar additions.

**Step 4: change stride.**

At stride two, height is $\lfloor31/2\rfloor+1=16$.
The output is `(1, 16, 16, 16)`.
MACs become $16\times16\times16\times27=110{,}592$.
Parameters stay 448 because spatial reuse does not change the kernel tensors.

**Step 5: change dilation with matching padding.**

With dilation two and padding two, the effective kernel span is five.
Height is $\lfloor(32+4-4-1)/1\rfloor+1=32$.
The learned kernel still contains nine spatial weights per channel pair.
Thus parameters and direct MACs stay the same as the original layer.
The receptive field grows, while sample locations within it become more spread out.

</details>

## Part B — Code

Run each solution as a separate script with Python and PyTorch.
For a standalone environment, use `uv run --with torch python script.py`.
No exercise downloads a dataset or model checkpoint.
Expected outputs describe deterministic values or explicit shape and finiteness checks.
Do not compare random floating-point losses bit for bit across hardware.

### B1. Match manual MLP derivatives to autograd — ●●

Implement the A1 network with tensors.
Compute manual gradients from the saved forward values without calling autograd for those formulas.
Compare every parameter gradient and the input gradient with autograd.
Then take a small gradient-descent step and report whether the loss decreases.
Use float64 and learning rate 0.001.

<details markdown="1">
<summary>Solution</summary>

The manual calculation uses the same forward values but applies the chain rule explicitly.
The comparison checks the complete gradient tensors, not only the loss.
The parameter update runs under `no_grad` so it does not become part of another graph.

```python
import torch

dtype = torch.float64
x = torch.tensor([1.0, 2.0], dtype=dtype, requires_grad=True)
w1 = torch.tensor([[1.0, -1.0], [1.0, 1.0]], dtype=dtype, requires_grad=True)
b1 = torch.zeros(2, dtype=dtype, requires_grad=True)
w2 = torch.tensor([[2.0, -1.0]], dtype=dtype, requires_grad=True)
b2 = torch.zeros(1, dtype=dtype, requires_grad=True)
z = w1 @ x + b1
a = torch.relu(z)
prediction = w2 @ a + b2
loss = 0.5 * (prediction - 1).square().sum()

with torch.no_grad():
    dy = prediction - 1
    da = w2.T @ dy
    dz = da * (z > 0)
    manual = [
        dz[:, None] @ x[None, :],
        dz.clone(),
        dy[:, None] @ a[None, :],
        dy.clone(),
        w1.T @ dz,
    ]

loss.backward()
tensors = [w1, b1, w2, b2, x]
matches = [torch.allclose(t.grad, g) for t, g in zip(tensors, manual)]
before = loss.item()
with torch.no_grad():
    for parameter in (w1, b1, w2, b2):
        parameter -= 0.001 * parameter.grad
    after = 0.5 * ((w2 @ torch.relu(w1 @ x + b1) + b2) - 1).square().sum()
print("gradient matches:", matches)
print("initial loss:", before)
print("loss decreased:", after.item() < before)
```

Expected output:

```text
gradient matches: [True, True, True, True, True]
initial loss: 8.0
loss decreased: True
```

The input is not a parameter in the update loop.
Its derivative still exists and is checked.
A small step decreases this example's loss, but a positive learning rate does not guarantee descent for every nonlinear model and every step size.

</details>

### B2. Diagnose FP16 overflow and clip unscaled gradients — ●●

Show that squaring 300 overflows in FP16 but not in FP32.
Compute an ordinary gradient for a scalar quadratic loss, simulate a fixed loss scale, unscale the gradient, and clip its norm.
Compare the update with a correctly ordered CUDA AMP step from [the notes](notes.md#runnable-snippet-6-amp-training-step-with-a-cpu-fallback).
Keep this exercise runnable on CPU.

<details markdown="1">
<summary>Solution</summary>

The fixed-scale example demonstrates the algebra of unscaling.
It does not implement dynamic overflow detection or replace `GradScaler`.
The parameter stays FP32 while the loss is scaled.

```python
import torch

bad = torch.tensor(300.0, dtype=torch.float16).square()
good = torch.tensor(300.0, dtype=torch.float32).square()
print("FP16 finite:", torch.isfinite(bad).item())
print("FP32 square:", good.item())

parameter = torch.nn.Parameter(torch.tensor(3.0))
optimizer = torch.optim.SGD([parameter], lr=0.1)
optimizer.zero_grad(set_to_none=True)
scale = 128.0
loss = 0.5 * parameter.square()
(scale * loss).backward()
print("scaled gradient:", parameter.grad.item())
with torch.no_grad():
    parameter.grad.div_(scale)
print("unscaled gradient:", parameter.grad.item())
preclip = torch.nn.utils.clip_grad_norm_([parameter], max_norm=1.0)
optimizer.step()
print("preclip norm:", preclip.item())
print("updated parameter:", round(parameter.item(), 4))
```

Expected output:

```text
FP16 finite: False
FP32 square: 90000.0
scaled gradient: 384.0
unscaled gradient: 3.0
preclip norm: 3.0
updated parameter: 2.9
```

The unclipped gradient is three because the derivative of $\theta^2/2$ is $\theta$.
Scaling multiplies it by 128, giving 384.
Unscaling restores three, and norm clipping reduces its magnitude to approximately one.
The SGD update subtracts approximately 0.1.
PyTorch includes a small denominator safeguard in clipping, so the printed result rounds to 2.9.
If you clipped 384 before unscaling, the final gradient would be about $1/128$ instead of one.
Dynamic AMP also checks nonfinite gradients and can skip the step; this fixed-scale demonstration assumes finite gradients.

</details>

### B3. Compare a convolutional projection with a tiny ViT — ●●●

Build a tiny image classifier for inputs `(2, 3, 32, 32)`.
Use patch side eight, hidden width 48, one transformer encoder layer, four attention heads, and three output classes.
Prepend a learned class token and add learned position embeddings.
Return logits, compute cross-entropy, and run backward once.
Also count the parameters in the patch projection and the standard convolution from A6.
This is a shape and gradient lab, not an accuracy benchmark.

<details markdown="1">
<summary>Solution</summary>

The input has 16 patches, so the sequence has 17 positions after adding the class token.
Four heads divide hidden width 48 into head dimensions of 12.
The code uses `batch_first=True` to keep `(batch, sequence, hidden)` order.
It uses pre-norm and disables nested-tensor conversion because this example uses dense fixed-length sequences.

```python
import torch
from torch import nn

class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 48, kernel_size=8, stride=8)
        self.cls = nn.Parameter(torch.zeros(1, 1, 48))
        self.position = nn.Parameter(torch.zeros(1, 17, 48))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=48,
            nhead=4,
            dim_feedforward=96,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(48)
        self.head = nn.Linear(48, 3)

    def forward(self, image):
        if image.shape[1:] != (3, 32, 32):
            raise ValueError("Expected images with shape (B, 3, 32, 32)")
        tokens = self.patch(image).flatten(2).transpose(1, 2)
        cls = self.cls.expand(image.shape[0], -1, -1)
        sequence = torch.cat([cls, tokens], dim=1) + self.position
        encoded = self.encoder(sequence)
        return self.head(self.norm(encoded[:, 0]))

torch.manual_seed(0)
model = TinyViT()
images = torch.randn(2, 3, 32, 32)
labels = torch.tensor([0, 2])
logits = model(images)
loss = nn.functional.cross_entropy(logits, labels)
loss.backward()
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
print("logits:", tuple(logits.shape))
print("patch parameters:", sum(p.numel() for p in model.patch.parameters()))
print("A6 convolution parameters:", sum(p.numel() for p in conv.parameters()))
print("finite loss:", torch.isfinite(loss).item())
print("patch gradient finite:", torch.isfinite(model.patch.weight.grad).all().item())
```

Expected output:

```text
logits: (2, 3)
patch parameters: 9264
A6 convolution parameters: 448
finite loss: True
patch gradient finite: True
```

The patch parameter count is $48(3\times8\times8)+48=9{,}264$.
It does not include the encoder, class token, position table, final norm, or head.
The larger patch-projection count does not establish which full architecture is more expensive.
Compare complete models at the same input resolution and task.
For a real training experiment, add a dataset split and a training loop rather than interpreting this single random batch as evidence of learning.

</details>

### B4. Compute CLIP loss and zero-shot predictions — ●●

Use the two basis vectors in two dimensions as image embeddings and matching text embeddings.
Normalize both sets, set temperature to one, and compute both directions of contrastive cross-entropy.
Then classify two query image embeddings against the same text class prototypes.
Use queries `(0.9, 0.1)` and `(0.2, 0.8)` and class names `horizontal` and `vertical`.
Explain what a real pretrained CLIP pipeline adds.

<details markdown="1">
<summary>Solution</summary>

The loss target is the diagonal pairing, represented by integer labels `[0, 1]`.
The transpose reverses retrieval direction.
The query classifications use the same normalized cosine geometry as the training objective.

```python
import torch
import torch.nn.functional as F

image = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
text = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
temperature = 1.0
scores = image @ text.T / temperature
targets = torch.arange(2)
image_to_text = F.cross_entropy(scores, targets)
text_to_image = F.cross_entropy(scores.T, targets)
loss = 0.5 * (image_to_text + text_to_image)

queries = F.normalize(torch.tensor([[0.9, 0.1], [0.2, 0.8]]), dim=-1)
class_scores = queries @ text.T / temperature
probabilities = class_scores.softmax(dim=-1)
names = ["horizontal", "vertical"]
predictions = [names[i] for i in class_scores.argmax(dim=-1).tolist()]
print("symmetric loss:", round(loss.item(), 6))
print("predictions:", predictions)
print("probability rows sum to one:", torch.allclose(probabilities.sum(-1), torch.ones(2)))
```

Expected output:

```text
symmetric loss: 0.313262
predictions: ['horizontal', 'vertical']
probability rows sum to one: True
```

Each positive score is one and each negative score is zero.
For either row, the positive probability is $e/(e+1)$.
Its negative log is $\log(1+e^{-1})\approx0.313262$.
The row and column losses match, so averaging them leaves the value unchanged.
For real images, a pretrained checkpoint provides image preprocessing, a text tokenizer, image and text encoders, and a learned score scale.
You encode prompts such as “a photo of a horizontal object” only if those prompts describe the actual target classes appropriately.
The toy basis vectors stand in for those final embeddings and do not demonstrate learned semantic recognition.

</details>

## Reflection

Record one failed calculation and the assumption that caused it.
Label the issue as a shape error, a chain-rule error, a reduction error, a dtype error, or a model-mode error.
Repeat that calculation without the solution the next day.
Use [the exit criteria](index.md#exit-criteria) before you move to Branch 4.
