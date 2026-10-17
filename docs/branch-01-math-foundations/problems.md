# Branch 1 problems

Use the [notes](notes.md) and [schedule](index.md#session-schedule) to place these exercises within the eight-hour branch.
Attempt each problem before opening its solution.
Difficulty markers mean ● routine, ●● multi-step, and ●●● synthesis.
Use natural logarithms unless a problem states otherwise.
Record shapes and intermediate values so you can diagnose an incorrect result.

## Part A — Pen and paper

### A1. Derive and check an orthogonal projection ●●

Let $a=(1,2)^\top$ and $b=(3,1)^\top$.
Derive the coefficient that minimizes $\|b-ca\|_2^2$.
Compute the projection matrix, projected vector, residual, and squared residual norm.
Show that the projector is symmetric and idempotent, meaning $P^\top=P$ and $P^2=P$.

<details markdown="1">
<summary>Solution</summary>

**1.** Expand the objective:

$$
f(c)=b^\top b-2c\,a^\top b+c^2a^\top a.
$$

**2.** Differentiate and solve:

$$
f'(c)=-2a^\top b+2c\,a^\top a=0,
\qquad c=\frac{a^\top b}{a^\top a}.
$$

Here $a^\top b=3+2=5$ and $a^\top a=1+4=5$, so $c=1$.

**3.** Construct the matrix:

$$
P=\frac{aa^\top}{a^\top a}
=\frac15\begin{bmatrix}1&2\\2&4\end{bmatrix}.
$$

**4.** Multiply by $b$ to get $Pb=(1,2)^\top$.
The residual is $r=b-Pb=(2,-1)^\top$.
Its squared norm is $2^2+(-1)^2=5$.
Orthogonality holds because $a^\top r=2-2=0$.

**5.** Symmetry follows from $(aa^\top)^\top=aa^\top$.
For idempotence,

$$
P^2=\frac{aa^\top aa^\top}{(a^\top a)^2}
=\frac{a(a^\top a)a^\top}{(a^\top a)^2}=P.
$$

**6.** The second derivative is $f''(c)=2a^\top a=10>0$.
Thus the stationary point is the unique minimizing coefficient.

</details>

### A2. Compute an eigen-decomposition and SVD ●●

For $C=\begin{bmatrix}2&1\\1&2\end{bmatrix}$, compute both eigenvalues and normalized eigenvectors by hand.
Use them to construct an SVD.
Compute the best rank-one approximation and its squared Frobenius error.
Explain why the eigenvalues and singular values coincide for this matrix.

<details markdown="1">
<summary>Solution</summary>

**1.** Solve the characteristic equation:

$$
\det(C-\lambda I)=(2-\lambda)^2-1
=\lambda^2-4\lambda+3=(\lambda-3)(\lambda-1)=0.
$$

The eigenvalues are 3 and 1.

**2.** For $\lambda=3$, $(C-3I)v=0$ gives $v_1=v_2$.
Choose $q_1=(1,1)^\top/\sqrt2$.
For $\lambda=1$, $(C-I)v=0$ gives $v_1=-v_2$.
Choose $q_2=(1,-1)^\top/\sqrt2$.

**3.** The vectors have norm one and dot product zero.
Let $Q=[q_1\ q_2]$.
Then $Q^\top Q=I$ and $C=Q\operatorname{diag}(3,1)Q^\top$.

**4.** Set $U=Q$, $\Sigma=\operatorname{diag}(3,1)$, and $V=Q$.
This is an SVD because both factors are orthogonal and the diagonal entries are nonnegative and sorted.
It works because $C$ is symmetric positive definite.
For an indefinite symmetric matrix, signed eigenvalues would not directly form an SVD diagonal.

**5.** Keep the leading singular component:

$$
C_1=3q_1q_1^\top
=\frac32\begin{bmatrix}1&1\\1&1\end{bmatrix}.
$$

**6.** The residual is

$$
C-C_1=\begin{bmatrix}0.5&-0.5\\-0.5&0.5\end{bmatrix}.
$$

Its squared Frobenius norm is $4(0.5^2)=1$.
This equals the square of the omitted singular value.
The rank-one approximation is optimal under this norm.

</details>

### A3. Derive softmax-cross-entropy and a dense gradient ●●●

Let $X=[1,2]$ and $W=\begin{bmatrix}0&0\\0&0\end{bmatrix}$.
Compute logits $z=XW$, probabilities $p=\operatorname{softmax}(z)$, and loss $L=-\log p_1$ for target $t=(1,0)$.
Derive $\partial p_i/\partial z_j$, then derive $\nabla_zL$ and $\nabla_WL$.
State the gradient shape for $Q$ if an attention head uses $Q:(2,3)$, $K:(4,3)$, and $V:(4,5)$.

<details markdown="1">
<summary>Solution</summary>

**1.** Matrix multiplication gives $z=(0,0)$.
Both exponentials equal one, so $p=(1/2,1/2)$.
The loss is $-\log(1/2)=\log2\approx0.693147$.

**2.** Write $Z=\sum_k e^{z_k}$ and apply the quotient rule:

$$
\frac{\partial p_i}{\partial z_j}
=\frac{\delta_{ij}e^{z_i}Z-e^{z_i}e^{z_j}}{Z^2}
=p_i(\delta_{ij}-p_j).
$$

**3.** Use $L=-\sum_i t_i\log p_i$ and the chain rule:

$$
\frac{\partial L}{\partial z_j}
=-\sum_i\frac{t_i}{p_i}p_i(\delta_{ij}-p_j)
=-t_j+p_j\sum_i t_i=p_j-t_j.
$$

Thus the row-shaped upstream gradient is $G_z=[-1/2,1/2]$.
Its entries sum to zero, as required by invariance to a shared logit offset.

**4.** Expand $z_j=\sum_aX_aW_{aj}$.
Then $\partial L/\partial W_{aj}=X_a(G_z)_j$.
Therefore

$$
\nabla_WL=X^\top G_z
=\begin{bmatrix}-1/2&1/2\\-1&1\end{bmatrix}.
$$

The gradient shape is $(2,2)$, matching $W$.
Since $W=0$, the input gradient $G_zW^\top$ is zero at this particular point.

**5.** For attention, $S$ and $P$ have shape $(2,4)$ and $O$ has shape $(2,5)$.
An upstream $G_O:(2,5)$ gives $G_P=G_OV^\top:(2,4)$.
Row-softmax differentiation preserves shape $(2,4)$.
Thus $G_Q=G_SK/\sqrt3$ has shape $(2,3)$, matching $Q$.
Also $G_K:(4,3)$ and $G_V:(4,5)$.

</details>

### A4. Derive MLE and MAP estimates ●●

Observe Bernoulli data $(1,0,1,1,0)$.
Derive the MLE and the MAP estimate with a Beta$(2,2)$ prior.
For Gaussian data $(1,2,3)$, derive the mean and variance MLEs.
Then find the mean MAP assuming known observation variance 1 and prior $\mu\sim\mathcal N(0,1)$.
Explain why the variance MLE differs from the unbiased sample variance.

<details markdown="1">
<summary>Solution</summary>

**1.** The Bernoulli sample has $n=5$ and $s=3$ successes.
Its likelihood is $\theta^3(1-\theta)^2$.
The derivative of the log-likelihood is $3/\theta-2/(1-\theta)$.
Setting it to zero gives $3(1-\theta)=2\theta$, hence $\hat\theta=3/5=0.6$.
The second derivative $-3/\theta^2-2/(1-\theta)^2$ is negative in the interior.

**2.** The Beta$(2,2)$ prior contributes $\theta(1-\theta)$.
The posterior density is proportional to $\theta^4(1-\theta)^3$.
Its log derivative is $4/\theta-3/(1-\theta)$.
Set it to zero to get $4(1-\theta)=3\theta$, or $\hat\theta_{\mathrm{MAP}}=4/7$.
Both posterior shape parameters, 5 and 4, exceed one, so the interior-mode formula applies.
The posterior mean is $5/9$, not $4/7$.

**3.** For Gaussian data, write $v=\sigma^2$:

$$
\ell(\mu,v)=-\frac32\log(2\pi v)
-\frac{(1-\mu)^2+(2-\mu)^2+(3-\mu)^2}{2v}.
$$

Differentiation in $\mu$ yields $(6-3\mu)/v=0$, so $\hat\mu=2$.

**4.** At this mean, the squared residual sum is $1+0+1=2$.
Differentiation in $v$ yields $-3/(2v)+2/(2v^2)=0$.
Therefore $\hat v=2/3$.
The unbiased sample variance divides by $n-1=2$ and equals 1.
Estimating the mean consumes one degree of freedom; the MLE criterion does not correct for this bias.

**5.** With known variance 1 and prior variance 1, the log posterior in $\mu$ adds $-\mu^2/2$.
Its derivative is $(6-3\mu)-\mu=6-4\mu$.
Thus $\hat\mu_{\mathrm{MAP}}=1.5$.
The second derivative is $-4$, so this point is the unique maximum.
The prior pulls the estimate from 2 toward its prior mean 0.

</details>

### A5. Compute cross-entropy and KL divergence ●●

Let $p=(0.5,0.5)$ and $q=(0.75,0.25)$.
Compute $H(p)$, $H(p,q)$, $D_{\mathrm{KL}}(p\|q)$, and $D_{\mathrm{KL}}(q\|p)$ in nats.
Verify the entropy-plus-KL identity numerically.
State what changes if $q$ assigns zero probability to the second outcome.

<details markdown="1">
<summary>Solution</summary>

**1.** Entropy is

$$
H(p)=-2(0.5\log0.5)=\log2\approx0.693147.
$$

**2.** Cross-entropy is

$$
H(p,q)=-0.5\log0.75-0.5\log0.25
=0.143841+0.693147\approx0.836988.
$$

**3.** Forward KL is

$$
D_{\mathrm{KL}}(p\|q)
=0.5\log(2/3)+0.5\log2
=0.5\log(4/3)\approx0.143841.
$$

**4.** Summing entropy and forward KL gives $0.693147+0.143841=0.836988$ after rounding.
This agrees with cross-entropy.
Algebraically, split $\log(p_i/q_i)$ into $\log p_i-\log q_i$ and collect the two sums.

**5.** Reverse KL is

$$
D_{\mathrm{KL}}(q\|p)
=0.75\log1.5+0.25\log0.5
\approx0.304099-0.173287=0.130812.
$$

It differs from the forward KL, so KL is not a distance metric with symmetry.

**6.** If $q_2=0$ while $p_2=0.5$, the second cross-entropy term is infinite.
Forward KL is also infinite.
Silently clipping probabilities changes the evaluated model; report any numerical clipping policy rather than treating it as the exact quantity.

</details>

### A6. Relate perplexity, moments, and aggregation ●●

A model assigns conditional probabilities $(1/2,1/4,1/8)$ to three observed tokens.
Compute total NLL, mean NLL, and perplexity.
Suppose the first sequence has one token with NLL $\log2$ and the second has two tokens with NLLs $\log4$ and $\log8$.
Explain why averaging the two sequence perplexities gives the wrong corpus result.
Also compute the variance of a Bernoulli variable with success probability $1/4$.

<details markdown="1">
<summary>Solution</summary>

**1.** The total NLL is

$$
-\log(1/2)-\log(1/4)-\log(1/8)
=\log2+\log4+\log8=\log64\approx4.158883.
$$

**2.** Divide by three tokens to get mean NLL $\log4\approx1.386294$.
Exponentiation gives perplexity 4.
Equivalently, the product probability is $1/64$ and its inverse cube root is 4.

**3.** The first sequence has perplexity 2.
The second has mean NLL $(\log4+\log8)/2=\log\sqrt{32}$ and perplexity $\sqrt{32}\approx5.656854$.
Their arithmetic average is about $3.828427$, not 4.
Corpus perplexity first sums token NLLs, divides by total token count, and then exponentiates.
Equal weighting of sequence perplexities applies a different weighting and a nonlinear transformation in the wrong order.

**4.** For Bernoulli $X$, $X^2=X$ and $\mathbb EX=1/4$.
Thus

$$
\operatorname{Var}(X)=\mathbb E[X^2]-(\mathbb EX)^2
=\frac14-\frac1{16}=\frac3{16}=0.1875.
$$

</details>

## Part B — Code

Use the environment from the [notes](notes.md#working-environment).
All reference solutions below are complete scripts or explicitly named files.
Timings are observations, not pass thresholds.
Do not copy files into the syllabus repository when doing the lab; use your separate lab directory.

### B1. Check attention gradients numerically ●●●

Implement one unmasked attention head with $Q:(2,3)$, $K:(4,3)$, and $V:(4,2)$.
Use a scalar loss $L=\langle O,R\rangle_F$ for a fixed upstream array $R$.
Implement the analytic gradients of $Q$, $K$, and $V$ without autograd.
Compare each gradient element with a centered finite difference using float64 inputs.
Print gradient shapes and the largest absolute discrepancy.
Expected behavior: the shapes match all three inputs and the discrepancy is below `1e-7` for the supplied deterministic data.

<details markdown="1">
<summary>Solution</summary>

Save as `attention_check.py` and run `uv run python attention_check.py`.

```python
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def attention(q: Array, k: Array, v: Array) -> tuple[Array, Array]:
    scores = q @ k.T / np.sqrt(q.shape[1])
    shifted = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    p = weights / weights.sum(axis=1, keepdims=True)
    return p @ v, p


def backward(q: Array, k: Array, v: Array, r: Array) -> list[Array]:
    _, p = attention(q, k, v)
    gv = p.T @ r
    gp = r @ v.T
    gs = p * (gp - (p * gp).sum(axis=1, keepdims=True))
    scale = np.sqrt(q.shape[1])
    gq = gs @ k / scale
    gk = gs.T @ q / scale
    return [gq, gk, gv]


rng = np.random.default_rng(7)
arrays = [rng.normal(size=(2, 3)), rng.normal(size=(4, 3)), rng.normal(size=(4, 2))]
r = rng.normal(size=(2, 2))
analytic = backward(*arrays, r)
numeric: list[Array] = []
epsilon = 1e-6
for array in arrays:
    gradient = np.empty_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + epsilon
        plus = float(np.sum(attention(*arrays)[0] * r))
        array[index] = original - epsilon
        minus = float(np.sum(attention(*arrays)[0] * r))
        array[index] = original
        gradient[index] = (plus - minus) / (2 * epsilon)
    numeric.append(gradient)
errors = [float(np.max(np.abs(a - n))) for a, n in zip(analytic, numeric)]
for expected, actual, array in zip(analytic, numeric, arrays):
    assert expected.shape == array.shape
    assert np.allclose(expected, actual, atol=1e-7, rtol=1e-6)
print([gradient.shape for gradient in analytic])
print(f"max absolute error: {max(errors):.3e}")
assert max(errors) < 1e-7
```

The first line is `[(2, 3), (4, 3), (4, 2)]`.
The second line reports a platform-dependent small floating-point error, usually around `1e-10` to `1e-9`.
The centered difference perturbs only one element at a time and restores it before moving to the next.
Float64 and moderate logits keep roundoff and saturation from hiding the derivative.
Finite differences check this example, not every possible input or masking policy.

</details>

### B2. Compare asyncio, threads, and processes ●●

Simulate four waiting operations and four pure-Python CPU jobs.
Use asyncio for cooperative waits, threads for blocking waits, and a process pool for CPU jobs.
Also time the same CPU jobs sequentially and in a thread pool.
Require identical result lists, not a fixed speedup.
Explain the expected effect of the GIL on the CPU thread case in a conventional CPython build.
Expected behavior: both waiting implementations return `[0, 1, 2, 3]`, and all CPU implementations agree exactly.

<details markdown="1">
<summary>Solution</summary>

Save as `concurrency_lab.py` and run `uv run python concurrency_lab.py` from a terminal.
The entry-point guard makes the module safe for process spawning.

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from time import perf_counter, sleep


def blocking_wait(item: int) -> int:
    sleep(0.05)
    return item


async def async_wait(item: int) -> int:
    await asyncio.sleep(0.05)
    return item


async def async_batch() -> list[int]:
    return list(await asyncio.gather(*(async_wait(i) for i in range(4))))


def cpu_job(n: int) -> int:
    return sum(i * i for i in range(n))


def main() -> None:
    started = perf_counter()
    async_results = asyncio.run(async_batch())
    print(f"async wait: {perf_counter() - started:.4f}s")
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        thread_results = list(pool.map(blocking_wait, range(4)))
    print(f"thread wait: {perf_counter() - started:.4f}s")
    assert async_results == thread_results == [0, 1, 2, 3]
    print(async_results)

    jobs = [200_000, 210_000, 220_000, 230_000]
    started = perf_counter()
    sequential = [cpu_job(n) for n in jobs]
    print(f"sequential CPU: {perf_counter() - started:.4f}s")
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        threaded = list(pool.map(cpu_job, jobs))
    print(f"thread CPU: {perf_counter() - started:.4f}s")
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        processed = list(pool.map(cpu_job, jobs))
    print(f"process CPU including startup: {perf_counter() - started:.4f}s")
    assert sequential == threaded == processed
    print("All CPU results agree")


if __name__ == "__main__":
    main()
```

Expected observations include `[0, 1, 2, 3]` and `All CPU results agree`.
Waiting operations overlap because each either yields to the event loop or blocks in an operation that releases the GIL.
The pure-Python CPU generator executes bytecode and does not gain general CPU parallelism from threads on a GIL-enabled interpreter.
Processes can execute those jobs on separate cores, but these small jobs may not repay startup and transfer costs.
Repeat with larger jobs before drawing a throughput conclusion.
On a free-threaded build, report that configuration and measure rather than repeating the conventional-GIL prediction.

</details>

### B3. Profile a vectorized and compiled reduction ●●

Compute the sum of squared distances from a fixed center for 100,000 float64 values.
Implement a Python loop, NumPy vectorization, and a fused Numba loop.
Warm up Numba, repeat timings, verify numerical agreement, and print the input payload size.
Report Python-relative speed ratios without requiring them to exceed one.
Also express a filtered rate calculation in Polars without a row callback.
Expected behavior: all reduction values agree within tolerance, the input payload is `800000` bytes, and Polars produces `[10.0, 15.0]`.

<details markdown="1">
<summary>Solution</summary>

Save as `profile_lab.py` and run `uv run python profile_lab.py`.

```python
from collections.abc import Callable
from statistics import median
from time import perf_counter

import numpy as np
import polars as pl
from numba import njit


def python_distance(x: np.ndarray, center: float) -> float:
    total = 0.0
    for value in x:
        delta = float(value) - center
        total += delta * delta
    return total


def numpy_distance(x: np.ndarray, center: float) -> float:
    return float(np.sum((x - center) ** 2))


@njit
def numba_distance(x: np.ndarray, center: float) -> float:
    total = 0.0
    for value in x:
        delta = value - center
        total += delta * delta
    return total


def measure(operation: Callable[[], float]) -> tuple[float, float]:
    times = []
    result = 0.0
    for _ in range(7):
        started = perf_counter()
        result = operation()
        times.append(perf_counter() - started)
    return result, median(times)


x = np.arange(100_000, dtype=np.float64) / 100_000
center = 0.5
_ = numba_distance(x, center)
operations = {
    "python": lambda: python_distance(x, center),
    "numpy": lambda: numpy_distance(x, center),
    "numba": lambda: numba_distance(x, center),
}
measurements = {name: measure(operation) for name, operation in operations.items()}
reference = measurements["python"][0]
for name, (value, seconds) in measurements.items():
    assert np.isclose(value, reference, rtol=1e-10, atol=1e-8)
    ratio = measurements["python"][1] / seconds
    print(f"{name}: value={value:.6f}, median={seconds:.6f}s, ratio={ratio:.2f}x")
print(f"payload={x.nbytes} bytes")
assert x.nbytes == 800_000

frame = pl.DataFrame({"tokens": [10, 30, 20], "seconds": [1.0, 2.0, 0.0]})
rates = (
    frame.lazy()
    .filter(pl.col("seconds") > 0)
    .select((pl.col("tokens") / pl.col("seconds")).alias("rate"))
    .collect()["rate"]
    .to_list()
)
assert rates == [10.0, 15.0]
print(rates)
```

All three displayed values round to `8333.333335`.
The payload and rate lines are `payload=800000 bytes` and `[10.0, 15.0]`.
The NumPy expression may create intermediate arrays, whereas the Numba loop accumulates a scalar without full-size temporaries.
All implementations still read the input array.
Use the [notes' memory profiling procedure](notes.md#memory-profiling-and-numba) to inspect traced peaks and state its native-memory limitation.
Report whether the measured ratios support a speedup on your hardware instead of promising a universal result.

</details>

### B4. Validate a prediction boundary with Pydantic v2 and pytest ●●

Implement a two-feature linear prediction with weights $(2,-1)$.
Require exactly two finite numeric features and reject unknown fields.
Reject a string feature, a missing feature, an infinite feature, and an extra field.
Keep the numeric function independent of Pydantic.
Provide pytest contract checks and a command that runs them.
Expected behavior: `{"features": [3.0, 4.0]}` predicts `2.0`, and all six tests pass.

<details markdown="1">
<summary>Solution</summary>

Save the first block as `prediction.py`.
The boundary validates input once, then calls a function that accepts ordinary floats.

```python
from collections.abc import Mapping
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    features: list[FiniteFloat] = Field(min_length=2, max_length=2)


def linear_score(first: float, second: float) -> float:
    return 2.0 * first - second


def predict(payload: Mapping[str, object]) -> float:
    request = PredictionRequest.model_validate(payload)
    first, second = request.features
    return linear_score(first, second)


if __name__ == "__main__":
    print(predict({"features": [3.0, 4.0]}))
```

Save the second block as `test_prediction.py` in the same directory.
These checks exercise the public computation and input boundary.
They do not inspect Pydantic's internal schema or the kernel's implementation.

```python
import pytest
from pydantic import ValidationError

from prediction import linear_score, predict


def test_valid_prediction() -> None:
    assert predict({"features": [3.0, 4.0]}) == pytest.approx(2.0)


def test_numeric_kernel() -> None:
    assert linear_score(0.5, -1.0) == pytest.approx(2.0)


def test_string_feature_is_rejected() -> None:
    with pytest.raises(ValidationError):
        predict({"features": ["3.0", 4.0]})


def test_wrong_length_is_rejected() -> None:
    with pytest.raises(ValidationError):
        predict({"features": [3.0]})


def test_infinite_feature_is_rejected() -> None:
    with pytest.raises(ValidationError):
        predict({"features": [float("inf"), 4.0]})


def test_extra_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        predict({"features": [3.0, 4.0], "debug": True})
```

Run the public example and checks:

```bash
uv run python prediction.py
uv run pytest -q test_prediction.py
uv run ruff check prediction.py test_prediction.py
uv run mypy prediction.py test_prediction.py
```

Expected output includes `2.0`, `6 passed`, and `Success: no issues found in 2 source files`.
Ruff reports no violations.
Pydantic strict floats accept integer numeric inputs but reject numeric strings; this exercise deliberately accepts finite numbers, not only Python objects whose exact type is `float`.
The field length constraint makes unpacking into two features valid after successful validation.
Type hints alone cannot reject malformed external input; the runtime boundary provides that contract.
Finite inputs can still overflow a floating-point computation if they are extremely large.
For a production service, bound feature magnitudes or reject nonfinite outputs according to its domain contract.
The [CI example](notes.md#cicd-for-ml-with-github-actions) runs these same checks from a lockfile-backed environment.

</details>

## Review your evidence

Record the six handwritten solutions and four coding outcomes.
Explain one mistake you corrected in a shape, probability calculation, or timing interpretation.
Pass the [exit criteria](index.md#exit-criteria) before starting another branch.
