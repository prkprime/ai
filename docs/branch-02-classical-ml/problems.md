# Branch 2 problems — Classical statistical and machine learning

Use the [notes](notes.md) only after your first attempt.
The [schedule](index.md#session-schedule) assigns these problems to seven core hours.
Write intermediate values so another reader can recompute your answer.
All random code examples set a seed, but use the stated observable behavior rather than exact fitted scores as the contract.

Difficulty: ● applies one method; ●● combines methods; ●●● diagnoses an ambiguous situation.
Part A has eight problems and Part B has four problems.
Score Part A out of eight, giving one point for a correct method and result in each problem.
Pass with at least six correct Part A solutions and all four Part B behaviors reproduced.

## Part A — Pen and paper

### A1. Ridge by hand ●

Use $X=(1,2)^T$ and $y=(1,2)^T$ with no intercept.
The objective is $J(w)=\tfrac12\|Xw-y\|^2+\tfrac\lambda2w^2$.
Set $\lambda=1$.

1. Derive the gradient without quoting the matrix solution.
2. Calculate the minimizer, fitted values, and residual sum of squares.
3. Compare the objective with the unregularized minimizer $w=1$.
4. Explain why you must state whether the loss contains a factor $1/n$.

<details markdown="1">
<summary>Solution</summary>

Expand the objective in this one-dimensional case:

$$
J(w)=\frac12[(w-1)^2+(2w-2)^2]+\frac12w^2.
$$

Differentiate each term:

$$
J'(w)=(w-1)+2(2w-2)+w=6w-5.
$$

Set $J'(w)=0$ to obtain $\hat w=5/6$.
The second derivative is $6>0$, so this is the unique minimum.
Equivalently, $X^TX=5$, $X^Ty=5$, and $(5+1)^{-1}5=5/6$.

The fitted values are $(5/6,5/3)$.
The residuals $X\hat w-y$ are $(-1/6,-1/3)$.
Their squared sum is

$$
RSS=\frac1{36}+\frac19=\frac5{36}.
$$

The penalty contribution is $\tfrac12(25/36)=25/72$.
The data-loss contribution is $5/72$.
The total objective is $30/72=5/12\approx0.4167$.

At $w=1$, the residual sum is zero but the objective is $1/2$.
The penalized minimizer accepts training error to reduce coefficient magnitude.
Dividing only the data loss by $n$ changes its relative weight against the penalty.
To preserve the minimizer, adjust the penalty normalization too.

</details>

### A2. Logistic gradient and one update ●

Use two observations with scalar features $x_1=1$, $x_2=2$ and labels $y_1=0$, $y_2=1$.
Start with $w=0$, $b=0$ and no penalty.
Use mean binary cross-entropy and learning rate $\eta=0.4$.

1. Derive $\partial\ell/\partial z$ from the Bernoulli likelihood.
2. Calculate the initial probabilities, coefficient gradient, and intercept gradient.
3. Take one full-batch gradient step.
4. State the shapes if you instead have $n$ observations and $d$ features.

<details markdown="1">
<summary>Solution</summary>

For one row, $\ell=-y\log p-(1-y)\log(1-p)$ and $p=\sigma(z)$.
Differentiate with respect to $p$:

$$
\frac{\partial\ell}{\partial p}=-\frac yp+\frac{1-y}{1-p}.
$$

Multiply by $\partial p/\partial z=p(1-p)$:

$$
\frac{\partial\ell}{\partial z}
=-y(1-p)+(1-y)p=p-y.
$$

Initially $z=(0,0)$ and $p=(0.5,0.5)$.
Therefore $p-y=(0.5,-0.5)$.
The mean gradients are

$$
\nabla_wJ=\frac{1(0.5)+2(-0.5)}2=-0.25,
\qquad
\partial_bJ=\frac{0.5-0.5}2=0.
$$

The update gives $w_{new}=0-0.4(-0.25)=0.1$ and $b_{new}=0$.
New probabilities are $\sigma(0.1)\approx0.52498$ and $\sigma(0.2)\approx0.54983$.
Both move above $0.5$, but the positive row receives the larger increase.
The shared linear model cannot independently move each observation's probability.

For the general case, $X$ has shape $(n,d)$.
The residual vector $p-y$ has shape $(n,)$.
$X^T(p-y)/n$ has shape $(d,)$ and the intercept gradient is scalar.
An L2 penalty adds $\lambda w$ to the coefficient gradient but not to an unpenalized intercept.

</details>

### A3. Gini, entropy, and a boosted split ●●

A classification node contains eight observations, with four positive and four negative labels.
A candidate split creates a left child with three positives and one negative, and a right child with one positive and three negatives.

1. Calculate parent and child Gini impurities and the weighted gain.
2. Calculate parent and child binary entropies in bits and the weighted gain.
3. In a separate boosted-tree example, use $G_L=-2$, $G_R=2$, $H_L=H_R=1$, $\lambda=1$, and $\gamma=0$.
   Calculate both leaf weights and the second-order gain.
4. Explain why impurity gain and second-order gain are not the same objective.

<details markdown="1">
<summary>Solution</summary>

The parent Gini impurity is $1-0.5^2-0.5^2=0.5$.
Each child has proportions $0.75$ and $0.25$, so each impurity is

$$
1-0.75^2-0.25^2=1-0.5625-0.0625=0.375.
$$

The weighted child impurity is $\tfrac48(0.375)+\tfrac48(0.375)=0.375$.
The gain is $0.5-0.375=0.125$.

The parent entropy is one bit.
Each child's entropy is

$$
-0.75\log_2(0.75)-0.25\log_2(0.25)
\approx0.811278.
$$

The weighted child entropy is also $0.811278$.
The entropy gain is $1-0.811278=0.188722$ bits.
Both impurity criteria prefer more homogeneous children, but their numeric scales differ.

For the boosted example, the optimal left weight is $-G_L/(H_L+\lambda)=2/2=1$.
The right weight is $-2/2=-1$.
The parent gradient sum is zero, so its optimized contribution has zero gain term.
The split gain is

$$
\frac12\left[\frac{(-2)^2}{2}+\frac{2^2}{2}-\frac{0^2}{3}\right]=2.
$$

This gain approximates improvement in the chosen differentiable loss, including leaf regularization.
Gini and entropy gains measure class impurity reductions directly.
Do not compare their raw numbers across these two separate examples.

</details>

### A4. Imbalance, PR, ROC, and F1 ●●

A detector evaluates 1,000 observations, including 10 positives.
At one threshold it finds eight true positives and produces 20 false positives.

1. Construct the confusion counts.
2. Calculate precision, recall, FPR, binary F1, and accuracy.
3. Explain why this is an attractive-looking ROC point but a costly alert stream.
4. Can you calculate ROC AUC or average precision from these counts alone?
5. For a different single-label two-class classifier, the confusion matrix is `[[90, 0], [10, 0]]`, with rows as true classes.
   Calculate macro and micro F1.

<details markdown="1">
<summary>Solution</summary>

There are $TP=8$ and $FN=10-8=2$.
There are $FP=20$ and $TN=990-20=970$.
The four counts sum to $8+2+20+970=1000$.

Precision is $8/(8+20)=2/7\approx0.285714$.
Recall is $8/10=0.8$.
FPR is $20/990=2/99\approx0.020202$.
Binary F1 is

$$
\frac{2TP}{2TP+FP+FN}=\frac{16}{16+20+2}=\frac8{19}\approx0.421053.
$$

Accuracy is $(8+970)/1000=0.978$.
A detector that always predicts negative achieves $0.99$ accuracy but zero positive recall.
Accuracy alone therefore gives a misleading comparison for the detection goal.

The point has 80% recall at about 2% FPR.
Because negatives are numerous, those 2% create 20 false alerts for only eight correct alerts.
You need alert cost, review capacity, and benefits before choosing this threshold.
One threshold does not reveal the ordering of scores elsewhere.
ROC AUC and average precision require the ranking or the full threshold curve.

In the second example, class 0 has $TP_0=90$, $FP_0=10$, and $FN_0=0$.
Its F1 is $180/(180+10)=18/19\approx0.947368$.
Class 1 has no correct positive predictions, so its F1 is zero.
Macro F1 is $(18/19+0)/2=9/19\approx0.473684$.

For micro averaging, pooled $TP=90$, $FP=10$, and $FN=10$.
Micro F1 is $180/(180+10+10)=0.9$.
It equals accuracy here because this is single-label classification with both classes included.
The low macro value exposes complete failure on class 1.

</details>

### A5. Numeric bias–variance decomposition ●●

At a fixed input, the true mean response is $f(x)=3$.
Independent test noise is $-1$ or $1$ with equal probability.
Training procedure A predicts $2$ or $4$ with equal probability across training samples.
Procedure B always predicts $2.5$.

1. Derive the decomposition by adding and subtracting the mean fitted prediction.
2. Calculate squared bias, variance, noise variance, and expected test MSE for both procedures.
3. Check procedure A by enumerating all four equally likely prediction/noise pairs.
4. State the independence assumption that makes the cross terms vanish.

<details markdown="1">
<summary>Solution</summary>

Let $m=E_D[\hat f_D(x)]$.
Write $Y-\hat f=\varepsilon+(f-m)+(m-\hat f)$.
Square and take expectation.
The noise has zero mean and is independent of the training-based prediction.
Also $E_D[m-\hat f]=0$.
All cross terms vanish, leaving

$$
E[(Y-\hat f)^2]=\sigma^2+(f-m)^2+E[(\hat f-m)^2].
$$

For A, $m=(2+4)/2=3$.
Squared bias is $(3-3)^2=0$.
Variance is $[(2-3)^2+(4-3)^2]/2=1$.
Noise variance is $[(-1)^2+1^2]/2=1$.
Expected MSE is $0+1+1=2$.

For B, $m=2.5$.
Squared bias is $(3-2.5)^2=0.25$ and prediction variance is zero.
Expected MSE is $0.25+0+1=1.25$.

For the direct A check, the test response is $Y=2$ or $4$.
The four pairs $(Y,\hat f)$ are $(2,2)$, $(2,4)$, $(4,2)$, and $(4,4)$.
Squared errors are $0,4,4,0$.
Their average is $8/4=2$.

For B, the two errors are $(2-2.5)^2=0.25$ and $(4-2.5)^2=2.25$.
Their mean is $1.25$.
If test noise correlates with the training procedure, you cannot discard its cross term without further analysis.

</details>

### A6. Brier score and reliability ●●

First use $p=(0.1,0.8,0.6,0.3)$ and $y=(0,1,1,0)$.
Calculate the unscaled binary Brier score.
Then consider eight observations in two forecast groups:

- Four observations receive probability $0.25$; exactly one has label 1.
- Four observations receive probability $0.75$; exactly three have label 1.

Calculate reliability, resolution, uncertainty, and Brier score for the grouped forecasts.
Explain why the same decomposition needs care if a plotting bin contains many different probabilities.

<details markdown="1">
<summary>Solution</summary>

For the first dataset, compute each squared error:

$$
(0.1-0)^2=0.01,\quad
(0.8-1)^2=0.04,\quad
(0.6-1)^2=0.16,\quad
(0.3-0)^2=0.09.
$$

The sum is $0.30$ and the mean is $0.30/4=0.075$.

For the grouped dataset, the empirical base rate is $(1+3)/8=0.5$.
Within-group rates are $0.25$ and $0.75$.
They match their forecast values exactly, so

$$
\mathrm{REL}=\frac48(0.25-0.25)^2+\frac48(0.75-0.75)^2=0.
$$

Resolution is

$$
\mathrm{RES}=\frac48(0.25-0.5)^2+\frac48(0.75-0.5)^2=0.0625.
$$

Uncertainty is $\mathrm{UNC}=0.5(1-0.5)=0.25$.
Therefore $BS=\mathrm{REL}-\mathrm{RES}+\mathrm{UNC}=0.1875$.

Check directly in the low-probability group.
The one event has squared error $(0.25-1)^2=0.5625$.
The three nonevents contribute $3(0.25^2)=0.1875$.
This group's error sum is $0.75$.
The high-probability group has the same error sum by symmetry.
Total Brier score is $(0.75+0.75)/8=0.1875$.

The exact identity groups identical forecast values.
A reliability plot usually groups different forecast values into one bin.
Replacing all those forecasts by a bin mean changes the original squared-error calculation.
The simple three-term formula then describes coarsened forecasts unless you include the additional within-bin effects.
Empirical agreement on eight observations also does not establish population calibration.

</details>

### A7. PCA, Lloyd updates, and density ●●

Use covariance matrix $S=\begin{pmatrix}2&1\\1&2\end{pmatrix}$.
Find its leading unit eigenvector, explained-variance fraction, and one-component reconstruction of centered point $(2,0)$.
Then perform one Lloyd iteration for points $0,2,8,10$ initialized at $0,10$.
Finally classify $0,0.1,0.2,3$ as core, border, or noise for DBSCAN with radius $0.15$ and `min_samples=3`.

<details markdown="1">
<summary>Solution</summary>

The characteristic polynomial is $(2-\lambda)^2-1$.
Its roots are $\lambda_1=3$ and $\lambda_2=1$.
For $\lambda_1=3$, the eigenvector equation gives $v_1=v_2$.
Normalize to obtain $v=(1,1)/\sqrt2$.
The explained-variance fraction is $3/(3+1)=0.75$.

The projected score is $(2,0)\cdot v=2/\sqrt2=\sqrt2$.
Reconstruction is $\sqrt2\,v=(1,1)$.
The residual is $(1,-1)$ and its squared norm is $2$.
The reconstruction uses centered coordinates because the point is already centered.

Lloyd assignment sends $0,2$ to centroid $0$ and $8,10$ to centroid $10$.
The means update to $(0+2)/2=1$ and $(8+10)/2=9$.
The new objective is $1+1+1+1=4$.
The initial objective was $0+4+4+0=8$.

DBSCAN counts each point itself.
The radius-neighborhood of $0.1$ contains $0,0.1,0.2$, so it is core.
The neighborhoods of $0$ and $0.2$ each contain two points.
They are not core, but they lie within the core point's radius, so they are border points.
Point $3$ has only itself and is not reachable from a core, so it is noise.
One cluster contains the three small values.

</details>

### A8. Convexity, sparsity, and validation choices ●●●

1. Prove that the Hessian of unpenalized mean logistic loss is positive semidefinite.
2. Under $X^TX/n=I$, calculate the Lasso optimum for $z=X^Ty/n=(0.2,1.4)$ and $\alpha=0.5$.
3. Calculate two Momentum steps for $J(w)=w^2/2$, $w_0=2$, $v_0=0$, $\eta=0.1$, and $\beta=0.9$ using $v_{t+1}=\beta v_t+g_t$.
4. Choose a validation design for fraud transactions from repeated cards over time when deployment predicts next month's transactions for a mixture of existing and unseen cards.
5. Recent inputs change but labels arrive after 30 days.
   State what you can claim today and what you must defer.

<details markdown="1">
<summary>Solution</summary>

For any vector $a$,

$$
a^T\left(\frac1nX^TWX\right)a
=\frac1n\sum_i p_i(1-p_i)(x_i^Ta)^2\geq0.
$$

Thus the Hessian is positive semidefinite and the loss is convex in its coefficients.
It need not be strictly convex when the design lacks rank.
Unpenalized separable data can lack a finite minimizer.

For orthonormal columns, apply soft thresholding.
The first coordinate becomes $\max(0.2-0.5,0)=0$.
The second becomes $1.4-0.5=0.9$.
The solution is $(0,0.9)$.
The L1 subgradient interval at zero permits the first coefficient to remain exactly zero.

For Momentum, $g_0=2$, $v_1=0.9(0)+2=2$, and $w_1=2-0.1(2)=1.8$.
Next $g_1=1.8$, $v_2=0.9(2)+1.8=3.6$, and $w_2=1.8-0.1(3.6)=1.44$.

For fraud, use forward time windows that reproduce next-month prediction.
Train features only from information available before each prediction time.
Choose a gap from chargeback delay, label horizon, and feature latency.
Report separate existing-card and unseen-card slices in the later holdout.
If you also claim generalization to entirely unseen cards, create group-disjoint evaluation for that claim while preserving time ordering.
Random stratification alone leaks chronology and can mix the same cards across folds.
Plain group folds alone can train on future transactions.

Today you can report input-distribution changes, schema failures, and prediction-distribution changes.
You cannot establish a change in $P(Y\mid X)$ or measured performance loss without outcome evidence or justified additional assumptions.
Wait for mature labels, then compare ranking, calibration, costs, and slices on a later time window.
Do not assume immediate retraining fixes the problem.

</details>

## Part B — Code

Run each reference as a separate Python program.
Use `uv run --with numpy --with scikit-learn python solution.py`.
The reference solutions use only synthetic arrays, so they need no external dataset.
Each solution keeps assertions on observable outputs or data-partition contracts.

### B1. Verify a Ridge implementation and build a pipeline ●●

Reproduce A1 with `numpy.linalg.solve` and `Ridge`.
Then build a median-imputation, scaling, and Ridge pipeline for synthetic regression data with missing values.
Evaluate five independent shuffled folds.
Do not fit the scaler before cross-validation.
Print the manual coefficient, the fitted library coefficient, and the number of finite fold scores.

<details markdown="1">
<summary>Reference solution</summary>

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_small = np.array([[1.0], [2.0]])
y_small = np.array([1.0, 2.0])
manual = np.linalg.solve(
    X_small.T @ X_small + np.eye(1), X_small.T @ y_small
)[0]
fitted = Ridge(alpha=1.0, fit_intercept=False).fit(X_small, y_small)
np.testing.assert_allclose(fitted.coef_, [5 / 6])
print(f"manual={manual:.6f}, sklearn={fitted.coef_[0]:.6f}")

X, y = make_regression(n_samples=120, n_features=5, noise=5, random_state=7)
X[::11, 0] = np.nan
pipeline = make_pipeline(
    SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)
)
cv = KFold(n_splits=5, shuffle=True, random_state=7)
scores = cross_val_score(
    pipeline, X, y, cv=cv, scoring="neg_mean_squared_error"
)
assert scores.shape == (5,)
assert np.isfinite(scores).all()
print("finite folds:", np.isfinite(scores).sum())
```

Expected output:

```text
manual=0.833333, sklearn=0.833333
finite folds: 5
```

The small calculation disables the intercept to match A1 exactly.
The full pipeline keeps its intercept because that is appropriate for the generated regression task.
Negative scoring is an API convention: larger values remain better, so MSE scorers return its negative.
The program checks finite held-out losses, not an arbitrary accuracy threshold on random data.

</details>

### B2. Reproduce PCA and distinguish clustering assumptions ●●

Calculate the eigendecomposition of the covariance matrix in A7.
Use its leading direction to reconstruct $(2,0)$.
Then fit one-dimensional k-means to $0,2,8,10$ with explicit initial centers and DBSCAN to $0,0.1,0.2,3$.
Print the variance fraction, reconstruction, sorted k-means centers, k-means inertia, and DBSCAN labels.
Do not assert arbitrary k-means label numbers because cluster identifiers can permute.

<details markdown="1">
<summary>Reference solution</summary>

```python
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

S = np.array([[2.0, 1.0], [1.0, 2.0]])
values, vectors = np.linalg.eigh(S)
v = vectors[:, np.argmax(values)]
point = np.array([2.0, 0.0])
reconstruction = (point @ v) * v
fraction = values.max() / values.sum()
np.testing.assert_allclose(reconstruction, [1.0, 1.0])
print("variance fraction:", fraction)
print("reconstruction:", np.round(reconstruction, 6).tolist())

X = np.array([[0.0], [2.0], [8.0], [10.0]])
kmeans = KMeans(
    n_clusters=2, init=np.array([[0.0], [10.0]]),
    n_init=1, random_state=7,
).fit(X)
print("centers:", np.sort(kmeans.cluster_centers_.ravel()).tolist())
print("inertia:", kmeans.inertia_)

Z = np.array([[0.0], [0.1], [0.2], [3.0]])
dbscan = DBSCAN(eps=0.15, min_samples=3).fit(Z)
print("DBSCAN:", dbscan.labels_.tolist())
assert len(set(dbscan.labels_) - {-1}) == 1
assert dbscan.labels_[-1] == -1
```

Expected output:

```text
variance fraction: 0.75
reconstruction: [1.0, 1.0]
centers: [1.0, 9.0]
inertia: 4.0
DBSCAN: [0, 0, 0, -1]
```

The eigenvector sign does not affect the reconstruction because it appears twice.
`eigh` returns eigenvalues in ascending order, so the program selects the largest explicitly.
K-means inertia is the sum of squared distances to assigned centers.
DBSCAN uses connectivity and a noise label instead of minimizing that inertia.

</details>

### B3. Inspect deployment-aligned cross-validation ●●●

Create 24 rows with six groups of four rows and balanced alternating binary labels.
Show that stratified validation preserves class counts, grouped validation prevents group overlap, and temporal validation leaves a two-row gap.
Print each strategy's validation sizes and the temporal train/validation boundaries.
Explain which strategy you use for unseen-device deployment and which for forecasting.

<details markdown="1">
<summary>Reference solution</summary>

```python
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold, TimeSeriesSplit

X = np.arange(48).reshape(24, 2)
y = np.tile([0, 1], 12)
groups = np.repeat(np.arange(6), 4)

sizes = []
for train, val in StratifiedKFold(
    n_splits=3, shuffle=True, random_state=7
).split(X, y):
    assert set(train).isdisjoint(val)
    assert np.bincount(y[val]).tolist() == [4, 4]
    sizes.append(len(val))
print("stratified sizes:", sizes)

sizes = []
for train, val in GroupKFold(n_splits=3).split(X, y, groups=groups):
    assert set(groups[train]).isdisjoint(groups[val])
    sizes.append(len(val))
print("grouped sizes:", sizes)

boundaries = []
for train, val in TimeSeriesSplit(
    n_splits=3, test_size=4, gap=2
).split(X):
    assert train.max() + 2 < val.min()
    boundaries.append((int(train.max()), int(val.min())))
print("temporal boundaries:", boundaries)
```

Expected output:

```text
stratified sizes: [8, 8, 8]
grouped sizes: [8, 8, 8]
temporal boundaries: [(9, 12), (13, 16), (17, 20)]
```

Use device identifiers as groups when your claim concerns unseen devices.
Use forward splits when the target concerns future observations.
For future unseen devices, enforce both restrictions rather than choosing whichever gives the higher score.
The two-row gap in this exercise is an index contract, not a justified label-delay policy for a real dataset.

</details>

### B4. Report discrimination and calibrate probabilities ●●●

First reproduce the binary Brier score from A6 exactly.
Then generate an imbalanced classification dataset with a fixed seed.
Train a standardized logistic pipeline through cross-validated sigmoid calibration.
Reserve a final test split before all fitting.
Report ROC AUC, average precision, Brier score, and a reliability diagram's numeric bins.
Do not assert that calibration always improves every metric on a finite test sample.

<details markdown="1">
<summary>Reference solution</summary>

```python
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

observed = np.array([0, 1, 1, 0])
forecast = np.array([0.1, 0.8, 0.6, 0.3])
exact_brier = brier_score_loss(observed, forecast)
np.testing.assert_allclose(exact_brier, 0.075)
print("hand Brier:", round(exact_brier, 3))

X, y = make_classification(
    n_samples=800, n_features=8, n_informative=5,
    weights=[0.9, 0.1], random_state=7,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=7
)
base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=cv)
model.fit(X_train, y_train)
p = model.predict_proba(X_test)[:, 1]
metrics = {
    "ROC AUC": roc_auc_score(y_test, p),
    "average precision": average_precision_score(y_test, p),
    "Brier": brier_score_loss(y_test, p),
}
for name, value in metrics.items():
    assert np.isfinite(value) and 0 <= value <= 1
    print(f"{name}: {value:.4f}")

fraction_positive, mean_probability = calibration_curve(
    y_test, p, n_bins=5, strategy="quantile"
)
assert fraction_positive.shape == mean_probability.shape
assert 1 <= len(fraction_positive) <= 5
print("test probabilities:", p.shape)
print("calibration bins:", len(fraction_positive))
print("bin means:", np.round(mean_probability, 3).tolist())
print("bin frequencies:", np.round(fraction_positive, 3).tolist())
```

Expected behavior:

- The first line is `hand Brier: 0.075`.
- All three reported metrics are finite numbers between zero and one.
- Test probabilities have shape `(200,)`.
- The synthetic data produces five populated quantile bins.
- Bin means and bin frequencies have the same length and lie in `[0, 1]`.
- Calibration does not access `y_test` during `fit`.

Average precision and ROC AUC evaluate ranking with different weighting.
Brier score evaluates probability quality, including both reliability and resolution.
The calibration table estimates empirical frequencies from a small test set, so expect sampling uncertainty.
Do not tune the calibration method after reading these test results and still call the same split untouched.

</details>

## Completion check

Recompute A1, A3, and A6 on a blank page.
Explain why all four coding exercises keep preprocessing or calibration inside the correct training boundary.
Use the [exit criteria](index.md#exit-criteria) to record what you can derive and implement without a reference.
