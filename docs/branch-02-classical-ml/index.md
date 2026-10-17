# Branch 2 — Classical statistical and machine learning

Build a trustworthy tabular baseline before you build a neural network.
This branch connects loss functions to estimators, then connects validation design to deployment claims.
You spend exactly **7 core hours**: 4 hours on algorithms and 3 hours on optimization, validation, and diagnostics.
Use the [notes](notes.md) to study and the [problems](problems.md) to check your work.

## Learning objectives

- Derive Ridge and logistic gradients, and calculate a Ridge solution by hand for 2.1.
- Compare Random Forests, boosted trees, PCA, neighborhood embeddings, and clustering methods for 2.1.
- Implement a preprocessing pipeline that fits transformations inside each training fold for 2.1.
- Derive convexity conditions, optimizer updates, and the squared-error bias–variance decomposition for 2.2.
- Select a cross-validator from the deployment unit and time ordering for 2.2.
- Calculate imbalance-sensitive metrics and diagnose calibration, overfitting, and drift for 2.2.

## Prerequisites

- Review linear algebra, eigendecomposition, and derivatives in [Branch 1 notes, subtopic 1.1](../branch-01-math-foundations/notes.md).
- Review expectation, variance, conditional probability, and entropy in [Branch 1 notes, subtopic 1.2](../branch-01-math-foundations/notes.md).
- Complete the relevant linear algebra and probability exercises in [Branch 1 problems](../branch-01-math-foundations/problems.md).
- Bring Python, NumPy array operations, and basic environment-management skills from Branch 1.

## Hour budget

| Subtopic | Hours | Evidence you produce |
|---|---:|---|
| 2.1 Supervised and unsupervised algorithms | 4 | Derived objectives, tree split, PCA projection, clustering comparison |
| 2.2 Optimization, validation, and error diagnostics | 3 | Split policy, metric report, calibration and drift decision |
| **Total** | **7** | **4 + 3 = 7** |

## Session schedule

Each row includes reading, selected video segments, and practice.
Do not add video time on top of these seven hours.
Treat optional experiments as extensions outside the core budget.

| Hour | Subtopic | Study and practice | Completion evidence |
|---|---|---|---|
| 1 | 2.1 | 15 min Ridge video; 25 min linear/logistic objectives; 20 min A1–A2 | Ridge solution and logistic gradient |
| 2 | 2.1 | 10 min forest video; 30 min trees and boosting; 20 min A3 and early-stopping code | Split gain and validation-only stopping |
| 3 | 2.1 | 15 min PCA video; 25 min PCA, t-SNE, UMAP; 20 min B1 | Projection caveats and leakage-safe pipeline |
| 4 | 2.1 | 25 min clustering; 35 min A7 and B2 | Clustering assumptions and density |
| 5 | 2.2 | 20 min convexity and optimizers; 7 min cross-validation video; 33 min A8 and B3 | Correct split invariants |
| 6 | 2.2 | 25 min metrics and calibration; 35 min A4, A6, and B4 | Counts, Brier score, calibrated probabilities |
| 7 | 2.2 | 20 min bias–variance and drift; 20 min A5; 20 min exit check | Error decomposition and deployment decision |
| **Total** | | **7 × 60 min = 420 min** | |

## Subtopic map

| Subtopic | Teaching sections | Primary sources |
|---|---|---|
| 2.1 Generalized linear models | [Objectives and derivatives](notes.md#generalized-linear-models) | ISLP; scikit-learn linear models; CS229 |
| 2.1 Tree ensembles | [Split finding and boosting](notes.md#tree-ensembles) | ISLP; XGBoost; LightGBM; CatBoost |
| 2.1 Dimensionality reduction | [PCA and neighborhood embeddings](notes.md#dimensionality-reduction) | scikit-learn decomposition; original t-SNE paper; UMAP |
| 2.1 Clustering | [Objectives and neighborhood structure](notes.md#clustering) | scikit-learn clustering; ISLP |
| 2.2 Convex optimization and updates | [Convexity](notes.md#convex-optimization-basics) · [Optimizers](notes.md#gradient-descent-sgd-and-momentum) | CS229; scikit-learn linear models |
| 2.2 Validation | [Deployment-aligned splits](notes.md#cross-validation-strategies) | scikit-learn cross-validation |
| 2.2 Evaluation | [Metrics and calibration](notes.md#evaluation-metrics) | scikit-learn user guide and calibration |
| 2.2 Error diagnosis | [Bias and variance](notes.md#biasvariance-tradeoff) · [Regularization](notes.md#regularization-geometry) · [Drift](notes.md#handling-data-drift) | ISLP; scikit-learn user guide |

## Video playlist

All four resources are real videos from **StatQuest with Josh Starmer**.
Durations use the video player metadata; a player can round the last second differently.
Watch the assigned segments during the schedule, then finish the videos as optional review.

| Title | Channel | Duration | Link | Why it earns study time |
|---|---|---:|---|---|
| Regularization Part 1: Ridge (L2) Regression | StatQuest with Josh Starmer | 20:27 | [Watch](https://www.youtube.com/watch?v=Q81RR3yKn30) | Connect coefficient shrinkage to a penalized objective. |
| StatQuest: Random Forests Part 1 - Building, Using and Evaluating | StatQuest with Josh Starmer | 9:54 | [Watch](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) | Separate bootstrap sampling from random feature selection. |
| StatQuest: Principal Component Analysis (PCA), Step-by-Step | StatQuest with Josh Starmer | 21:58 | [Watch](https://www.youtube.com/watch?v=FgakZw6K1QQ) | See centering and projection before deriving eigenvectors. |
| Machine Learning Fundamentals: Cross Validation | StatQuest with Josh Starmer | 6:05 | [Watch](https://www.youtube.com/watch?v=fSytzGwwBVw) | Explain why training accuracy does not estimate deployment error. |

## Embedded playlist

Use these players on the built site or the links above on GitHub.
Each player opens the same verified video as its table row.

<iframe width="560" height="315" src="https://www.youtube.com/embed/Q81RR3yKn30" title="Regularization Part 1: Ridge (L2) Regression — StatQuest with Josh Starmer" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/J4Wdy0Wc_xQ" title="StatQuest: Random Forests Part 1 - Building, Using and Evaluating — StatQuest with Josh Starmer" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/FgakZw6K1QQ" title="StatQuest: Principal Component Analysis (PCA), Step-by-Step — StatQuest with Josh Starmer" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/fSytzGwwBVw" title="Machine Learning Fundamentals: Cross Validation — StatQuest with Josh Starmer" frameborder="0" allowfullscreen></iframe>

## Key sources

Every URL below resolves when checked on 2026-10-16.
Use official documentation for library parameters and the textbooks for derivations.
The supplementary workflow book does not replace a primary algorithm reference.

| Author or organization | Title | Type | Link | Covers | Access |
|---|---|---|---|---|---|
| Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor | An Introduction to Statistical Learning, with Applications in Python | Open textbook | [Official book site](https://www.statlearning.com/) | 2.1 regression, trees, PCA, clustering; 2.2 resampling and bias–variance | accessed 2026-10-16 |
| scikit-learn developers | User guide | Official documentation | [Guide](https://scikit-learn.org/stable/user_guide.html) | 2.1 algorithms; 2.2 optimization and evaluation | accessed 2026-10-16 |
| scikit-learn developers | Linear models | Official documentation | [Linear models](https://scikit-learn.org/stable/modules/linear_model.html) | 2.1 loss normalization and estimators; 2.2 regularization | accessed 2026-10-16 |
| Andrew Ng and Stanford CS229 staff | CS229 course materials | University lectures and notes | [Course](https://cs229.stanford.edu/) | 2.1 GLMs; 2.2 optimization and learning theory | accessed 2026-10-16 |
| Abhishek Thakur | Approaching (Almost) Any Machine Learning Problem | Supplementary book repository | [Repository](https://github.com/abhishekkrthakur/approachingalmost) | 2.2 practical validation and metrics | accessed 2026-10-16 |
| XGBoost developers | Introduction to boosted trees | Official documentation | [Tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) | 2.1 second-order loss and split gain | accessed 2026-10-16 |
| LightGBM developers | Features | Official documentation | [Features](https://lightgbm.readthedocs.io/en/stable/Features.html) | 2.1 histograms, leaf-wise growth, categorical splits | accessed 2026-10-16 |
| CatBoost developers | Categorical features | Official documentation | [Categorical features](https://catboost.ai/docs/en/features/categorical-features) | 2.1 one-hot and target-statistic handling | accessed 2026-10-16 |
| CatBoost developers | Main stages of choosing the tree structure | Official documentation | [Algorithm stages](https://catboost.ai/docs/en/concepts/algorithm-main-stages) | 2.1 ordered statistics and tree construction | accessed 2026-10-16 |
| Laurens van der Maaten and Geoffrey Hinton | Visualizing Data using t-SNE | Original paper, JMLR 2008 | [Paper](https://www.jmlr.org/papers/v9/vandermaaten08a.html) | 2.1 KL objective, neighborhoods, perplexity | accessed 2026-10-16 |
| UMAP developers | How UMAP works | Official explanation | [Explanation](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html) | 2.1 fuzzy neighborhood graphs and topology | accessed 2026-10-16 |
| scikit-learn developers | Decomposing signals in components | Official documentation | [Decomposition](https://scikit-learn.org/stable/modules/decomposition.html) | 2.1 PCA and eigenvalue conventions | accessed 2026-10-16 |
| scikit-learn developers | Clustering | Official documentation | [Clustering](https://scikit-learn.org/stable/modules/clustering.html) | 2.1 k-means, DBSCAN, hierarchical linkage | accessed 2026-10-16 |
| scikit-learn developers | Cross-validation | Official documentation | [Validation](https://scikit-learn.org/stable/modules/cross_validation.html) | 2.2 stratified, temporal, and grouped splits | accessed 2026-10-16 |
| scikit-learn developers | Probability calibration | Official documentation | [Calibration](https://scikit-learn.org/stable/modules/calibration.html) | 2.2 reliability diagrams and Brier decomposition | accessed 2026-10-16 |

## Exit criteria

- [ ] Derive the Ridge solution and logistic gradient without copying the notes.
- [ ] Explain the difference between bagging, boosting, and second-order split scoring.
- [ ] Calculate a PCA projection and explain why t-SNE distances between clusters can mislead you.
- [ ] Choose between k-means, DBSCAN, and hierarchical clustering from geometry and noise assumptions.
- [ ] Implement preprocessing inside a cross-validation pipeline.
- [ ] Defend your choice of stratified, grouped, or temporal validation.
- [ ] Report precision, recall, an appropriate area summary, and probability calibration separately.
- [ ] Derive bias–variance and distinguish drift evidence from a retraining decision.
- [ ] Score at least 70% on Part A and reproduce all four Part B behaviors.

Return to the [syllabus overview](../index.md) after you complete the branch.
