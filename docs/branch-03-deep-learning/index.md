# Branch 3 — Deep learning foundations and representation learning

Build the mathematical and implementation habits that make deep networks debuggable.
You trace derivatives, diagnose unstable updates, and compare image encoders before you study transformers in depth.
The core budget is **8 hours**.
Use [the notes](notes.md) for instruction and [the problems](problems.md) for assessment.

## Learning objectives

- **3.1:** Derive every parameter gradient of a two-layer ReLU multilayer perceptron (MLP) and reproduce it with PyTorch autograd.
- **3.1:** Select an initialization and a normalization layer from their variance and axis assumptions.
- **3.1:** Compute activation derivatives, including a softmax Jacobian and the two branches of a SwiGLU gate.
- **3.2:** Compute an AdamW update and distinguish adaptive preconditioning from decoupled weight decay.
- **3.2:** Diagnose an overflow, apply loss scaling in the correct order, and select a learning-rate schedule.
- **3.3:** Calculate convolution shapes, parameter counts, operation counts, and receptive fields.
- **3.3:** Implement patch embeddings and describe how a vision transformer processes an image.
- **3.3:** Compute a symmetric CLIP loss and produce zero-shot class scores from normalized embeddings.

## Prerequisites

- Review vectors, matrix multiplication, the chain rule, and Jacobians in [Branch 1 notes, subtopic 1.1](../branch-01-math-foundations/notes.md).
- Review numerical precision, tensor shapes, and Python execution in [Branch 1 notes, subtopic 1.3](../branch-01-math-foundations/notes.md).
- Review SGD, momentum, train/validation separation, and bias–variance diagnosis in [Branch 2 notes, subtopic 2.2](../branch-02-classical-ml/notes.md).

You need Python and PyTorch for the code.
Run the CPU snippets in an environment with `torch` installed.
Use your project's dependency workflow; for a standalone scratch environment, run `uv run --with torch python your_script.py`.
Only the optional CUDA branch of the mixed-precision example needs a GPU.
The toy multimodal example downloads no checkpoints and makes no claim about real-image accuracy.

## Hour budget and subtopic map

| Subtopic | Hours | Notes | Assessment |
|---|---:|---|---|
| 3.1 Neural network fundamentals and autograd mechanics | 3 | [Graphs, derivatives, initialization, normalization](notes.md#31-neural-network-fundamentals-and-autograd-mechanics) | A1–A3, B1 |
| 3.2 Optimization dynamics and training stability | 2.5 | [Optimizers, schedules, precision, gradients](notes.md#32-optimization-dynamics-and-training-stability) | A5, B2 |
| 3.3 Computer vision and multimodal baselines | 2.5 | [Convolutions, ViT, cross-attention, CLIP](notes.md#33-computer-vision-and-multimodal-baselines) | A4, A6, B3–B4 |
| **Total** | **8** | **3 + 2.5 + 2.5 = 8** | **6 paper + 4 code problems** |

### Session schedule

The schedule includes reading, selected video segments, and exercises.
Do not add the full video runtimes to the core budget.
The complete videos serve as optional reference material.
Work the short numeric cases before you open their solutions.

| Elapsed time | Duration | Session | Observable result |
|---|---:|---|---|
| 0:00–1:00 | 1 h | 3.1: Graphs, chain rule, hand backprop; A1 | Write all six gradient tensors with their shapes. |
| 1:00–2:00 | 1 h | 3.1: Activations, softmax; A3 and B1 | Match manual gradients to autograd. |
| 2:00–3:00 | 1 h | 3.1: Initialization and normalization; A2 | Derive the He scale and identify normalization axes. |
| 3:00–4:00 | 1 h | 3.2: RMSProp, Adam, AdamW, Lion; A5 | Compute one adaptive update without a library. |
| 4:00–5:00 | 1 h | 3.2: Warmup, cosine, 1Cycle, precision; B2 | Demonstrate overflow and inspect finite gradients. |
| 5:00–5:30 | 0.5 h | 3.2: Residuals, clipping, diagnosis | Order unscale, clip, step, and scaler update. |
| 5:30–6:30 | 1 h | 3.3: Convolutions and ConvNeXt; A4 and A6 | Recompute parameters, operations, and receptive field. |
| 6:30–7:30 | 1 h | 3.3: ViT and cross-attention; B3 | Run a tiny image encoder and inspect token shapes. |
| 7:30–8:00 | 0.5 h | 3.3: CLIP and zero-shot classification; B4 | Compute the contrastive loss and class predictions. |
| **Total** | **8 h** | | |

If you need more practice, repeat a problem outside the core budget.
Prefer recomputing a gradient to passively watching another full lecture.

## Video playlist

These four verified videos belong to Andrej Karpathy's [Neural Networks: Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).
Durations round the public player metadata to the nearest second.
All links and YouTube oEmbed titles pass verification on 2026-10-16.
Choose the indicated segments by topic rather than treating every video as required viewing.

| Title | Channel | Duration | Why it earns study time |
|---|---|---:|---|
| [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) | Andrej Karpathy | 2:25:52 | Follow scalar local derivatives and gradient accumulation before applying the matrix derivation in 3.1. |
| [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) | Andrej Karpathy | 1:57:45 | Compare count-based scores with trainable logits and negative log-likelihood. |
| [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) | Andrej Karpathy | 1:15:39 | Inspect tensor dimensions, minibatches, and learning-rate selection in a small MLP. |
| [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) | Andrej Karpathy | 1:55:24 | Check manual derivatives against autograd, including reduction and normalization steps. |

Within the first three hours, spend at most 20 minutes on selected graph and MLP segments.
Use the remaining videos only when a failed paper exercise exposes a specific gap.

## Embedded playlist

Play the selected videos here when you use the built site.
Use the table links when your Markdown viewer does not display embedded videos.

<iframe width="560" height="315" src="https://www.youtube.com/embed/VMj-3S1tku0" title="The spelled-out intro to neural networks and backpropagation: building micrograd — Andrej Karpathy" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/PaCmpygFfXo" title="The spelled-out intro to language modeling: building makemore — Andrej Karpathy" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/TCH_1BHY58I" title="Building makemore Part 2: MLP — Andrej Karpathy" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/q8SA3rM6ckI" title="Building makemore Part 4: Becoming a Backprop Ninja — Andrej Karpathy" frameborder="0" allowfullscreen></iframe>

## Key sources

The table maps every subtopic to a primary or authoritative teaching source.
Read the named chapter or API topic rather than an entire book.
Every URL below returns an accepted HTTP status during authoring.

| Author or organization | Title | Type | Link | Covers | Access date |
|---|---|---|---|---|---|
| Ian Goodfellow, Yoshua Bengio, Aaron Courville | Deep Learning | Open textbook | [Official book](https://www.deeplearningbook.org/) | 3.1: chapters 6 and 8, chain rule and initialization; 3.2: optimization | accessed 2026-10-16 |
| PyTorch contributors | Automatic differentiation package | Official API docs | [torch.autograd](https://docs.pytorch.org/docs/stable/autograd.html) | 3.1: graph recording, backward, gradient accumulation | accessed 2026-10-16 |
| PyTorch contributors | Automatic mixed precision package | Official API docs | [torch.amp](https://docs.pytorch.org/docs/stable/amp.html) | 3.2: autocast and GradScaler; preferred API over deprecated torch.cuda.amp entry points | accessed 2026-10-16 |
| Aston Zhang, Zachary Lipton, Mu Li, Alex Smola | Dive into Deep Learning | Open textbook | [D2L](https://d2l.ai/) | 3.1–3.3: MLPs, optimization, convolutions, attention | accessed 2026-10-16 |
| Jeremy Howard and fast.ai | Practical Deep Learning for Coders | Course | [Course](https://course.fast.ai/) | 3.2–3.3: practical training and image baselines | accessed 2026-10-16 |
| Stanford CS231n course team | Deep Learning for Computer Vision | University course | [CS231n](https://cs231n.stanford.edu/) | 3.1–3.3: backprop, training, image architectures | accessed 2026-10-16 |
| Ilya Loshchilov, Frank Hutter | Decoupled Weight Decay Regularization | Paper | [AdamW](https://arxiv.org/abs/1711.05101) | 3.2: decoupled decay versus L2 under adaptive scaling | accessed 2026-10-16 |
| Xiangning Chen et al. | Symbolic Discovery of Optimization Algorithms | Paper | [Lion](https://arxiv.org/abs/2302.06675) | 3.2: sign updates and distinct momentum coefficients | accessed 2026-10-16 |
| Noam Shazeer | GLU Variants Improve Transformer | Paper | [Gated activations](https://arxiv.org/abs/2002.05202) | 3.1: SwiGLU and gated feed-forward layers | accessed 2026-10-16 |
| Zhuang Liu et al. | A ConvNet for the 2020s | Paper | [ConvNeXt](https://arxiv.org/abs/2201.03545) | 3.3: modern convolutional architecture design | accessed 2026-10-16 |
| Alexey Dosovitskiy et al. | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | Paper | [ViT](https://arxiv.org/abs/2010.11929) | 3.3: patch tokens, class token, position embeddings | accessed 2026-10-16 |
| Alec Radford et al. | Learning Transferable Visual Models From Natural Language Supervision | Paper | [CLIP](https://arxiv.org/abs/2103.00020) | 3.3: symmetric contrastive training and zero-shot inference | accessed 2026-10-16 |

## Exit criteria

- [ ] Derive the A1 two-neuron backprop result on a blank page.
- [ ] Explain why biases can start at zero while all hidden weights should not.
- [ ] Derive the Xavier and He scales and state their assumptions.
- [ ] Compute the A3 softmax Jacobian and verify its row sums.
- [ ] Contrast BatchNorm training/evaluation with LayerNorm and RMSNorm.
- [ ] Distinguish Adam, AdamW, RMSProp, and Lion from their update equations.
- [ ] Explain why loss scaling cannot repair an already overflowing forward activation.
- [ ] Put gradient unscaling before norm clipping in the training loop.
- [ ] Recompute the convolution operation count with an explicit FLOP convention.
- [ ] Run the MLP and tiny ViT code exercises with the stated shapes.
- [ ] Explain which tower changes at zero-shot inference and which weights stay frozen.
- [ ] Score at least 70% on Part A before opening the solutions and complete all four code exercises.

Continue to [Branch 4 — Transformers and modern NLP](../branch-04-transformers-nlp/index.md) after you meet these criteria.
