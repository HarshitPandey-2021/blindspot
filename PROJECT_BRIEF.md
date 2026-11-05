# BlindSpot — Full Project Brief

**Days 1–7 complete.** Everything we did, why, and every number.

Deep Learning (NAI-701) · Solo semester project · Harshit Pandey · University of Lucknow
Last updated: 31 October 2026

---

## 1. What the project is

A deep learning semester project, graded by an in-depth technical viva with the professor.

**In one line:** build a program that tells real photographs apart from AI-generated faces, then break it with changes invisible to the human eye, then fix it.

**Build it. Break it. Fix it.**

1. Train a CNN to classify face photos as "real photograph" or "AI-generated (StyleGAN)". Currently at **95.22%**.
2. Attack it with FGSM — a tiny perturbation, invisible to a human, that makes the detector confidently wrong.
3. Defend it with adversarial training, and measure what the defence costs.

---

## 2. Why this project

### The practical application

The question every examiner asks. Four answers, all real deployed systems:

- **Content moderation** — platforms run detectors like this to flag AI-generated images. If invisible noise defeats them, the filter is theatre.
- **KYC / identity verification** — banks check that a submitted selfie is a real person. A generated face plus noise could pass.
- **The reverse harm** — a *real* person's photo pushed over the line into "fake", so a genuine user is wrongly rejected. Rarely tested. This project tests it.
- **The general lesson** — this isn't one broken program. Every neural network trained the normal way has this weakness.

### Why it's different from a typical student project

Most student projects build a classifier that already exists — cats vs dogs, digit recognition, sentiment analysis. The model is the deliverable, the code came from a tutorial, and the question *"what did you contribute?"* has no good answer.

This one builds a working system and then **attacks it and measures where it breaks**. That's the shape of research, not the shape of a tutorial.

### The structural reason the viva is safe

FGSM computes the gradient of the loss with respect to the **input pixels** instead of the weights. It is backpropagation run for a different purpose, stopped one step earlier.

So every gradient question the examiner asks is a question about this project's own code. There is no gap between the theory and the build for him to find.

---

## 3. Setup

| Thing | Detail |
|---|---|
| Compute | Kaggle Notebooks, free T4 GPU, 30 hrs/week. The laptop only runs a browser. |
| Dataset | "140k Real and Fake Faces" (xhlulu, Kaggle). 70,000 real photographs (Flickr/FFHQ) + 70,000 StyleGAN-generated. |
| Splits | train 100,000 / valid 20,000 / test 20,000 |
| Labels | **fake = 0, real = 1** |
| Repo | `github.com/HarshitPandey-2021/blindspot`, local at `C:/dev/blindspot` |
| Discipline | Validation set for every decision. **Test set untouched** until the very end. |

---

# 4. Day by day

## Day 1 — what a network is, and the problem with deep ones

**Built:** a simple MLP on MNIST handwritten digits. 784 inputs → 256 hidden → 10 outputs.

**Result:** 94.39% → 96.41% → **96.98%** over three epochs, on digits it had never seen.

**Then the real experiment.** A deliberately bad 20-layer network, measuring how strong the learning signal was at each layer after one backward pass:

```
sigmoid    layer 1: 1.722e-19   last layer: 2.325e-02   ratio 1.35e17
relu       layer 1: 4.328e-11   last layer: 6.420e-04   ratio 1.48e7
```

The first layer received an instruction **135 quadrillion times weaker** than the last layer. That is the **vanishing gradient problem**, measured rather than read about.

**Finding:** ReLU improved layer 1 by a factor of 100 million — but `4.3e-11` is still far too weak to learn from. **ReLU alone does not fix it.**

---

## Day 2 — the right shape for images

Same digit task, two network shapes, same data and training time:

```
flat network (MLP)   97.10%    290 errors out of 10,000    203,530 params
CNN                  98.70%    130 errors out of 10,000    421,642 params
```

Errors more than halved.

**Finding with a twist:** the CNN has *more* total parameters, but only ~18,800 are in the convolutional layers. Those 18,800 beat a 200,000-parameter flat layer. The expensive part of a CNN is the dense head at the end — which is why Day 4's detector replaced it with global average pooling.

---

## Day 3 — the real data

Loaded the faces. **No training that day, deliberately** — if you load data wrong *and* train at the same time, you cannot tell which one broke.

```
train 100,000   valid 20,000   test 20,000
fake = 0, real = 1
tensor shape [3, 128, 128]   (vs MNIST's [1, 28, 28])
```

**Observation that mattered later:** the real photos are varied and messy — odd angles, cluttered backgrounds, candid shots. The fakes are all tidy centred portraits.

**Risk flagged:** the model might learn *"tidy framing = fake"* and score well for entirely the wrong reason. Marked for testing on Day 5.

---

## Day 4 — training the detector

### The first problem: 16 minutes per epoch

Profiled it — the GPU was idle. The dataset is 140,000 small files on a network-mounted drive, and reading them ran at ~31 images/second. **Disk-bound, not compute-bound.**

**Fix:** load a chunk into GPU memory once as `uint8`, then train from there. **Epochs went from 16 minutes to 8 seconds.**

### Three experiments

```
v1   20,000 images @ 96px,  lr 1e-3                          91.37%
v2   20,000 images @ 96px,  lr 3e-4 + cosine decay + flips   90.50%
v3   45,000 images @ 128px, same optimiser settings          95.22%
```

**v1** worked but validation accuracy swung wildly between epochs — 0.70, 0.59, 0.86, 0.53 — while training accuracy climbed smoothly.

*Diagnosis:* learning rate too high, model oscillating. Training accuracy is averaged *across* an epoch so it hides this; validation is a single snapshot at epoch end.

**v2** tested that diagnosis. The swinging stopped exactly when cosine decay pulled the learning rate below ~1.4e-4 — epochs 13–25 climbed smoothly from 0.867 to 0.905. **Diagnosis confirmed.** But the peak accuracy did not improve, so **optimisation was not the bottleneck.**

**v3** changed the data instead: more images, higher resolution. **+4 points.** StyleGAN artifacts are fine, high-frequency detail, and 96px was destroying them.

### The final detector

Four convolutional blocks — `Conv2d + BatchNorm + ReLU + MaxPool` — going 3→32→64→128→128 channels while the image halves each time (128→64→32→16→8). Then global average pooling, then `Linear(128, 2)`.

**241,794 parameters. Train 0.9912 / valid 0.9522.**

### Operational lesson

Lost both `.pth` files to a Kaggle session timeout because persistence was off. Had to retrain. **Rule since: download the `.pth` the moment training ends.**

---

## Day 5 — interrogating the detector

### Confusion matrix (6,000 validation images, overall 0.9522)

```
           said fake   said real
is fake      2,809        163      → 94.52% of fakes caught
is real        124      2,904      → 95.90% of reals passed
```

Slightly weaker at catching fakes than at clearing real people.

**The two errors are not equally costly.** 163 fakes through = verification bypass. 124 reals flagged = innocent users rejected. Which is worse depends entirely on what the system protects.

### Where it fails

287 wrong out of 6,000. The confident mistakes cluster on **occlusions and atypical faces** — sunglasses, glasses, hats, a wide-open mouth, a young child. Exactly where the training data is thinnest.

### The shortcut-learning test

Answering the Day 3 worry:

```
original          0.9522
blurred           0.5085    ← collapsed to a coin flip
border removed    0.8553    ← mostly held up
```

Blur destroys fine texture but leaves composition and framing perfectly intact. **If the model had been keying on "tidy centred portrait", blur would barely have touched it.** It collapsed to random instead.

**Conclusion: the model reads high-frequency texture, which is where StyleGAN leaves its fingerprints. Day 3 worry refuted.** Border removal cost ~10 points, so background contributes something, but the face carries most of the signal.

### A prediction, recorded in advance

> FGSM produces a small high-frequency perturbation. This detector depends on high-frequency texture. **So it should be unusually easy to fool — accuracy should collapse at very small epsilon.**

To be tested on Day 9. A prediction made on Day 5 and confirmed on Day 9 is what turns a project into a piece of research.

### Implied limitation

JPEG compression also removes high frequencies, so this detector would likely degrade badly on re-compressed images. Worth testing if time allows.

---

## Day 6 — making the vanishing gradient happen, properly

### Gradient profile at initialisation

Sigmoid MLPs on the face data, one backward pass each:

```
 5 layers:  layer1 1.520e-06   last 3.924e-02   ratio 2.6e4
10 layers:  layer1 3.737e-11   last 2.806e-02   ratio 7.5e8
20 layers:  layer1 1.788e-19   last 4.122e-02   ratio 2.3e20
40 layers:  layer1 3.659e-37   last 1.386e-03   ratio 3.8e39
```

**Doubling depth squares the ratio** — consistent with repeated multiplication, which is exactly what the chain rule does.

All four lines are **straight on a log axis**, and that straightness *is* the proof of exponential decay. If the cause were something other than repeated multiplication, the lines would curve. Worth saying out loud in the viva.

### A failed experiment, kept deliberately

The first attempt at training all four depths returned exactly **0.5047** for every one. That number is 3,028/6,000 — the "always predict real" baseline. Every network collapsed, including the 5-layer one.

**With no positive control the result was uninterpretable** — impossible to separate *"depth killed it"* from *"the setup is broken."*

*Causes:* 49,152 raw inputs into a sigmoid layer saturates it even at 5 layers, and the inputs were not zero-centred. Also hit a CUDA out-of-memory error converting 45k images to float in one go — fixed by chunking.

*Fix:* 32×32 input (3,072 features), centred to [−1, 1], and **always run a control that must succeed.**

### Second attempt (8 epochs, plain SGD lr=0.1)

```
CONTROL  relu  5 layers: 0.7122   ← control passes, setup is sound
sigmoid  5/10/20/40     : 0.5047  (all = always-predict-real)
relu    10/20/40        : 0.4953  (all = always-predict-fake, 2972/6000)
```

**Findings:**

1. Sigmoid fails at *every* depth, even 5 — predicted in advance by the gradient profile (layer-1 gradient 1.5e-06 × lr 0.1 = negligible updates).
2. ReLU rescues depth 5 only. **Necessary but not sufficient.**
3. The dead networks collapsed to **constant predictors**, not weak learners. Checked, not assumed.

### Why plain SGD and not Adam

Adam adapts the step size per weight based on gradient history, so it *compensates* for tiny gradients and **partially masks** the phenomenon. Plain SGD multiplies the gradient by the learning rate and nothing else, so a vanished gradient means a vanished update, visibly.

Expect to be asked this. That paragraph is the answer.

---

## Day 7 — the four fixes

### Gradient flow, 20-layer networks

```
sigmoid             layer1 3.617e-19   last 1.514e-01   ratio 4.19e17
relu                layer1 5.945e-11   last 5.811e-04   ratio 9.77e6
relu + He init      layer1 2.840e-03   last 5.631e-02   ratio 19.8
relu + He + BN      layer1 1.060e-01   last 8.394e-02   ratio 0.8     ← near-perfect flow
residual + He       layer1 2.698e+01   last 2.582e+03   ratio 95.7    ← magnitudes ~1000x TOO LARGE
residual + BN + He  layer1 2.971e-02   last 4.004e+00   ratio 134.8   ← BN tamed it (~900x down)
```

### Accuracy

```
config              d=5      d=10     d=20     d=40
sigmoid             0.5047   0.5047   0.5047   0.5047
relu                0.7122   0.4953   0.4953   0.4953
relu + He           0.7943   0.6552   0.7840   0.7692
relu + He + BN      0.8173   0.7972   0.7363   0.5287
residual + He       0.7732   0.4953   0.4953   0.4953
residual + BN + He  0.7760   0.7713   0.5995   0.5108
```

### What each fix does

| Fix | What it attacks |
|---|---|
| **He initialisation** | Scales starting weights so they stop shrinking the signal. ReLU fixed the activation's contribution; this fixes the weights'. |
| **Batch normalization** | Re-centres each layer's numbers so they stay in a healthy range instead of drifting. |
| **Residual / skip** | `y = x + ReLU(Wx)`. At an addition node the gradient passes through **unchanged**, bypassing the multiplication chain. This is ResNet in four lines. |

### Findings

1. **He initialisation is the biggest single win** — moves the depth wall from 5 to beyond 40.
2. **Batch norm gives the best shallow accuracy** (0.8173 at d=5) and the best gradient ratio (0.8), but degrades with depth and collapses at 40.
3. **Exploding gradients, unplanned.** Residual-only gradients were ~1000× too *large*. `y = x + f(x)` accumulates signal through the stack, so at lr=0.1 every update overshoots — dead at depth 10+. **Both failure modes from the syllabus now demonstrated from own runs.**
4. **Adding BN to the residual net** cut magnitudes ~900× and rescued depth 10 (0.4953 → 0.7713). This is exactly why real ResNets pair skip connections *with* normalisation.
5. **The overall winner at depth is the simplest fix** — ReLU + He init alone, 0.7692 at 40 layers, beating BN, residual, and residual+BN. **The prediction that residual+BN would win was wrong.**
6. **Healthy gradient flow is necessary but not sufficient** — residual-only had a reasonable ratio (95.7) and still failed, because the magnitudes were wrong.

### Caveats (state these before he does)

Single seed per configuration. Learning rate fixed at 0.1 for every architecture, never tuned per-config. 8 epochs may be too few for the larger residual+BN network. The He-init dip at d=10 then recovery at d=20 is probably run-to-run noise.

---

# 5. The concepts, in plain words

**A neural network is a stack of dials.** Picture in one end, answer out the other. Training means finding the right position for every dial. A "dial" is properly called a **weight**.

**Loss is a golf score.** One number saying how wrong the guess was. Lower is better, zero is perfect.

**A gradient tells one dial which way to turn** to make the loss smaller. **Backpropagation** computes the gradient for every dial at once.

**Backpropagation is blame passed backwards down a factory line.** The inspector at the end finds a defect, tells worker 20, who works out how much came from worker 19 and passes that share back, and so on down the line.

**Vanishing gradient is the whisper game.** Twenty people in a line, each passing on a quarter of the volume. Person 1 hears silence and never changes anything. Mathematically: backprop multiplies by each layer's derivative on the way back, and sigmoid's derivative is at most 0.25, so twenty multiplications shrink the signal to nothing.

**A CNN slides one small filter across the whole image** instead of learning each position separately. Learn "edge" once, use it everywhere.

**The attack is backpropagation pointed at the pixels instead of the dials.**

### Vocabulary

| Term | One line |
|---|---|
| Tensor | An array of numbers that tracks its own history so gradients can be computed |
| Weight | A number inside the model, adjusted during training |
| Loss | A single number measuring how wrong the model is right now |
| Gradient | Which direction to nudge something to reduce the loss |
| Backpropagation | The efficient algorithm for computing gradients for every weight at once |
| Epoch | One full pass through the training data |
| Batch | A small group processed together before one weight update |
| Learning rate | How big each nudge is |
| Overfitting | Memorising the training data instead of learning the pattern |
| Epsilon (ε) | How much each pixel may be changed in the attack |
| FGSM | One-step attack — nudge pixels once, in the direction that hurts most |
| PGD | Multi-step FGSM. Slower, much stronger, harder to defend against |
| White-box | You have the model's weights, so you can compute its gradients. Ours. |
| Black-box | You can only query the model, not see inside it. Harder. |

### The three lines that matter

```python
opt.zero_grad()   # wipe old gradients — PyTorch ADDS, it doesn't replace
loss.backward()   # backpropagation — gradient for every weight
opt.step()        # turn every dial slightly
```

Expect to be asked what happens if you remove `zero_grad()`. Answer: gradients accumulate across batches, so every update after the first is based on a sum of stale gradients, and training degrades or diverges.

---

# 6. What's left

| Day | Work |
|---|---|
| **8** | Slack / catch-up day |
| **9–10** | **The attack.** FGSM from scratch. Epsilon sweep (the headline chart). The two identical-looking faces with different verdicts. Both directions: fake→real and real→fake. Then PGD. |
| **11–12** | Adversarial training as the defence, and the clean-accuracy cost measured honestly |
| **13** | Defended vs undefended curves on one chart; Gradio demo |
| **14** | Report, slides, viva drilling |

### Demo day plan

**Never train live in front of the examiner.** Demo inference only — load the saved `.pth`, run on an image, under a second, works on any machine.

Three layers of backup:

1. **Slides with every chart as a PNG** — if every piece of technology in the room fails, the whole presentation still works. Build this first.
2. **Pendrive with the repo plus the `.pth` files.** They are gitignored, so the pendrive and a cloud copy are their only homes.
3. **The live Gradio demo.** Check whether the lab has internet a few days before, not on the day.

---

# 7. Strongest viva answers available so far

- *"My first training run was 120× too slow. I profiled it, found it was disk-bound rather than compute-bound, and restructured the pipeline so the data lives in GPU memory."*
- *"I thought my accuracy ceiling was the optimiser. I tested that, proved it wasn't, and traced it to data resolution instead."*
- *"My first depth experiment had no positive control, so the result was uninterpretable. I added one and re-ran."*
- *"My detector depends on high-frequency texture — I proved it by blurring the inputs and watching accuracy collapse to chance, which also refuted my earlier worry about shortcut learning."*
- *"I demonstrated both vanishing and exploding gradients from my own runs — the exploding case came from residual connections without normalisation."*
- *"The simplest fix, He initialisation, beat every more sophisticated one at depth. I predicted the opposite and was wrong."*
- *"I know transfer learning would score higher. I trained from scratch because the project is about the mechanism, not the leaderboard."*
- *"Accuracy isn't one number — 163 fakes slipped through and 124 real people were wrongly flagged, and those are different harms."*

---

# 8. Files in the repo

```
blindspot/
├── README.md                      the project plan
├── PLAN.md                        14-day dated calendar
├── FOUNDATIONS.md                 concepts from zero, every line of code explained
├── PROJECT_BRIEF.md               this file
├── lab-notebook.md                every experiment, every result, every failure
├── notebooks/
│   ├── 01_first_model.ipynb       MNIST MLP + first vanishing gradient measurement
│   ├── 02_cnn_mnist.ipynb         MLP vs CNN
│   ├── 03_faces_data.ipynb        dataset loading and inspection
│   └── 04-detector.ipynb          detector training, analysis, gradient experiments
├── models/
│   ├── detector_v3.pth            THE model — 95.22%, 128px input
│   └── detector_v1_96px_OLD.pth   superseded, 91.37%, 96px input — do not mix them up
└── results/figures/
    ├── sample_faces.png
    ├── training_curve_v3.png
    ├── confusion_matrix.png
    ├── confident_mistakes.png
    ├── grad_profile_20layer.png
    ├── grad_depth_sweep.png
    ├── accuracy_by_depth.png
    ├── grad_fixes_20layer.png
    └── depth_wall_by_fix.png
```

Nine figures. Seven days. Every number in this document came from a run on this project's own data.
