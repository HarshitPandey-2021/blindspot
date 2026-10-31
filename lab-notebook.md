# Lab Notebook — BlindSpot

Every experiment gets an entry, written the moment it finishes running.
Date, what changed, what happened, what you think it meant.

**Failures get entries too.** They are worth more marks than the successes,
because nobody else will have recorded theirs.

---

## Template

```
## YYYY-MM-DD
- Ran: <what, with what settings>
- Result: <the numbers>
- Observation: <what you noticed>
- Interpretation: <what you think it means>
- Saved: results/figures/<filename>.png
- Next: <what you'll try because of this>
```

---

## 2026-10-29
- Project scoped and plan finalised. See README.md.
- Nothing run yet.
- Next: Phase 0 — Kaggle account, GitHub repo, 3B1B chapter 1.

---


## 2026-10-29
- Ran: 2-layer MLP (784 -> 256 -> 10), ReLU, Adam lr=1e-3, batch 128, 3 epochs, MNIST.
- Result: 94.39% -> 96.41% -> 96.98% test accuracy.
- Observation: biggest jump was epoch 1->2. Gains flattening by epoch 3.
- Next: Phase 1 done. Break it on purpose to see a vanishing gradient.


## 2026-10-30
- Ran: 20-layer MLP, one batch, one backward pass. Measured mean |gradient| per layer.
- sigmoid: layer 1 = 1.722e-19, layer 20 = 2.325e-02. Ratio 1.35e17 (~8x decay per layer).
- relu:    layer 1 = 4.328e-11, layer 20 = 6.420e-04. Ratio 1.48e7 (~2.4x decay per layer).
- Both lines straight on a log axis => exponential decay => repeated multiplication. Chain rule, visible.
- SURPRISE: ReLU alone did NOT fix it. 1e8 times better than sigmoid, still far too weak to learn.
  Default PyTorch init isn't scaled for 20 layers - the weights shrink the signal too.
- This is why He init / batch norm / skip connections exist. Test each on Day 7.
- Saved: results/figures/grad_profile_20layer.png


## 2026-10-30 (Day 2)
- Ran: MLP vs CNN on MNIST, 3 epochs each, Adam lr=1e-3, batch 128.
- MLP: 97.10% (203,530 params, ~8s/epoch)
- CNN: 98.70% (421,642 params, ~8.5s/epoch)
- CNN cut errors from 290 to 130 out of 10,000. Same compute time.
- CNN has MORE total params, but conv layers are only ~18.8k. The Linear(3136,128) head
  holds the rest. Convolutions do the visual work cheaply; the dense head is the expensive part.
- Next: Day 3, attach the 140k Real/Fake Faces dataset.


## 2026-10-30 (Day 3)
- Attached 140k Real/Fake Faces (xhlulu), mounted under /kaggle/input/datasets/xhlulu/...
- ImageFolder: train 100,000 / valid 20,000 / test 20,000. Balanced. fake=0, real=1.
- 128x128 RGB, batch shape [128,3,128,128].
- Viewed samples: could not reliably tell fake from real by eye.
- NOTE: real images look more varied (candid, odd angles); fakes are all centred portraits.
  Possible shortcut learning - model might key on framing, not AI artifacts. Check on Day 5
  by inspecting misclassified images.
- Next: Day 4, train the detector.
## 2026-10-31 (Day 4) - detector trained
- v1: 20k @96px, lr 1e-3, 10 ep -> valid 0.9137. Valid swung +/-30pts while train climbed smoothly.
- Diagnosis: lr too high, model oscillating. Train acc is averaged across the epoch so it
  hides this; valid is a single snapshot at epoch end.
- v2: 20k @96px, lr 3e-4 + cosine + hflip, 25 ep -> valid 0.9050.
  CONFIRMED: epochs 1-12 (lr>1.4e-4) still swung; 13-25 climbed smoothly.
  Stability fixed, peak unchanged => optimisation was NOT the bottleneck.
- v3: 45k @128px, same settings -> best valid 0.9522 (epoch 25), train 0.9912.
  => data quantity + resolution were the real limit. StyleGAN artifacts are high-frequency
  and 96px was destroying them.
- Overfit gap down to 3.9pts (was 7pt at 20k images).
- Earlier preload fix: epochs went from ~16 min (disk-bound) to 32s. /kaggle/input is
  network-mounted and the dataset is 140k small JPEGs - ~31 imgs/sec.
- Lost both .pth files to a session timeout (persistence was off). Re-ran. Persistence now on.
  Rule: download the .pth the moment training ends.
- Saved: models/detector_v3.pth, results/figures/training_curve_v3.png
- Test set NOT touched. Reserved for the final measurement.
- Next: Day 5 - confusion matrix, inspect errors, test the shortcut-learning worry from Day 3.



## 2026-10-31 (Day 5) - analysis
- Confusion matrix on 6,000 validation images (overall 0.9522):
    is fake -> said fake 2,809 | said real   163   (94.52% of fakes caught)
    is real -> said fake   124 | said real 2,904   (95.90% of reals passed)
  Slightly weaker at catching fakes than clearing reals. The two errors are not equally
  costly: 163 fakes through = verification bypass; 124 reals flagged = innocent users rejected.
- 287 wrong out of 6,000. Confident mistakes cluster on OCCLUSIONS and ATYPICAL FACES:
  sunglasses, glasses, hats, wide-open mouth, a young child.
- SHORTCUT-LEARNING TEST (following the Day 3 worry that fakes are all tidy centred portraits):
    original         0.9522
    blurred          0.5085   <- collapses to chance
    border removed   0.8553   <- mostly holds
  Blur removes fine texture but preserves layout. Accuracy collapsed to random =>
  the model is reading HIGH-FREQUENCY TEXTURE, not composition. Day 3 worry refuted.
  Border removal cost ~10pts => background contributes some signal, but the face carries most.
- PREDICTION FOR DAY 9 (made today, before building the attack):
  FGSM produces a small high-frequency perturbation. This detector depends on high-frequency
  texture. So it should be unusually EASY to fool - accuracy should collapse at very small
  epsilon. To be tested.
- IMPLIED LIMITATION: JPEG compression also removes high frequencies, so this detector would
  likely degrade badly on re-compressed images. Worth testing if time allows.
- Test set still untouched.


## 2026-10-31 (Day 6) - vanishing gradient, depth sweep
- Gradient profile at init, sigmoid MLPs on BlindSpot data (one backward pass each):
     5 layers: layer1 1.520e-06, last 3.924e-02, ratio 2.6e4
    10 layers: layer1 3.737e-11, last 2.806e-02, ratio 7.5e8
    20 layers: layer1 1.788e-19, last 4.122e-02, ratio 2.3e20
    40 layers: layer1 3.659e-37, last 1.386e-03, ratio 3.8e39
  Doubling depth SQUARES the ratio -> consistent with repeated multiplication (chain rule).
  All four lines straight on a log axis => exponential decay. Saved: grad_depth_sweep.png

- FAILED FIRST ATTEMPT (kept deliberately): trained all four depths, every one returned
  exactly 0.5047 = 3028/6000 = "always predict real". All collapsed including 5 layers.
  With no positive control the result was uninterpretable: could not separate "depth killed
  it" from "setup broken".
  Cause: 49,152 raw inputs into a sigmoid layer saturates it even at 5 layers; inputs were
  not zero-centred.
  Fix: 32x32 input (3,072 features), centred to [-1,1], and ALWAYS run a control first.
  Also hit a CUDA OOM converting 45k images to float in one go - fixed by chunking.

- SECOND ATTEMPT (8 epochs, plain SGD lr=0.1, 32x32 centred):
    CONTROL relu  5 layers: 0.7122   <- control passes, setup is sound
    sigmoid 5/10/20/40     : 0.5047  (all = always-predict-real)
    relu   10/20/40        : 0.4953  (all = always-predict-fake, 2972/6000)
  Findings:
   1. Sigmoid fails at EVERY depth, even 5. Predicted by the gradient profile (layer-1
      grad 1.5e-06 x lr 0.1 = negligible updates).
   2. ReLU rescues depth 5 only. ReLU is necessary but NOT sufficient.
   3. Dead networks collapsed to constant predictors, not weak learners (checked).
  Caveat: 8 epochs of plain SGD is a tight budget; identical for every configuration, so
  the comparison is fair, but longer training might change absolute numbers.

- Baseline for Day 7: the wall is at depth 5 with ReLU alone. Question: how much deeper do
  He init / batch norm / skip connections push it?


## 2026-10-31 (Day 7) - the four fixes
- Gradient profile, 20-layer networks, one backward pass each:
    sigmoid             layer1 3.617e-19   last 1.514e-01   ratio 4.19e17
    relu                layer1 5.945e-11   last 5.811e-04   ratio 9.77e6
    relu + He init      layer1 2.840e-03   last 5.631e-02   ratio 19.8
    relu + He + BN      layer1 1.060e-01   last 8.394e-02   ratio 0.8    <- near-perfect flow
    residual + He       layer1 2.698e+01   last 2.582e+03   ratio 95.7   <- magnitudes ~1000x TOO LARGE
    residual + BN + He  layer1 2.971e-02   last 4.004e+00   ratio 134.8  <- BN tamed it (~900x down)

- Accuracy (8 epochs, plain SGD lr=0.1, 32x32 centred input):
    config              d=5      d=10     d=20     d=40
    sigmoid             0.5047   0.5047   0.5047   0.5047
    relu                0.7122   0.4953   0.4953   0.4953
    relu + He           0.7943   0.6552   0.7840   0.7692
    relu + He + BN      0.8173   0.7972   0.7363   0.5287
    residual + He       0.7732   0.4953   0.4953   0.4953
    residual + BN + He  0.7760   0.7713   0.5995   0.5108

- FINDINGS:
  1. He init is the single biggest win: moves the depth wall from 5 to beyond 40.
  2. BN gives the best shallow accuracy (0.8173 @ d=5) and the best gradient ratio (0.8),
     but degrades with depth and collapses at 40.
  3. EXPLODING GRADIENTS, unplanned: residual-only gradients were ~1000x too LARGE.
     y = x + f(x) accumulates signal through the stack, so with lr=0.1 every update
     overshoots -> dead at depth 10+. Both failure modes from the syllabus now demonstrated.
  4. Adding BN to the residual net cut gradient magnitudes ~900x and rescued depth 10
     (0.4953 -> 0.7713). This is exactly why real ResNets pair skips WITH normalisation.
  5. OVERALL WINNER AT DEPTH IS THE SIMPLEST FIX: relu + He init alone, 0.7692 at 40 layers,
     beating BN, residual, and residual+BN. My prediction that residual+BN would win was WRONG.
  6. Healthy gradient FLOW is necessary but not sufficient - residual-only had a reasonable
     ratio (95.7) and still failed, because the magnitudes were wrong.

- CAVEATS: single seed per config; lr fixed at 0.1 for every architecture (never tuned
  per-config); 8 epochs may be too few for the larger residual+BN network. The He-init dip
  at d=10 then recovery at d=20 is probably run-to-run noise.

- Saved: results/figures/grad_fixes_20layer.png, results/figures/depth_wall_by_fix.png