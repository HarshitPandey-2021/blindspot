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