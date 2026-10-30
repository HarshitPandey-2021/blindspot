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