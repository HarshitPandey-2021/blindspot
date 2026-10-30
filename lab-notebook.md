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