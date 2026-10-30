# BlindSpot — 14-Day Calendar

Start: Fri 30 Oct 2026 · Finish: Thu 12 Nov 2026

**Load** tells you how long the day needs: **L** ≈ 1–1.5 hrs · **M** ≈ 2–3 hrs · **H** ≈ 3–4 hrs.

Your week is 3 class days and 4 free days. Put the **H** days on free days and swap things around as needed — the *order* matters, the dates don't. If a day slips, take it out of Day 8, which is deliberately slack.

---

## Week 1 — learn the tools, build the detector

| Day | Date | Load | What |
|---|---|---|---|
| **1** | Fri 30 Oct | M | 3B1B chapters 2–4. Watch a 20-layer network die of vanishing gradients in your own output. |
| **2** | Sat 31 Oct | H | PyTorch 60-min blitz, typed not read. Then a CNN on MNIST — your first convolutional model. |
| **3** | Sun 1 Nov | M | Attach the 140k Real/Fake Faces dataset. Get images loading and displaying. No training yet. |
| **4** | Mon 2 Nov | H | Train the detector for real. First honest accuracy number on faces. |
| **5** | Tue 3 Nov | H | Tune to **≥95%**. Confusion matrix. Look at what it gets wrong. Save weights to `/kaggle/working/`. |
| **6** | Wed 4 Nov | H | Depth sweep: 5/10/20/40 layers, sigmoid. Gradient-per-layer plot for each. Watch them fail. |
| **7** | Thu 5 Nov | H | The four fixes — ReLU, He init, batch norm, skip connections. One plot each. |

**End of week 1 you have:** a working detector, saved weights, and the entire vanishing-gradient section of your report finished as charts.

---

## Week 2 — break it, fix it, present it

| Day | Date | Load | What |
|---|---|---|---|
| **8** | Fri 6 Nov | M | MLP vs CNN parameter counts. Optimiser comparison (SGD / momentum / Adam). **Slack day — use it to catch up if you're behind.** |
| **9** | Sat 7 Nov | H | FGSM from scratch. Attack one fake face. Watch the detector flip to "real". |
| **10** | Sun 8 Nov | H | Epsilon sweep — the headline chart. Side-by-side exhibits. Both directions (fake→real and real→fake). |
| **11** | Mon 9 Nov | M | PGD implemented. FGSM vs PGD at equal ε. |
| **12** | Tue 10 Nov | H | Adversarial training. Train the defended model. |
| **13** | Wed 11 Nov | H | Re-run the full sweep on the defended model. The money chart — both curves, one slide. Gradio demo. |
| **14** | Thu 12 Nov | H | Report, 12 slides, repo cleanup. Send it to me and I grill you on everything. |

---

## Non-negotiables

1. **Every session ends with a commit.** Download the notebook, move it into `notebooks/`, push. A repo with 20 commits over two weeks looks like work.
2. **Every experiment gets a `lab-notebook.md` entry**, written when it finishes, not later. Failures included — those are the ones that earn marks in the viva.
3. **Every chart gets saved as a PNG** to `/kaggle/working/` and downloaded into `results/figures/`. You do not want to re-run anything in week three.
4. **Save model weights the moment training finishes.** Kaggle kills idle sessions after ~20 minutes and your variables go with it. The code survives; the trained model does not.

---

## If you fall behind

Cut in this order — the first things listed are the cheapest to lose:

1. The optimiser comparison (Day 8)
2. The MLP vs CNN comparison (Day 8)
3. PGD (Day 11) — FGSM alone still makes the whole argument
4. The Gradio demo (Day 13) — static before/after images demo almost as well

**Never cut:** the detector, the gradient experiments, FGSM, or adversarial training. Those four are the project.
