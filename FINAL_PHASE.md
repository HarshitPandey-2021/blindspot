# BlindSpot — Final Phase

Diagnostic cells, the lab-notebook entry, corrected viva answers, and the plan to the finish.

Written 5 November 2026, after Days 9–12 completed.

---

## Where the project stands

| Phase | Status |
|---|---|
| Days 1–3 | Foundations, CNN vs MLP, dataset loaded | ✅ |
| Days 4–5 | Detector trained (95.22%), error analysis, shortcut test | ✅ |
| Days 6–7 | Vanishing + exploding gradients, six configurations | ✅ |
| Days 9–10 | FGSM, PGD, epsilon sweep, both directions | ✅ |
| Days 11–12 | Adversarial training, defended evaluation | ✅ |
| **Day 13** | **Diagnostics (this file) + Gradio demo** | ← you are here |
| **Day 14** | Report + slides | |
| **Day 15** | Viva drilling | |

---

# PART 1 — The diagnostic cells

Run these in `05_attack`. They answer the open questions in your results and each one
produces a line for the report.

**Prerequisite:** `detector`, `defended`, `Xva`, `Yva`, `Xtr`, `Ytr`, `accuracy`,
`accuracy_under`, `fgsm`, `pgd` must all be in memory. If the session died, re-run
Cells 1–12 from `DAYS_9_TO_12.md` first.

---

## D1 — BatchNorm recalibration (~40 seconds)

**Tests:** whether mixed clean+adversarial batches corrupted the BatchNorm running
statistics, which is one candidate for the 24.5-point clean accuracy drop.

```python
import copy

m = copy.deepcopy(defended)
for mod in m.modules():
    if isinstance(mod, nn.BatchNorm2d):
        mod.reset_running_stats()
        mod.momentum = None            # cumulative average, not exponential

m.train()                              # train mode so BN updates its running stats
with torch.no_grad():                  # but no weight updates
    for i in range(0, Xtr.size(0), 256):
        m(Xtr[i:i+256].float() / 255.)
m.eval()

a_before = accuracy(defended, Xva, Yva)
a_after  = accuracy(m,        Xva, Yva)
print(f"defended clean, BN stats from mixed batches : {a_before:.4f}")
print(f"defended clean, BN stats re-estimated clean : {a_after:.4f}")
print(f"recovered by recalibration: {a_after - a_before:+.4f}")

torch.save(m.state_dict(), "/kaggle/working/detector_defended_bnfix.pth")
```

**How to read it:**
- Recovers **more than ~3 points** → mixed-batch BN was a real contributor. Report it, and
  use this recalibrated model for the demo.
- Recovers **under ~1 point** → BN wasn't the problem. The cost is real, caused by
  something else.

Either way you have a measured answer instead of a guess.

---

## D2 — Is it gradient masking? (~4 minutes)

**Tests:** whether the defended model's robustness is genuine or just hiding its gradients.

Your existing numbers already pass three of the standard checks:
1. PGD ≤ FGSM at every epsilon ✓ (masking often makes PGD *worse* than FGSM)
2. Accuracy decreases monotonically with epsilon ✓
3. Accuracy → ~0 at large epsilon, no floor ✓

This cell runs the two remaining ones.

```python
# --- Check 1: transfer attack ---
# Craft adversarial examples on the UNDEFENDED model, test them on the DEFENDED one.
def transfer_acc(src, tgt, X, Y, eps, attack, bs=256):
    c = 0
    for i in range(0, X.size(0), bs):
        x = X[i:i+bs].float() / 255.; y = Y[i:i+bs]
        xa = attack(src, x, y, eps)
        with torch.no_grad():
            c += (tgt(xa).argmax(1) == y).sum().item()
    return c / X.size(0)

print("eps      defended white-box PGD    transfer from undefended")
for e in [0.005, 0.01, 0.02]:
    wb = accuracy_under(defended, Xva, Yva, e, pgd)
    tr = transfer_acc(detector, defended, Xva, Yva, e, pgd)
    print(f"{e:<8.3f} {wb:.4f}                   {tr:.4f}")
```

**How to read it:** transfer accuracy should be **higher** than white-box. An attack built
without seeing the defended model should be weaker than one built against it. If transfer
is *lower*, the white-box gradients are misleading the attacker — that is gradient masking,
and your robustness numbers would be inflated.

```python
# --- Check 2: a much stronger PGD ---
def pgd_strong(model, x, y, eps, steps=40, restarts=2):
    best, best_wrong = None, None
    for _ in range(restarts):
        xa = pgd(model, x, y, eps, steps=steps)
        with torch.no_grad():
            wrong = model(xa).argmax(1) != y
        if best is None:
            best, best_wrong = xa.clone(), wrong.clone()
        else:
            take = wrong & (~best_wrong)
            best[take] = xa[take]
            best_wrong |= wrong
    return best

print("\neps      PGD-10       PGD-40 x2 restarts")
for e in [0.005, 0.01]:
    a10 = accuracy_under(defended, Xva, Yva, e, pgd)
    a40 = accuracy_under(defended, Xva, Yva, e,
                         lambda m, x, y, ep: pgd_strong(m, x, y, ep, 40, 2))
    print(f"{e:<8.3f} {a10:.4f}       {a40:.4f}")
```

**How to read it:** a small drop (a few points) is normal. A **large** drop means your
10-step PGD number was optimistic and the stronger attack is the honest one — report
the stronger number.

---

## D3 — The high-frequency hypothesis (~1 minute)

**Tests:** your hypothesis that adversarial training cost so much clean accuracy because it
pushed the model off the high-frequency texture features the detector depends on.

On Day 5 you measured the undefended model: blur took it from 0.9522 to 0.5085 — it lost
**47% of its accuracy** when fine texture was removed. If the defended model loses much
less, it has shifted onto coarser features, which supports your hypothesis.

```python
import torchvision.transforms.functional as TF

def acc_blur(model, X, Y, sigma, bs=256):
    if sigma == 0:
        return accuracy(model, X, Y)
    model.eval(); c = 0
    with torch.no_grad():
        for i in range(0, X.size(0), bs):
            x = TF.gaussian_blur(X[i:i+bs].float()/255., kernel_size=9, sigma=sigma)
            c += (model(x).argmax(1) == Y[i:i+bs]).sum().item()
    return c / X.size(0)

sigmas = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
und, defn = [], []
print("sigma    undefended   defended")
for s in sigmas:
    a, b = acc_blur(detector, Xva, Yva, s), acc_blur(defended, Xva, Yva, s)
    und.append(a); defn.append(b)
    print(f"{s:<8} {a:.4f}       {b:.4f}")

print(f"\nfraction of accuracy lost at sigma=2.0:")
print(f"  undefended: {(und[0]-und[4])/und[0]:.3f}")
print(f"  defended  : {(defn[0]-defn[4])/defn[0]:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(sigmas, und,  marker="o", lw=2, label="undefended")
plt.plot(sigmas, defn, marker="s", lw=2, label="defended")
plt.axhline(0.5047, color="crimson", ls="--", label="chance")
plt.xlabel("Gaussian blur sigma  (more blur = less high-frequency detail)")
plt.ylabel("clean accuracy")
plt.title("Which model depends more on fine texture?")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("/kaggle/working/frequency_reliance.png", dpi=150, bbox_inches="tight")
plt.show()
```

**How to read it:** if the defended model loses a **smaller fraction** of its accuracy under
blur, your hypothesis is supported — adversarial training moved it onto lower-frequency
features, and that migration is what the 24.5 points bought.

This turns an untested hypothesis in your report into a measured finding. Worth the minute.

---

## D4 — Does the attack survive being saved as a real image? (~2 minutes)

**Tests:** your own observation that eps below 1/255 can't exist in an 8-bit file.

```python
def acc_quantized(model, X, Y, eps, attack, bs=256):
    """Attack, then round to 8-bit as if saved as PNG and reloaded."""
    c = 0
    for i in range(0, X.size(0), bs):
        x = X[i:i+bs].float()/255.; y = Y[i:i+bs]
        xa = attack(model, x, y, eps)
        xq = (xa * 255).round().clamp(0, 255) / 255.
        with torch.no_grad():
            c += (model(xq).argmax(1) == y).sum().item()
    return c / X.size(0)

print("eps      (x/255)   float    8-bit")
for e in [0.001, 0.002, 0.005, 0.01, 0.02]:
    f = accuracy_under(detector, Xva, Yva, e, fgsm)
    q = acc_quantized(detector, Xva, Yva, e, fgsm)
    print(f"{e:<8.3f} ({e*255:4.1f})    {f:.4f}   {q:.4f}")
```

**How to read it:** the epsilons where the 8-bit column still shows near-zero accuracy are
your **real-world** results. The ones that recover under quantization were float-space
artifacts. This gives you a defensible headline number and pre-empts the question
"would this work on an actual saved image?"

---

## D5 — Retrain at a smaller epsilon (~20 minutes) — optional

**Tests:** whether eps=0.01 was simply too aggressive a training target for a model that
already collapses completely at 0.002.

```python
defended_small = make_detector()
defended_small.load_state_dict(torch.load(MODEL_PATH, map_location=device))
best_small = train_adversarial(defended_small, Xtr, Ytr, Xva, Yva,
                               epochs=12, eps=0.003,
                               save_to="/kaggle/working/detector_defended_eps003.pth")

print(f"\nclean accuracy:")
print(f"  undefended            0.9522")
print(f"  defended (eps=0.010)  0.7072")
print(f"  defended (eps=0.003)  {accuracy(defended_small, Xva, Yva):.4f}")

print("\neps      defended(0.010) PGD   defended(0.003) PGD")
for e in [0.002, 0.005, 0.01]:
    a1 = accuracy_under(defended,       Xva, Yva, e, pgd)
    a2 = accuracy_under(defended_small, Xva, Yva, e, pgd)
    print(f"{e:<8.3f} {a1:.4f}                {a2:.4f}")
```

**How to read it:** if clean accuracy recovers a lot (say to 0.85+) while robustness at
small epsilon stays similar, then eps=0.01 was the problem and you have a clean
**robustness/accuracy tradeoff curve** — two defended models at two training budgets. That
is a genuinely strong extra result.

Run this only if D1 shows BN wasn't the main cause and you have the time.

---

## D6 — Fix the chart title

Your Cell 14 title claims more than your data shows. Find:

```python
plt.title("BlindSpot — adversarial training restores robustness")
```

Replace with:

```python
plt.title("BlindSpot — adversarial training trades clean accuracy for partial robustness")
```

Re-run the cell and re-save `defence_comparison.png`.

---

# PART 2 — Lab notebook entry

Paste this into `lab-notebook.md`. Fill the `____` blanks from the diagnostic cells.

```markdown
## 2026-11-05 (Days 9-12) - attack and defence

### Setup
- New notebook 05_attack. Loaded detector_v3.pth, 241,794 params.
- Paths resolved with glob (dataset mounted under xhlulu/, model under my own "Pathhh" dataset).
- Preloaded 6,000 validation images @128px (seed=1, same sample as Day 5), 131s.
- SANITY CHECK PASSED: baseline clean accuracy 0.9522, exactly matching Day 4.

### FGSM implemented from scratch
    x_adv = clamp(x + eps * sign(dLoss/dx), 0, 1)
- Gradient taken w.r.t. the INPUT pixels, not the weights. Same backprop machinery,
  stopped one step earlier.
- sign() rather than the raw gradient bounds the per-pixel change to exactly eps,
  which is what keeps the perturbation invisible.

### Single-image demo (validation image #0, eps=0.01)
- fake (0.9995 confident)  ->  real (1.0000 confident)
- Largest pixel change 0.0100 = 2.6 / 255 brightness levels.
- Images visually identical. Saved: results/figures/attack_exhibit.png

### Epsilon sweep, undefended (6,000 images)
    eps      (x/255)   FGSM      PGD-10
    0.000    ( 0.0)    0.9522    0.9522
    0.001    ( 0.3)    0.0090    0.0013
    0.002    ( 0.5)    0.0000    0.0000
    0.005    ( 1.3)    0.0000    0.0000
    0.010    ( 2.6)    0.0000    0.0000
    0.020    ( 5.1)    0.0050    0.0000
    0.030    ( 7.6)    0.0545    0.0000
    0.050    (12.8)    0.1517    0.0000
    0.100    (25.5)    0.0110    0.0000

- COLLAPSE IS TOTAL BY eps=0.002 (0.5/255). Not one of 6,000 images survived.
- DAY 5 PREDICTION CONFIRMED. I predicted on 2026-10-31, before writing any attack code,
  that this detector would break at unusually small epsilon because the blur test showed
  it depends on high-frequency texture (0.9522 -> 0.5085 under blur), which is exactly the
  band FGSM perturbs. Collapse at 0.5/255 is far below the 8/255 budget conventionally
  used as the robustness benchmark for image classifiers.
  CAVEAT: I did not train a reference model on this dataset, so I can say "extremely
  fragile" and compare against the conventional benchmark budget, but I cannot claim
  "more fragile than a comparable detector" - that was never measured.

- NON-MONOTONIC FGSM at eps 0.02-0.05 (accuracy rises to 0.1517 then falls again).
  Explanation: FGSM takes ONE large step along the sign of the gradient. On a curved loss
  surface a large step can overshoot past the adversarial region. PGD, which takes many
  small projected steps, stays at 0.0000 at every one of those epsilons - which confirms
  the adversarial examples still exist there and FGSM is simply failing to find them.
  This is a property of the ATTACK, not robustness of the model.

### Both attack directions (FGSM)
    eps      fake->real (evasion)   real->fake (false accusation)
    0.002    0.0000                 0.0000
    0.005    0.0000                 0.0000
    0.010    0.0000                 0.0000
    0.020    0.0101                 0.0000
- Both harms are equally available to an attacker. Synthetic faces can be pushed past the
  detector, and genuine photographs can be pushed into being flagged as fake.

### Adversarial training (the defence)
- 12 epochs, FGSM eps=0.01 generated from the current model each batch, clean and
  adversarial concatenated into one batch, started from detector_v3.pth, 45,000 images,
  Adam lr=3e-4 with cosine decay, horizontal flips retained.
- Clean accuracy: 0.9522 -> 0.7072. COST OF THE DEFENCE: 24.5 points.

    eps      undefended FGSM   defended FGSM   defended PGD
    0.000    0.9522            0.7072          0.7072
    0.001    0.0090            0.6853          0.6853
    0.002    0.0000            0.6688          0.6673
    0.005    0.0000            0.6173          0.6095
    0.010    0.0000            0.5297          0.5080
    0.020    0.0050            0.3885          0.3193
    0.030    0.0545            0.2730          0.1712
    0.050    0.1517            0.1353          0.0340
    0.100    0.0110            0.0717          0.0013

- Single image #0 at eps=0.01: undefended fake(0.999)->real(1.000);
  defended fake(0.634)->fake(0.569). The defended model holds its verdict.

### Reading the defence honestly
- ABSOLUTE robustness is modest. At the trained eps=0.01 the defended model scores 0.5080
  against PGD, versus a 0.5047 always-predict-real baseline. Against a full-strength
  attack at the budget it was trained on, the defence buys essentially nothing.
- RELATIVE robustness is large. Measured as accuracy retained against each model's own
  clean baseline, at eps=0.005:
      undefended  0.9522 -> 0.0000   (loses 100% of its accuracy)
      defended    0.7072 -> 0.6095   (loses 14%)
  Both numbers must be reported. The first says the defended model is a worse detector;
  the second says it is a far more stable one.
- Robustness is concentrated at and BELOW the trained epsilon and decays above it.
- NO strong gradient masking: PGD <= FGSM at every epsilon, accuracy decreases
  monotonically in eps, and accuracy reaches ~0 at large eps. All three are standard
  checks and all three pass. [Transfer and strong-PGD checks: ____]
- My guide predicted the pattern "holds against FGSM, collapses against PGD". I did NOT
  observe that. The FGSM/PGD gap is small below eps=0.01 and only opens beyond it
  (at 0.05: 0.1353 vs 0.0340), which is the expected behaviour of a stronger attack
  outside the trained budget, not a sign of a fake defence.

### Diagnostics (Day 13)
- BN recalibration on clean data: 0.7072 -> ____  (recovered ____)
- Transfer attack (craft on undefended, test on defended), eps=0.01: white-box ____ vs transfer ____
- Stronger PGD (40 steps, 2 restarts) at eps=0.01: ____ (vs PGD-10 0.5080)
- 8-bit quantization: at eps=0.005 float ____ vs quantized ____
  => real-world headline epsilon is ____
- Blur sensitivity: fraction of accuracy lost at sigma=2.0,
  undefended ____ vs defended ____
  => high-frequency hypothesis ____SUPPORTED / NOT SUPPORTED____

### Open and untested
- Did not retrain at a smaller eps, so I cannot separate "eps=0.01 was too aggressive"
  from "this architecture simply cannot be both accurate and robust here".
- Did not test JPEG compression (only 8-bit rounding).
- Single seed throughout.
- Test set still untouched.
```

---

# PART 3 — Corrected viva answers

Your instinct to not overclaim is right. Here is how to say each thing accurately.

### "Did your defence work?"

> Partially, and the honest answer depends on how you measure it. In absolute terms, at the
> epsilon I trained on, the defended model scores 0.508 against PGD, which is barely above
> the 0.505 always-predict-one-class baseline — so against a full-strength attack at that
> budget it buys almost nothing. But measured against each model's own clean accuracy at a
> slightly smaller epsilon, the undefended model loses 100% of its accuracy while the
> defended one loses 14%. So it is a worse detector that degrades far more gracefully. Both
> numbers belong in the report.

### "Why did clean accuracy drop so much — 24.5 points?"

> Larger than the roughly 10 points usually quoted for adversarial training, and I have a
> hypothesis I tested rather than assumed. My Day 5 blur experiment showed this detector
> depends on high-frequency texture — blurring the input drops it to chance. FGSM
> perturbations live in that same frequency band. So robustness training is forcing the
> model off the very feature it was using. I measured the frequency sensitivity of both
> models to check this. I also tested whether mixed clean/adversarial batches had corrupted
> the BatchNorm statistics, and what I did **not** isolate is whether eps=0.01 was simply
> too aggressive for a model that already collapses at 0.002.

### "How do I know this isn't gradient masking?"

> Four checks. PGD is at least as strong as FGSM at every epsilon — under masking it is
> often weaker. Accuracy decreases monotonically with epsilon. Accuracy approaches zero at
> large epsilon rather than hitting a floor. And a transfer attack crafted on the
> undefended model does worse against the defended one than its own white-box attack, which
> is the direction you want. [Adjust once D2 has run.]

### "Is your detector unusually fragile?"

> It is extremely fragile — every one of 6,000 validation images was misclassified at
> eps=0.002, which is half a brightness level out of 255, and the conventional benchmark
> budget for this kind of work is 8/255. But I want to be careful: I did not train a
> reference detector on this dataset, so I can say it is extremely fragile and far below
> the conventional budget. I cannot say it is more fragile than a comparable model, because
> I never measured one.

### "Why does FGSM accuracy go UP at eps 0.03–0.05?"

> That is a weakness of the attack, not strength of the model. FGSM takes one large step
> along the sign of the gradient. On a curved loss surface a big step can overshoot past
> the adversarial region and land somewhere the model happens to classify correctly. PGD
> takes many small projected steps within the same budget and stays at zero accuracy at
> every one of those epsilons — so the adversarial examples are still there, FGSM just
> fails to find them.

### "Would this work on a real image file?"

> Partly, and I measured it. Epsilons below 1/255 cannot survive being saved as an 8-bit
> image — they round away. I re-ran the sweep with the attacked images quantized to 8 bits
> to simulate saving and reloading a PNG, and [result]. So my real-world headline is
> eps=____, not the smaller float-space numbers. I did not test JPEG compression, which
> would be harder still since it also removes high frequencies.

---

# PART 4 — What's left

## Day 13 — diagnostics and demo (~3 hours)

1. Run D1, D2, D3, D4 above (~8 minutes of compute). D5 if time allows.
2. Fix the chart title (D6) and re-save `defence_comparison.png`.
3. Build the Gradio demo.
4. Download everything, push.

### The Gradio demo

```python
!pip install -q gradio
import gradio as gr
import numpy as np

tf_demo = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

def run_demo(image, eps, which_model):
    model = detector if which_model == "undefended" else defended
    x = tf_demo(image).unsqueeze(0).to(device)

    with torch.no_grad():
        p0 = torch.softmax(model(x), 1)[0]
    y = p0.argmax().unsqueeze(0)

    if eps > 0:
        xa = fgsm(model, x, y, eps)
    else:
        xa = x
    with torch.no_grad():
        p1 = torch.softmax(model(xa), 1)[0]

    orig_img = x[0].permute(1,2,0).cpu().numpy()
    att_img  = xa[0].detach().permute(1,2,0).cpu().numpy()
    d = (xa - x)[0]
    pert_img = ((d - d.min())/(d.max()-d.min()+1e-12)).detach().permute(1,2,0).cpu().numpy()

    return (orig_img,
            f"{names[p0.argmax()]}  ({p0.max():.3f})",
            pert_img,
            att_img,
            f"{names[p1.argmax()]}  ({p1.max():.3f})")

gr.Interface(
    fn=run_demo,
    inputs=[gr.Image(type="pil", label="upload a face"),
            gr.Slider(0, 0.05, value=0.01, step=0.001, label="epsilon (attack strength)"),
            gr.Radio(["undefended", "defended"], value="undefended", label="model")],
    outputs=[gr.Image(label="original"), gr.Textbox(label="verdict on original"),
             gr.Image(label="perturbation (amplified)"),
             gr.Image(label="attacked"),  gr.Textbox(label="verdict on attacked")],
    title="BlindSpot — fooling an AI-generated-face detector",
    description="Upload a face. Drag epsilon. Watch the verdict change while the image doesn't.",
).launch(share=True, debug=False)
```

`share=True` gives a public link that works for ~72 hours. **Test it with a face from
thispersondoesnotexist.com and with a photo of yourself.**

## Day 14 — report and slides (~5 hours)

**Report structure** (the figures you already have in brackets):

1. Introduction — what adversarial examples are, why detection systems matter
2. Background — MLPs, backprop, CNNs, the vanishing gradient problem
3. The detector — architecture, training, results `[training_curve_v3, confusion_matrix, confident_mistakes]`
4. Gradient flow experiments — the depth series and six configurations `[grad_depth_sweep, accuracy_by_depth, grad_fixes_20layer, depth_wall_by_fix]`
5. The attack — FGSM maths, PGD, epsilon sweep, both directions `[attack_exhibit, epsilon_sweep, fgsm_vs_pgd]`
6. The defence — adversarial training, the honest tradeoff `[defence_comparison, frequency_reliance]`
7. Discussion — what worked, what didn't, the wrong predictions
8. Conclusion
9. References — Goodfellow et al. 2014 (FGSM), Madry et al. 2017 (PGD), Karras et al. (StyleGAN)

**Slides, ~12:**

1. Title + the attack exhibit (three panels)
2. The problem: detectors are deployed everywhere — four applications
3. The dataset and the detector architecture
4. Training results, 95.22% + confusion matrix
5. What the detector actually looks at — the blur test, shortcut learning refuted
6. Vanishing gradients: the depth sweep chart
7. The four fixes: gradient flow + depth wall charts
8. Exploding gradients — the unplanned finding
9. The attack: FGSM in one equation + the epsilon sweep
10. FGSM vs PGD, and why FGSM is non-monotonic
11. The defence: both curves, and the honest cost
12. Limitations and future work + syllabus coverage table

**Put a "what I got wrong" slide in.** The residual+BN prediction, the uninterpretable
depth experiment, the overclaimed chart title. Examiners trust a student who flags their
own errors far more than one with a flawless-looking story.

## Day 15 — viva drilling (~3 hours)

Send me the report and the slides and I will ask you every question he could ask, until you
are sick of me. Rehearse the answers **out loud**. Reading them silently is a different skill.

### Demo-day checklist

- [ ] Slides as PDF with every chart embedded — works if all technology fails
- [ ] Pendrive: repo + `detector_v3.pth` + `detector_defended.pth` + report + slides
- [ ] `.pth` files backed up to Drive/WhatsApp as well — they are gitignored
- [ ] Epsilon sweep chart printed on paper
- [ ] `lab-notebook.md` open in a tab for "did you try X?"
- [ ] Never train live. Inference only.

---

## Files to download before closing the session

| File | Why |
|---|---|
| `detector_defended.pth` | 20 minutes of GPU time, gitignored |
| `detector_defended_bnfix.pth` | from D1 |
| `frequency_reliance.png` | from D3 |
| `defence_comparison.png` | re-saved with the corrected title |
| `File → Download notebook` | `05_attack.ipynb` |

```powershell
Move-Item "C:\Users\hp605\Downloads\05_attack.ipynb" "C:\dev\blindspot\notebooks\" -Force
Move-Item "C:\Users\hp605\Downloads\*.pth" "C:\dev\blindspot\models\" -Force
Move-Item "C:\Users\hp605\Downloads\*.png" "C:\dev\blindspot\results\figures\" -Force
```

```bash
git add .
git commit -m "Days 9-13: FGSM/PGD attacks, adversarial training, robustness diagnostics"
git push
```
