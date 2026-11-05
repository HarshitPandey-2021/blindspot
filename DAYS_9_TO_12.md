# BlindSpot — Days 9 to 12: The Attack and The Defence

**Self-contained handoff.** Every cell you need, in order, with what to expect from each.
All of this runs in one Kaggle notebook called `05_attack`.

Written 5 November 2026. Days 1–7 are complete (see `PROJECT_BRIEF.md`).

---

## PART 0 — Prompt to paste into another AI

If you need help from a different model, paste everything between the lines.

---

> I'm a final-year B.Tech CS (AI) student working on a solo Deep Learning semester project called **BlindSpot**, graded by an in-depth technical viva. I'm new to deep learning — I know Python (loops, functions, some numpy/pandas) but the DL concepts are new.
>
> **The project:** I trained a CNN that classifies face photos as "real photograph" or "AI-generated (StyleGAN)". It gets **95.22%** on validation. Now I'm attacking it with FGSM — adding a perturbation invisible to the human eye that makes the detector confidently wrong — and then defending it with adversarial training.
>
> **Setup:** Kaggle Notebooks, free T4 GPU. Dataset is "140k Real and Fake Faces" (xhlulu) — 70k real photos + 70k StyleGAN faces, split train 100k / valid 20k / test 20k. Labels are **fake = 0, real = 1**. Images loaded at 128×128 RGB. I preload a subset into GPU memory as uint8 because reading the files off Kaggle's network drive is 120× slower than training.
>
> **My detector:** 4 blocks of `Conv2d(3x3, pad=1) + BatchNorm2d + ReLU + MaxPool2d(2)`, channels 3→32→64→128→128, then `AdaptiveAvgPool2d(1)`, `Flatten`, `Linear(128, 2)`. 241,794 parameters. Trained with Adam lr=3e-4, cosine decay, random horizontal flips, 25 epochs on 45,000 images. Saved as `detector_v3.pth`.
>
> **What I already found (Days 1–7):**
> - Measured the vanishing gradient on 5/10/20/40-layer sigmoid MLPs. Layer-1 gradient went 1.520e-06 → 3.737e-11 → 1.788e-19 → 3.659e-37. Doubling depth squares the ratio, and all lines are straight on a log axis, which proves exponential decay from repeated multiplication (the chain rule).
> - ReLU alone rescues depth 5 only. He initialisation moves the wall past depth 40 (0.7692 at 40 layers) and was the single best fix. Batch norm gave the best gradient ratio (0.8) and best shallow accuracy (0.8173 at depth 5) but collapsed at depth 40.
> - Residual connections **without** batch norm produced **exploding** gradients (~1000× too large) and failed at depth 10+. Adding batch norm cut magnitudes ~900× and rescued depth 10. That's why real ResNets pair skips with normalisation.
> - My detector depends on **high-frequency texture**: blurring the input collapses accuracy from 0.9522 to 0.5085 (chance), while masking the image border only costs ~10 points. This refuted an earlier worry that it was keying on framing rather than AI artifacts.
> - **Prediction I made in advance and am now testing:** because the detector leans on high-frequency texture, and FGSM produces exactly a small high-frequency perturbation, this detector should be *unusually easy* to fool — accuracy should collapse at very small epsilon.
>
> **What I need help with:** [describe your specific problem here]
>
> Please explain things simply, give me complete code cells I can paste into Kaggle, and tell me what output to expect so I know whether it worked.

---

## Before you start

**Inputs the notebook needs:**

1. The dataset — `+ Add Input` → filter to **Datasets** → search `140k real and fake faces` → add **xhlulu's** (~4 GB). Do not upload your own copy.
2. The model — `Upload` → **Dataset** (not Model) → `detector_v3.pth` → name it `blindspot-model`.

**Session options:** Accelerator **GPU T4 x2**, Internet **On**, Persistence **Variables and Files**.

**The golden rule, learned the hard way:** download any `.pth` the moment training finishes. A session timeout wipes `/kaggle/working` and you repeat the whole run.

---

# DAY 9 — Build the attack

## Cell 1 — Setup, load the model, sanity check

The paths are discovered automatically, so this works regardless of what your inputs are named.

```python
import torch, torch.nn as nn, torch.nn.functional as F
import time, random, glob, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# --- find the inputs wherever Kaggle mounted them ---
d = glob.glob("/kaggle/input/**/real-vs-fake", recursive=True)
m = glob.glob("/kaggle/input/**/*.pth", recursive=True)
DATA       = d[0] if d else None
MODEL_PATH = m[0] if m else None
print("DATA       =", DATA)
print("MODEL_PATH =", MODEL_PATH)
assert DATA,       "Faces dataset not attached. Add Input -> Datasets -> '140k real and fake faces' (xhlulu)."
assert MODEL_PATH, "Model not attached. Upload detector_v3.pth as a Dataset."

# --- the detector architecture, exactly as trained on Day 4 ---
def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

def make_detector():
    return nn.Sequential(
        block(3, 32), block(32, 64), block(64, 128), block(128, 128),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 2),
    ).to(device)

detector = make_detector()
detector.load_state_dict(torch.load(MODEL_PATH, map_location=device))
detector.eval()
print(f"model loaded — {sum(p.numel() for p in detector.parameters()):,} parameters")
```

**Expect:** `device: cuda`, both paths printed, `241,794 parameters`.

## Cell 2 — Load the validation images (~4 minutes)

```python
SIZE, N_VALID = 128, 6000
tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor()])
va_ds = datasets.ImageFolder(f"{DATA}/valid", transform=tf)
print("classes:", va_ds.classes, "  (fake=0, real=1)")

def preload(ds, n, seed=1):
    idx = random.Random(seed).sample(range(len(ds)), n)
    dl = DataLoader(Subset(ds, idx), batch_size=256, shuffle=False, num_workers=4)
    xs, ys, t0 = [], [], time.time()
    for i, (x, y) in enumerate(dl):
        xs.append((x * 255).to(torch.uint8)); ys.append(y)
        if i % 5 == 0:
            print(f"   {min((i+1)*256, n):,}/{n:,}   {time.time()-t0:.0f}s", flush=True)
    return torch.cat(xs), torch.cat(ys)

Xva, Yva = preload(va_ds, N_VALID)
Xva, Yva = Xva.to(device), Yva.to(device)
print("valid:", Xva.shape, f"  GPU {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

`seed=1` matches the Day 4/5 validation sample, so your numbers stay comparable.

## Cell 3 — Confirm the model still works

**Do not skip this.** A model can load from a wrong file and silently predict nonsense.

```python
def accuracy(model, X, Y, bs=256):
    model.eval(); c = 0
    with torch.no_grad():
        for i in range(0, X.size(0), bs):
            c += (model(X[i:i+bs].float()/255.).argmax(1) == Y[i:i+bs]).sum().item()
    return c / X.size(0)

BASE = accuracy(detector, Xva, Yva)
print(f"baseline clean accuracy: {BASE:.4f}      (expect ~0.9522)")
```

**If this isn't around 0.95, stop.** Something's wrong with the model file or the data path.

## Cell 4 — FGSM, written from scratch

This is the heart of the project. Twenty lines.

```python
def fgsm(model, x, y, eps):
    """
    Fast Gradient Sign Method.

    Normally we compute the gradient of the loss with respect to the WEIGHTS,
    to know how to change the model so it is less wrong.
    Here we compute it with respect to the INPUT PIXELS instead, and move the
    pixels in the direction that makes the model MORE wrong.

    x : float tensor in [0,1], shape [N,3,H,W]
    y : true labels
    eps : how far each pixel may move (the attack budget)
    """
    x = x.clone().detach().requires_grad_(True)     # we want gradients on the IMAGE
    loss = nn.CrossEntropyLoss()(model(x), y)
    model.zero_grad()
    loss.backward()                                  # fills x.grad
    x_adv = x + eps * x.grad.sign()                  # one step, uphill on the loss
    return x_adv.clamp(0, 1).detach()                # keep it a valid image
```

**Why the `sign()` and not the gradient itself?** The sign makes every pixel move by exactly `eps`, so the largest change to any single pixel is bounded by `eps` no matter how steep the gradient is. That's what keeps the perturbation invisible. Using the raw gradient would move some pixels enormously and others barely at all. **Expect this question in your viva.**

## Cell 5 — Attack one face and watch it flip

```python
names = va_ds.classes       # ['fake', 'real']

def predict(model, X, bs=256):
    model.eval(); P = []
    with torch.no_grad():
        for i in range(0, X.size(0), bs):
            P.append(torch.softmax(model(X[i:i+bs].float()/255.), 1))
    return torch.cat(P)

P = predict(detector, Xva)
conf, pred = P.max(1)

# pick a FAKE image the detector confidently and correctly calls fake
cand = ((Yva == 0) & (pred == 0) & (conf > 0.99)).nonzero().squeeze(1)
i = cand[0].item()
print(f"chose validation image #{i} — a fake the detector was {conf[i]:.4f} sure about\n")

EPS = 0.01
x     = Xva[i:i+1].float() / 255.
y     = Yva[i:i+1]
x_adv = fgsm(detector, x, y, EPS)

with torch.no_grad():
    p0 = torch.softmax(detector(x),     1)[0]
    p1 = torch.softmax(detector(x_adv), 1)[0]

print(f"original : {names[p0.argmax()]:5s}  confidence {p0.max():.4f}")
print(f"attacked : {names[p1.argmax()]:5s}  confidence {p1.max():.4f}")
print(f"\nlargest change to any pixel: {(x_adv-x).abs().max().item():.4f}  (eps = {EPS})")
print(f"that is {(x_adv-x).abs().max().item()*255:.1f} out of 255 brightness levels")
```

**What you want to see:** `original : fake` with high confidence, `attacked : real` with high confidence. If it doesn't flip, raise `EPS` to 0.02 or 0.03 and run again.

## Cell 6 — The exhibit (your title slide)

```python
pert = (x_adv - x)
amp  = (pert - pert.min()) / (pert.max() - pert.min() + 1e-12)   # stretch to 0..1 so it's visible

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
panels = [
    (x[0],     f"ORIGINAL\ndetector says: {names[p0.argmax()]}  ({p0.max():.3f})"),
    (amp[0],   f"THE PERTURBATION\namplified ~{1/EPS:.0f}x so you can see it"),
    (x_adv[0], f"ATTACKED\ndetector says: {names[p1.argmax()]}  ({p1.max():.3f})"),
]
for ax, (img, title) in zip(axes, panels):
    ax.imshow(img.detach().permute(1, 2, 0).cpu().numpy())
    ax.set_title(title, fontsize=12, pad=10)
    ax.axis("off")
plt.suptitle(f"BlindSpot — FGSM attack at epsilon = {EPS}", fontsize=15)
plt.tight_layout(); plt.subplots_adjust(top=0.82)
plt.savefig("/kaggle/working/attack_exhibit.png", dpi=150, bbox_inches="tight")
plt.show()
```

The left and right images should look **identical** to your eye. Different verdicts underneath. That's the project in one picture.

**Download `attack_exhibit.png` now.**

---

# DAY 10 — How much noise does it take?

## Cell 7 — The epsilon sweep (your headline chart)

```python
def accuracy_under(model, X, Y, eps, attack, bs=256):
    c = 0
    for i in range(0, X.size(0), bs):
        x = X[i:i+bs].float() / 255.
        y = Y[i:i+bs]
        xa = attack(model, x, y, eps)
        with torch.no_grad():
            c += (model(xa).argmax(1) == y).sum().item()
    return c / X.size(0)

EPSILONS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]

acc_fgsm = []
for e in EPSILONS:
    a = accuracy_under(detector, Xva, Yva, e, fgsm)
    acc_fgsm.append(a)
    print(f"eps {e:<6.3f}  ({e*255:5.1f}/255)   accuracy {a:.4f}")
```

Takes a couple of minutes. **This table is the single most important result in your project.**

Look for the epsilon where accuracy falls below 0.60, and compare it to the epsilon where you can actually *see* damage in the image (try the exhibit cell at a few values). **The gap between those two numbers is the whole point of the project.**

This is also where your Day 5 prediction gets tested: you predicted the collapse would happen at unusually small epsilon because the detector relies on high-frequency texture. Write down whether it did.

## Cell 8 — Chart it

```python
plt.figure(figsize=(9, 5.5))
plt.plot([e*255 for e in EPSILONS], acc_fgsm, marker="o", linewidth=2, label="FGSM")
plt.axhline(0.5, color="crimson", ls="--", label="random guessing")
plt.xlabel("epsilon  (max change per pixel, out of 255)")
plt.ylabel("detector accuracy")
plt.title("BlindSpot — detector accuracy under FGSM attack")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("/kaggle/working/epsilon_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
```

## Cell 9 — Both directions

Two different harms. Most people only test one.

```python
fake_idx = (Yva == 0).nonzero().squeeze(1)
real_idx = (Yva == 1).nonzero().squeeze(1)

print("eps      fake->real (evasion)   real->fake (false accusation)")
for e in [0.002, 0.005, 0.01, 0.02]:
    af = accuracy_under(detector, Xva[fake_idx], Yva[fake_idx], e, fgsm)
    ar = accuracy_under(detector, Xva[real_idx], Yva[real_idx], e, fgsm)
    print(f"{e:<8.3f} {af:.4f}                  {ar:.4f}")
```

- **fake → real** is evasion: a synthetic face passes verification.
- **real → fake** is a genuine person being wrongly flagged.

If one direction is easier than the other, say so and say why it matters.

## Cell 10 — PGD, the stronger attack

```python
def pgd(model, x, y, eps, steps=10, alpha=None):
    """
    Projected Gradient Descent = FGSM done many times with small steps,
    projected back into the eps-ball after each one. Much stronger.
    """
    if alpha is None:
        alpha = max(eps / 4, 1e-4)
    x0 = x.clone().detach()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1).detach()
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(xa), y)
        model.zero_grad()
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1).detach()
    return xa

acc_pgd = []
for e in EPSILONS:
    a = accuracy_under(detector, Xva, Yva, e, pgd)
    acc_pgd.append(a)
    print(f"eps {e:<6.3f}   FGSM {acc_fgsm[EPSILONS.index(e)]:.4f}    PGD {a:.4f}")
```

Slower — ten forward/backward passes per batch instead of one. A few minutes.

**PGD should be strictly worse for the defender at every epsilon.** FGSM takes one big step and can overshoot; PGD takes ten careful ones and finds a better attack inside the same budget.

```python
plt.figure(figsize=(9, 5.5))
plt.plot([e*255 for e in EPSILONS], acc_fgsm, marker="o", label="FGSM (1 step)")
plt.plot([e*255 for e in EPSILONS], acc_pgd,  marker="s", label="PGD (10 steps)")
plt.axhline(0.5, color="crimson", ls="--", label="random guessing")
plt.xlabel("epsilon  (out of 255)"); plt.ylabel("detector accuracy")
plt.title("FGSM vs PGD — same budget, different strength")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("/kaggle/working/fgsm_vs_pgd.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Download `epsilon_sweep.png` and `fgsm_vs_pgd.png`.**

---

# DAYS 11–12 — The defence

## Cell 11 — Load the training images (~12 minutes)

```python
tr_ds = datasets.ImageFolder(f"{DATA}/train", transform=tf)
Xtr, Ytr = preload(tr_ds, 45000, seed=0)
Xtr, Ytr = Xtr.to(device), Ytr.to(device)
print("train:", Xtr.shape, f"  GPU {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

Stay on the tab. Go make tea.

## Cell 12 — Adversarial training (~20 minutes)

```python
def train_adversarial(model, X, Y, Xv, Yv, epochs=12, bs=128, lr=3e-4,
                      eps=0.01, save_to=None):
    """
    Each batch: generate FGSM examples from the CURRENT model, then train on
    clean AND attacked images together, both with their correct labels.
    The model learns that the perturbed version is still the same class.
    """
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    n, best = X.size(0), 0.0

    for ep in range(epochs):
        t0 = time.time(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i:i+bs]
            x, y = X[b].float() / 255., Y[b]

            f = torch.rand(x.size(0), device=device) < 0.5     # keep the flip augmentation
            x[f] = torch.flip(x[f], dims=[3])

            model.eval()                                        # don't pollute BN stats
            x_adv = fgsm(model, x, y, eps)
            model.train()

            x_mix = torch.cat([x, x_adv])
            y_mix = torch.cat([y, y])
            loss = loss_fn(model(x_mix), y_mix)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        va = accuracy(model, Xv, Yv)
        print(f"epoch {ep+1:2d}   clean valid {va:.4f}   ({time.time()-t0:.0f}s)")
        if save_to and va > best:
            best = va
            torch.save(model.state_dict(), save_to)
    return best

defended = make_detector()
defended.load_state_dict(torch.load(MODEL_PATH, map_location=device))   # start from the trained model
best_clean = train_adversarial(defended, Xtr, Ytr, Xva, Yva, epochs=12, eps=0.01,
                               save_to="/kaggle/working/detector_defended.pth")
print(f"\nbest clean accuracy after adversarial training: {best_clean:.4f}")
```

**DOWNLOAD `detector_defended.pth` THE MOMENT THIS FINISHES.**

Expect clean accuracy somewhere near 0.90–0.94 — **lower than the original 0.9522**. That drop is not a bug, it's the point. Measure it honestly; it's your bias-variance discussion.

## Cell 13 — Does the defence hold?

```python
defended.eval()
clean_def = accuracy(defended, Xva, Yva)
print(f"clean accuracy  —  undefended {BASE:.4f}   defended {clean_def:.4f}")
print(f"cost of the defence: {BASE - clean_def:+.4f}\n")

acc_def_fgsm, acc_def_pgd = [], []
for e in EPSILONS:
    a1 = accuracy_under(defended, Xva, Yva, e, fgsm)
    a2 = accuracy_under(defended, Xva, Yva, e, pgd)
    acc_def_fgsm.append(a1); acc_def_pgd.append(a2)
    print(f"eps {e:<6.3f}   undefended {acc_fgsm[EPSILONS.index(e)]:.4f}   "
          f"defended(FGSM) {a1:.4f}   defended(PGD) {a2:.4f}")
```

**The honest test is the PGD column.** The model was trained on FGSM examples. If it holds against FGSM but falls to PGD, it learned to resist *that specific attack* rather than becoming genuinely robust — a very well-known result, and saying it unprompted is worth real marks.

## Cell 14 — The money chart

One figure, the whole story.

```python
xs = [e*255 for e in EPSILONS]
plt.figure(figsize=(10, 6))
plt.plot(xs, acc_fgsm,      marker="o", lw=2, label="undefended — FGSM")
plt.plot(xs, acc_pgd,       marker="o", lw=2, ls=":", label="undefended — PGD")
plt.plot(xs, acc_def_fgsm,  marker="s", lw=2, label="defended — FGSM")
plt.plot(xs, acc_def_pgd,   marker="s", lw=2, ls=":", label="defended — PGD")
plt.axhline(0.5, color="crimson", ls="--", label="random guessing")
plt.xlabel("epsilon  (max change per pixel, out of 255)")
plt.ylabel("detector accuracy")
plt.title("BlindSpot — adversarial training restores robustness")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("/kaggle/working/defence_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

## Cell 15 — Side-by-side exhibit, defended model

```python
x_adv2 = fgsm(defended, x, y, EPS)
with torch.no_grad():
    q0 = torch.softmax(defended(x),      1)[0]
    q1 = torch.softmax(defended(x_adv2), 1)[0]

print(f"UNDEFENDED   original {names[p0.argmax()]} ({p0.max():.3f})  ->  attacked {names[p1.argmax()]} ({p1.max():.3f})")
print(f"DEFENDED     original {names[q0.argmax()]} ({q0.max():.3f})  ->  attacked {names[q1.argmax()]} ({q1.max():.3f})")
```

The defended model should hold its verdict where the undefended one flipped. **That's your before/after slide.**

---

# What to download before you close the tab

| File | Why |
|---|---|
| `detector_defended.pth` | **Critical.** 20 minutes of GPU time. Back it up off the laptop too. |
| `attack_exhibit.png` | Title slide |
| `epsilon_sweep.png` | Headline chart |
| `fgsm_vs_pgd.png` | Attack strength comparison |
| `defence_comparison.png` | The money chart |
| `File → Download notebook` | The notebook itself |

Then on the laptop:

```powershell
Move-Item "C:\Users\hp605\Downloads\05-attack.ipynb" "C:\dev\blindspot\notebooks\" -Force
Move-Item "C:\Users\hp605\Downloads\detector_defended.pth" "C:\dev\blindspot\models\" -Force
Move-Item "C:\Users\hp605\Downloads\*.png" "C:\dev\blindspot\results\figures\" -Force
```

```bash
git add .
git commit -m "Days 9-12: FGSM and PGD attacks, adversarial training defence"
git push
```

---

# Lab notebook template

Fill in your real numbers as you go.

```markdown
## 2026-11-__ (Days 9-12) - attack and defence
- Baseline clean accuracy on 6,000 validation images: 0.____  (matches Day 4's 0.9522)

- FGSM implemented from scratch: x_adv = clamp(x + eps * sign(dLoss/dx), 0, 1).
  Gradient taken w.r.t. the INPUT, not the weights. sign() bounds the per-pixel change
  to exactly eps, which is what keeps the perturbation invisible.

- Single-image demo at eps=____: detector went from "fake" (___) to "real" (___).
  Largest pixel change ____ /255. Images visually identical.
  Saved: results/figures/attack_exhibit.png

- EPSILON SWEEP (FGSM):
    eps 0.000 -> ____    eps 0.005 -> ____    eps 0.020 -> ____
    eps 0.001 -> ____    eps 0.010 -> ____    eps 0.050 -> ____
  Accuracy dropped below 0.60 at eps = ____ (= ____ /255 brightness levels).
  Damage first visible to the eye at roughly eps = ____.
  DAY 5 PREDICTION ___CONFIRMED / NOT CONFIRMED___: I predicted collapse at unusually
  small epsilon because the detector relies on high-frequency texture (blur test:
  0.9522 -> 0.5085). Actual result: ____

- BOTH DIRECTIONS at eps=0.01: fake->real ____, real->fake ____.
  ____ direction is easier, which matters because ____.

- PGD (10 steps) vs FGSM at equal eps: PGD stronger at every epsilon, e.g. at 0.01
  FGSM ____ vs PGD ____. Expected: FGSM takes one big step and can overshoot; PGD
  takes 10 small projected steps and finds a better point in the same budget.

- ADVERSARIAL TRAINING: 12 epochs, FGSM eps=0.01, clean+adv in every batch,
  started from detector_v3.pth.
    clean accuracy  undefended 0.____  ->  defended 0.____   (cost: ____)
    under FGSM @0.01  undefended ____  ->  defended ____
    under PGD  @0.01  undefended ____  ->  defended ____
  The defence was trained on FGSM. Against PGD it ____, which shows ____.
  Saved: models/detector_defended.pth, results/figures/defence_comparison.png
```

---

# Viva questions for this section

Have answers ready before Day 14.

**On the attack**
- Why the *sign* of the gradient and not the gradient itself? *(sign bounds the per-pixel change to exactly eps regardless of gradient steepness — that's what keeps it invisible)*
- Why is this a white-box attack? What would change in a black-box setting? *(we have the weights, so we can compute gradients; black-box means query-only, needing transfer attacks or gradient estimation)*
- What does epsilon control, and what is the largest epsilon a human still can't see?
- Why is PGD stronger than FGSM at the same epsilon?
- Would the attacked image still fool the detector after being saved as a JPEG? *(Honest answer: my Day 5 blur test showed the detector depends on high-frequency texture, and JPEG also removes high frequencies — so I'd expect both the attack AND the detector to degrade. I didn't test it.)*

**On the defence**
- How does adversarial training work, and why does it cost clean accuracy? *(the model must now fit a larger, harder distribution; capacity spent on robustness isn't spent on clean accuracy — a bias-variance tradeoff)*
- Does training on FGSM defend against PGD? Did you test it?
- What other defences exist and why didn't you use them? *(defensive distillation, input preprocessing, certified defences — most preprocessing defences were broken by adaptive attacks; certified defences need far more compute than a semester project)*
- What's the bias-variance reading of what the defence did?

---

# If something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `assert DATA` fails | Faces dataset not attached, or your own upload shadowing it | Add Input → **Datasets** → xhlulu's "140k real and fake faces" |
| Baseline accuracy ~0.50 | Wrong `.pth`, or `detector_v1_96px_OLD.pth` uploaded by mistake | Re-upload `detector_v3.pth` |
| `CUDA out of memory` | Converting too many images to float at once | Lower `bs` from 256 to 128 in the loops |
| Attack doesn't flip the label | Epsilon too small for that image | Raise `EPS` to 0.02 or 0.03 |
| `NameError` on anything | Session restarted; variables gone | Re-run cells from the top |
| Session dies mid-run | Kaggle kills idle sessions (~20 min) | Stay on the tab; persistence is on so `/kaggle/working` survives |

---

**Days 9–12 all fit in this one notebook.** Only the Gradio demo (Day 13) needs its own.

Total runtime, including the two preloads: about **an hour of waiting**, maybe 90 minutes of actual work.
