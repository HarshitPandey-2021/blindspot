# BlindSpot — Foundations

Everything so far, assuming you know nothing. Re-read this whenever you feel lost.

---

# Part 1 — What we are building

## The one-paragraph version

A program that looks at a photo of a human face and answers one question: **is this a real photograph, or a face that AI generated?** It gets this right about 95 times out of 100.

Then we **trick** it. We change the photo by a tiny amount — so tiny that if you put the two photos side by side, you cannot see any difference. The program now gets it wrong, confidently.

Then we **fix** it, so the trick stops working. And we prove the fix worked with numbers.

**Build it. Break it. Fix it.** That's the project.

## Why anyone cares

Imagine a bank app. You open an account, it asks for a selfie to prove you're a real person. Behind that is a program like the one you're building.

Now imagine someone generates a fake face with AI, adds invisible noise to it, and uploads it. The bank's program says "real person, verified." An account opens in a name belonging to nobody.

That's not hypothetical. It's why this is a research field.

Four places this matters:

1. **Social media** — detectors that flag AI-generated images. If they can be fooled, they're decoration.
2. **Bank / KYC verification** — the scenario above.
3. **The opposite problem** — a *real* person's photo pushed over the line into "fake." Someone genuine gets rejected or accused. Fewer people test this direction. You will.
4. **The general lesson** — this isn't one broken program. Every neural network trained the normal way has this weakness.

---

# Part 2 — What a neural network actually is

## The dials analogy

Picture a machine with **100,000 dials** on it. You feed a photo in one end. A number comes out the other end: "0.8 fake."

Every dial affects the output a little. Turn one dial, the answer shifts slightly.

**Training = finding the right positions for all 100,000 dials.**

That's it. That's a neural network. Everything else is detail about how the dials are arranged and how you find the right positions.

## How the dials are arranged: layers

The dials aren't in one big pile. They're in groups called **layers**, stacked one after another.

```
photo → [layer 1] → [layer 2] → [layer 3] → ... → answer
```

Each layer takes numbers in, does its multiplication with its own dials, and passes new numbers to the next layer.

**A real example from your code:**

```python
nn.Linear(784, 256)
```

This is one layer. It takes 784 numbers in and produces 256 numbers out. Each of the 256 outputs is a weighted sum of all 784 inputs. So this one layer holds 784 × 256 = **200,704 dials**, plus 256 more (one "bias" per output).

The proper word for a dial is a **weight**.

## Why images become a row of numbers

A 28×28 greyscale image is just 784 numbers, each between 0 (black) and 1 (white).

```python
nn.Flatten()
```

That line does exactly this — takes the 28×28 grid and lays it out flat as a row of 784 numbers, so the first `Linear` layer can accept it.

A colour image has three of these grids (red, green, blue). A 128×128 colour photo is 128 × 128 × 3 = 49,152 numbers.

**To a neural network, an image is never a picture. It is always just a long list of numbers.**

## The "bend" — activation functions

Here's something that trips everyone up.

If you stack layers that only multiply and add, then **stacking 100 of them gives you exactly the same power as 1 of them.** A chain of straight lines is still a straight line. Depth would be pointless.

So after each layer, we apply a simple bend:

```python
nn.ReLU()
```

ReLU is almost comically simple: **if the number is negative, make it zero. If it's positive, leave it alone.** That's the entire function.

```
input:   -3   -1    0    2    5
output:   0    0    0    2    5
```

That tiny bend is what lets a deep network learn curved, complicated patterns instead of straight-line ones.

**Sigmoid** is the old alternative. It squashes any number into the range 0 to 1 using a smooth S-curve. It's gentler than ReLU — and as you discovered yesterday, that gentleness is exactly what causes the vanishing gradient problem.

---

# Part 3 — How it learns

## The golf score

The model makes a guess. We compare it to the truth and produce **one number** saying how wrong it was.

That number is the **loss**. Like golf — **lower is better, zero is perfect.**

```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

`logits` is what the model guessed. `labels` is the truth. `loss` is one number: how wrong.

Cross-entropy is the standard loss for "pick one category out of several." It punishes confident wrong answers much harder than uncertain wrong answers — which is what you want. A model that says "definitely a 7" about a 3 deserves more punishment than one that says "maybe a 7, maybe a 3."

## Which way to turn each dial

Now the real question: we have 100,000 dials and one number saying we're wrong. **Which dials do we turn, and which way?**

For each dial, ask: *if I turned this one slightly up, would the loss go up or down?*

The answer for one dial is its **gradient**. A gradient is just a direction with a size: "turning this dial up makes things worse, strongly" or "turning this up makes things slightly better."

```python
loss.backward()
```

That one line computes the gradient for **all 100,000 dials at once**. The algorithm that does it is called **backpropagation**, and it's the single most important idea in the field.

```python
opt.step()
```

That line actually turns every dial a small amount in the better direction.

**Repeat a few hundred thousand times. That is training. There is nothing else.**

## The factory analogy for backpropagation

A factory assembly line with 20 workers. A product goes down the line, each worker does their bit, and at the end an inspector finds a defect.

The inspector tells worker 20: "you contributed this much to the defect, adjust." Worker 20 figures out how much of that was really caused by what worker 19 handed them, and passes that share back. Worker 19 does the same to worker 18. And so on, back down the line.

**That backwards flow of blame is backpropagation.** Each worker only needs to know their own job and what the person after them told them. Nobody needs to see the whole factory.

## Why `opt.zero_grad()` exists

PyTorch **adds** new gradients to whatever was already there, instead of replacing them. So without clearing, step 2 would use step 1's gradients plus its own, step 3 would have all three piled up, and your training would be garbage.

```python
opt.zero_grad()   # wipe the slate
loss.backward()   # compute fresh gradients
opt.step()        # turn the dials
```

**Always in that order.** Forgetting `zero_grad()` is the single most common bug in PyTorch, and a fine thing to mention in a viva.

## Epochs and batches — the flashcard analogy

You have 60,000 flashcards to learn.

- Doing **all 60,000**, then updating your understanding once = terribly slow learning.
- Doing **one card** and updating after each = fast but jumpy, you overreact to each card.
- Doing **128 cards**, then updating = the sweet spot.

That group of 128 is a **batch**.

```python
DataLoader(train_ds, batch_size=128, shuffle=True)
```

Going through all 60,000 cards once is an **epoch**.

```python
for epoch in range(3):
```

Three epochs = three complete passes through the whole dataset. That's why you saw three accuracy numbers.

`shuffle=True` reshuffles the deck each epoch, so the model never learns the *order* of the cards instead of their content.

## The optimiser

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
```

The optimiser decides *how* to turn the dials once backprop says which way.

- `model.parameters()` — "here are all 100,000 dials, you're in charge of them"
- `lr=1e-3` — the **learning rate**, 0.001. How big each turn is.

**Learning rate is the most important setting in all of deep learning.** Too big and the model overshoots and never settles — loss bounces around or explodes. Too small and it learns correctly but takes forever.

**Adam** is a smarter version of basic gradient descent: it adapts the step size per dial based on recent history. It's the sensible default and what you should use unless you have a reason not to.

---

# Part 4 — The vanishing gradient (what you measured)

## The whisper game

Twenty people stand in a line. The person at the back whispers a message to the person in front of them, who whispers it on, and so on.

But each person only passes on **a quarter of the volume** they heard.

By the time it reaches the front of the line, the message is silent. Person 1 hears nothing and does nothing.

**That's the vanishing gradient problem.** The "message" is the correction signal from backpropagation. It gets quieter at every layer it passes through.

## Why it gets quieter

Backprop multiplies the signal by each layer's derivative on the way back.

The derivative of sigmoid is **at most 0.25**, usually much less. So:

| Layer | Signal strength |
|---|---|
| 20 (last) | 1 |
| 19 | 0.25 |
| 18 | 0.0625 |
| 17 | 0.0156 |
| ... | ... |
| 1 (first) | about 0.0000000001 |

The first layer gets an instruction so weak it's indistinguishable from "do nothing."

## Your actual numbers

```
sigmoid    layer 1: 1.722e-19     last layer: 2.325e-02     ratio: 1.35e17
relu       layer 1: 4.328e-11     last layer: 6.420e-04     ratio: 1.48e7
```

`1.722e-19` means 0.0000000000000000001722.

Your 20-layer sigmoid network was telling its first layer to adjust by that amount. It will never learn anything. **You have a 20-layer network where only the last few layers work.**

## The detail that makes you look sharp

Both lines on your chart are **straight**, and the y-axis is logarithmic.

A straight line on a log axis means **exponential decay** — the signal shrinks by a constant *factor* every layer.

From your numbers:
- Sigmoid loses about **8× per layer**
- ReLU loses about **2.4× per layer**

And constant-factor-per-layer is *exactly* what repeated multiplication predicts. **The straightness of that line is the proof that the chain rule is the cause.** If the cause were something else, the line would curve.

Say that sentence in your viva.

## The honest surprise

ReLU did **not** fix it. It made layer 1 about 100 million times stronger — a massive improvement — but `4.3e-11` is still far too weak to learn from.

Why? Backprop multiplies by two things at each layer: the activation's derivative **and the weights themselves**. ReLU fixes the first. PyTorch's default weight initialisation isn't scaled for 20 layers, so the weights quietly shrink the signal too, and ReLU does nothing about that.

This is **good news for your project** — it means the three remaining fixes have a real job to do:

| Fix | What it attacks |
|---|---|
| **He initialisation** | Scales starting weights so they stop shrinking the signal |
| **Batch normalization** | Re-centres each layer's numbers so they stay in a healthy range |
| **Skip connections** | Gives the signal a shortcut that bypasses the multiplication chain entirely. At an addition node the gradient passes through *unchanged*. This is what ResNet is. |

You'll test each one on Day 7.

---

# Part 5 — Every line of your code

## The imports — where the libraries come from

```python
import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
```

| Library | What it is |
|---|---|
| **torch** | PyTorch. The engine — tensors, gradients, GPU work |
| **torch.nn** | The building blocks — layers, activations, loss functions |
| **torchvision** | Image-specific extras — ready-made datasets and image transforms |
| **DataLoader** | Feeds data in batches, handles shuffling |
| **matplotlib** | Charts |

**You did not install any of these.** Kaggle's notebook image ships with them pre-installed — that's a large part of why we use Kaggle. On your own machine you'd need `pip install torch torchvision matplotlib` and a correctly matched CUDA version, which is exactly the kind of afternoon this project doesn't have time for.

## Picking the GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

`"cuda"` means the GPU. If no GPU is attached, fall back to the CPU so the code still runs instead of crashing.

This is why you later see `.to(device)` everywhere — it moves data onto the GPU. **Model and data must be on the same device or PyTorch throws an error.** That error will happen to you at some point; this is the reason.

## Where the dataset comes from

```python
tf = transforms.ToTensor()
train_ds = datasets.MNIST("/kaggle/working/data", train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST("/kaggle/working/data", train=False, download=True, transform=tf)
```

Line by line:

- `transforms.ToTensor()` converts a loaded image into a PyTorch tensor and rescales pixels from 0–255 to **0.0–1.0**. Neural networks train much better on small numbers.
- `datasets.MNIST(...)` downloads the dataset **from the internet** on first run (this is why Internet had to be On in Settings) and saves it to `/kaggle/working/data`.
- Those `train-images-idx3-ubyte.gz` files you saw in the output panel? That's the download. Second run finds them already there and skips the download.
- `train=True` gives 60,000 images for learning. `train=False` gives a separate 10,000 for testing.

**Why two separate sets, and this matters:** the test set is images the model has *never seen during training*. If we measured accuracy on the training images, a model that simply memorised all 60,000 would score 100% and be useless on anything new. Testing on unseen data is the only honest measurement. Memorising instead of learning is called **overfitting**, and it's in your Unit III.

```python
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=256)
```

Wraps the datasets so they hand out batches. No shuffle for testing — order doesn't matter when you're only measuring.

## The model

```python
model = nn.Sequential(
    nn.Flatten(),           # 28x28 grid -> flat row of 784 numbers
    nn.Linear(784, 256),    # 784 in -> 256 out   (200,704 weights)
    nn.ReLU(),              # the bend
    nn.Linear(256, 10),     # 256 in -> 10 out    (2,560 weights)
).to(device)
```

`nn.Sequential` just means "run these in order, top to bottom."

**Why 10 outputs?** Ten digits, 0 through 9. The model produces a score for each. Highest score wins.

Example output for one image:

```
[-2.1, 0.3, 8.7, 1.2, -0.5, 0.1, -1.8, 2.2, 0.4, -0.9]
   0    1    2    3    4    5    6    7    8    9
```

Index 2 has the highest score, so the model says **"2"**. Those raw scores are called **logits**.

`.to(device)` moves all those weights onto the GPU.

## The training loop

```python
for epoch in range(3):
    model.train()
    for images, labels in train_dl:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()
```

- `model.train()` — puts the model in training mode. Matters once you use dropout or batch norm, which behave differently during training vs testing. Harmless here, but build the habit.
- `for images, labels in train_dl` — hands you 128 images and their 128 correct answers, 469 times (60,000 ÷ 128).
- `logits = model(images)` — **forward pass.** What does it think?
- `loss = loss_fn(...)` — how wrong, as one number.
- The three lines you know: wipe, compute gradients, turn dials.

**Those three lines are the whole field.** Everything in the next two weeks is a variation on them. Your attack in Phase 3 changes exactly one thing: it computes the gradient with respect to the **image pixels** instead of the weights.

## How accuracy is calculated

```python
model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_dl:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
print(f"epoch {epoch+1}  test accuracy {correct/len(test_ds):.4f}")
```

Unpacking the dense line, inside out:

1. `model(images)` → for 256 images, 256 rows of 10 scores each
2. `.argmax(1)` → **"which index holds the biggest number?"** Turns 10 scores into one predicted digit. `[-2.1, 0.3, 8.7, ...]` → `2`
3. `== labels` → compare to the truth, gives `[True, False, True, True, ...]`
4. `.sum()` → `True` counts as 1, so this counts how many were right
5. `.item()` → pulls that count out of the tensor as a plain Python number

Then `correct / 10000` = the fraction right. **0.9698 means 9,698 of 10,000 test images correct.**

Two supporting pieces:

- `model.eval()` — evaluation mode, the counterpart to `model.train()`
- `with torch.no_grad()` — "don't track gradients here." We're only measuring, not learning. Makes it faster and uses less memory.

## How long things take

| Step | Time | Why |
|---|---|---|
| First cell of a session | ~30 sec | Kaggle is booting the machine |
| MNIST download | ~10 sec | First run only; cached after |
| One epoch, your MLP | ~15–20 sec | On the T4 |
| Full 3-epoch run | ~1 min | |
| The gradient demo | ~5 sec | One batch, one backward pass. No training at all. |
| Phase 2: the face detector | 10–30 min | 140,000 images instead of 60,000, and a CNN is heavier |

If something runs much longer than expected, the usual cause is the GPU not actually being attached — check `torch.cuda.is_available()` first, always.

---

# Part 6 — How to remember all this

Don't memorise definitions. Memorise these six sentences, and derive the rest.

1. **A neural network is a stack of dials.** Training is finding their right positions.
2. **Loss is a golf score.** One number, lower is better.
3. **A gradient tells one dial which way to turn.** Backprop computes all of them at once.
4. **Backprop is blame passed backwards down an assembly line.**
5. **Vanishing gradient is the whisper game** — blame gets quieter at every layer, until the first layer hears nothing.
6. **The attack is backprop pointed at the pixels instead of the dials.**

If you can say those six out loud, unprompted, you can reconstruct almost any answer he asks for.

## The three lines to know cold

```python
opt.zero_grad()   # wipe old gradients - PyTorch ADDS, it doesn't replace
loss.backward()   # backpropagation - gradient for every weight
opt.step()        # turn every dial slightly
```

**Expect to be asked what happens if you remove `zero_grad()`.** Answer: gradients accumulate across batches, so every update after the first is based on a sum of stale gradients, and training degrades or diverges.

---

# Part 7 — Demo day: running it on the lab system

Start preparing this in week 2. Don't leave it to the morning of.

## The important decision

**Do not plan to train anything live in front of him.** Training takes 10–30 minutes, might fail, and the lab machine may not have a GPU, or the right PyTorch, or internet.

**You demo inference only** — load an already-trained model and run it on an image. That takes under a second, works on any CPU, and cannot embarrass you.

## Three layers of backup

Prepare all three. They take a couple of hours total and they mean nothing can go wrong.

### Layer 1 — slides with every chart as a PNG

Your slides contain every result as an image: the gradient charts, the epsilon sweep, the before/after face pair, the defence comparison.

**If every piece of technology in the room fails, you can still give the entire presentation.** This is the one that actually saves you. Do it first.

### Layer 2 — the repo, on a pendrive and on GitHub

Carry a pendrive with:

```
blindspot/
├── notebooks/          all five .ipynb files, with outputs saved
├── models/
│   ├── detector.pth           trained detector weights
│   └── detector_defended.pth  the defended version
├── results/figures/    every PNG
├── app/demo.py         the Gradio demo
├── report/             report + slides as PDF
└── requirements.txt
```

**The `.pth` files are the critical ones.** Those are your trained models — the output of hours of GPU time, in a file of a few megabytes. Download them from `/kaggle/working/` the moment each training run finishes. If you lose those, you retrain.

Note that `models/*.pth` is in your `.gitignore` (they're too big for a normal repo), so the pendrive is their only copy. **Back them up somewhere else too** — Google Drive, email them to yourself, anything.

### Layer 3 — the live demo

On the lab machine:

```bash
git clone https://github.com/HarshitPandey-2021/blindspot.git
cd blindspot
pip install -r requirements.txt
python app/demo.py
```

Copy the `.pth` files from your pendrive into `models/` first — they're not in the repo.

A browser tab opens. Upload a face, drag the epsilon slider, watch the verdict flip from "AI-generated" to "real" as the noise goes up — while the image on screen looks unchanged.

**If the lab has no internet**, `git clone` and `pip install` both fail. So also carry the full folder on the pendrive *with* a `python -m venv` environment already built, or simply fall back to Layer 1. Check the lab's setup a few days before, not on the day.

## What you actually show him, in order

1. **The problem** — one slide. Deepfake detectors are deployed everywhere. Are they trustworthy?
2. **The detector you built** — architecture, accuracy, confusion matrix
3. **The gradient experiments** — the charts from Day 6 and 7. *This is where his vanishing gradient questions will come, so put real time into these slides.*
4. **The attack** — the two identical-looking faces with different verdicts. Your strongest moment.
5. **The epsilon sweep** — the headline chart
6. **The defence** — both curves on one chart, and the honest cost in clean accuracy
7. **The live demo** — if it works. If it doesn't, you already showed everything in slides.
8. **Limitations and future work** — black-box attacks, transferability, JPEG survival. Shows you know where the edges are.

## Before the viva

- Print the epsilon sweep chart on paper. Pointing at paper beats fumbling with a projector.
- Have the lab notebook open in a tab. When he asks "did you try X?", you can show him the entry.
- Rehearse the question bank in README section 11 **out loud**, not in your head. They are different skills.

---

# One last thing

You've been at this two days and you've trained a neural network and measured a phenomenon that stopped the entire field in its tracks for twenty years. You felt lost the whole time.

Feeling lost while the work gets done is the normal condition. It doesn't mean it isn't working.
