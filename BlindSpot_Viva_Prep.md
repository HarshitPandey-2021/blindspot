# BlindSpot — Viva Preparation

**Harshit Pandey · Roll No. 2410013155315 · B.Tech CSE (AI) · Deep Learning (NAI-701)**
Guide: Dr. Prem Shanker Yadav, Assistant Professor, Dept. of CSE
Faculty of Engineering & Technology, University of Lucknow

Live demo: https://blindspott.streamlit.app/

---

## The numbers you must know cold

Do not open a laptop to recall these. Memorise the bold ones.

| | Validation (6,000) | **Test (6,000, touched once)** |
|---|---|---|
| Undefended, clean | 0.9522 | **0.9558** |
| Defended, clean | 0.7072 | **0.7098** |
| Undefended, ε = 0.005, FGSM | 0.0000 | **0.0000** |
| Undefended, ε = 0.005, PGD | 0.0000 | **0.0000** |
| Defended, ε = 0.005, PGD | 0.6095 | **0.6040** |

Majority-class baseline: **0.5033**

Full test-set table:

| Model | Clean | ε=0.002 FGSM / PGD | ε=0.005 FGSM / PGD | ε=0.010 FGSM / PGD |
|---|---|---|---|---|
| Undefended | 0.9558 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| Defended | 0.7098 | 0.6703 / 0.6688 | 0.6118 / 0.6040 | 0.5292 / 0.5043 |

**Skill above chance** — the honest way to read a balanced binary task.
Formula: `skill = (accuracy − 0.5033) / (1 − 0.5033)`

| | Accuracy | Skill above chance |
|---|---|---|
| Undefended, clean | 0.9558 | **+91.1%** |
| Undefended, attacked | 0.0000 | **−101.3%** |
| Defended, clean | 0.7098 | +41.6% |
| Defended, ε = 0.005 PGD | 0.6040 | +20.3% |
| Defended, ε = 0.010 PGD | 0.5043 | +0.2% |

At ε = 0.005 the defended model keeps **48.8%** of its own clean skill.
The undefended model goes **past zero into negative** — systematically wrong.

**Other figures:**
- Detector: 241,794 parameters, 4 conv blocks, 128×128 RGB input
- Trained on 45,000 images, 25 epochs, Adam lr 3e-4 + cosine decay
- Dataset: 140k Real and Fake Faces (70k FFHQ photographs, 70k StyleGAN)
- Vanishing gradient: 20-layer sigmoid, layer-1 gradient = **1.8e-19**
- He init at depth 40: **0.7692**. Plain ReLU died at depth 5.
- Blur ablation: undefended 95.2% → **50.9%**. Defended loses only **5%**.
- Training speed fix: **16 minutes → 8 seconds** per epoch (120×)

---

# A. Opening questions

**1. What is your project, in one sentence?**

I trained a CNN that distinguishes real photographs from StyleGAN-generated faces
at 95.6% accuracy, then broke it to 0.00% with a perturbation of about one
brightness level out of 255 — invisible to the eye — and then rebuilt it with
adversarial training to recover roughly half its skill.

**2. Why is this a deep learning project and not just an application?**

Because the subject is the mechanism, not the product. I trained from scratch
rather than fine-tuning so that every weight is mine; I measured vanishing and
exploding gradients directly rather than citing them; and the attack is
backpropagation itself, redirected at the input instead of the weights. The
detector is the instrument, not the result.

**3. What is the practical application?**

Platforms use detectors like this to flag AI-generated images; banks use them to
confirm an uploaded selfie is a real person, not a synthetic face. These systems
exist *because humans cannot make the judgment unaided.* If a detector can be
reversed by a change nobody can see, there is no fallback — the human reviewer
sees an ordinary image, because nothing visible has changed.

**4. Why should I care about an attack that needs the model's weights?**

Two reasons. First, white-box is the correct way to *measure* fragility — it
gives the upper bound on what an attacker can do, and if a model survives
white-box it survives everything weaker. Second, in deployment the weights often
are available: open-source detectors, models shipped inside apps, models exposed
through an API that leaks gradients. Transfer attacks between models are the
black-box version and are a documented phenomenon; I did not test them, so I
don't claim them.

---

# B. The detector

**5. Why train from scratch instead of fine-tuning ResNet-50?**

Because the project is about the mechanism, not the leaderboard. A fine-tuned
backbone would score higher and teach me nothing about initialisation, depth or
gradient flow — and I could not honestly say I understood what every layer was
doing. 95.6% from 241k parameters trained from zero is the more defensible
result for a project graded on understanding.

**6. Describe your architecture.**

Four blocks of `Conv2d(3×3, padding=1) → BatchNorm2d → ReLU → MaxPool2d(2)`,
channels 3 → 32 → 64 → 128 → 128. The image goes 128×128 → 64 → 32 → 16 → 8.
Then global average pooling to a 128-vector, then `Linear(128, 2)`.
241,794 parameters total.

**7. Why global average pooling instead of a dense head?**

I measured it. In an earlier CNN-vs-MLP comparison the convolutions cost about
18,800 parameters and the dense head cost over 400,000 — the head was 95% of the
model and most of the overfitting risk. Global average pooling collapses each
feature map to its mean, which removes the head entirely and makes the model
insensitive to *where* in the frame a feature appears.

**8. Why BatchNorm between the convolution and the ReLU?**

It renormalises each channel to roughly zero mean and unit variance before the
nonlinearity, which keeps activations in the region where the gradient is
healthy, allows a larger learning rate, and reduces the sensitivity to
initialisation. Placing it after the ReLU would normalise an already-rectified,
strictly non-negative distribution, which is less useful. This is the ordering
in the original BatchNorm paper.

**9. Why 3×3 kernels?**

Two stacked 3×3 convolutions cover the same receptive field as one 5×5 but with
fewer parameters (18 vs 25 per channel pair) and two nonlinearities instead of
one. It is the standard result from VGG.

**10. Why 128×128 and not 256 or 64?**

The artefacts I need are high-frequency texture, so I cannot go too low — and in
fact I started at 96px and moved up because of it. 128 was the largest size that
let me hold 45,000 images in GPU memory as uint8, which is what made the
training fast enough to iterate.

**11. Why Adam with cosine decay?**

Adam adapts per-parameter step sizes, which converges fast without a learning
rate search I had no compute budget for. Cosine decay anneals the rate smoothly
to near zero over the run, so the model takes large steps early and settles
precisely late. lr = 3e-4 is the common default for Adam on vision tasks.

**12. What augmentation did you use, and why so little?**

Random horizontal flips only. Anything stronger — colour jitter, blur, JPEG
compression — would destroy the high-frequency texture the detector depends on,
which I later proved with the blur ablation. Augmenting away your own signal is
a real failure mode, and avoiding it here was deliberate.

**13. How do you know it isn't overfitting?**

Three ways. The training and validation curves do not diverge. The test set,
held out from day three and touched exactly once at the end, scored 0.9558 —
*higher* than validation's 0.9522, so nothing leaked. And the model has only
241,794 parameters for 45,000 training images, which is a modest ratio.

**14. Your accuracy is 95%. Is that good?**

It depends on the baseline, which is why I report it. Majority class is 50.33%,
so chance is a coin flip. Measured as skill above chance, 95.58% is 91.1% of the
available skill. Quoting raw accuracy on a balanced binary task without the
baseline is how people make weak results look strong.

**15. What are the model's errors like?**

I did error analysis on the misclassified validation images. The failures
cluster on unusual lighting and heavy occlusion — sunglasses, hands across the
face — where the fine texture the model relies on is either washed out or
hidden. That is consistent with the frequency-reliance finding.

---

# C. Vanishing and exploding gradients

*(The professor emphasises this. Expect several questions. Have the figures open.)*

**16. What is the vanishing gradient problem?**

Backpropagation computes the gradient at layer *k* by multiplying together the
local derivatives of every layer above it — the chain rule. If each of those
factors is less than one, the product shrinks geometrically with depth. The
early layers then receive a gradient so small that their weights barely move, so
a deep network effectively trains only its last few layers.

**17. Why does sigmoid cause it specifically?**

The derivative of the sigmoid is `σ(x)(1 − σ(x))`, which has a maximum of
**0.25** at x = 0 and falls towards zero in both tails. So *in the best case*
every layer multiplies the gradient by a quarter — and in the saturated regions,
by far less. Twenty layers at the optimum alone gives 0.25²⁰ ≈ 1e-12.

**18. Show me that it actually happens in your network.**

In my 20-layer sigmoid network, the gradient magnitude reaching layer 1 is
**1.8e-19** — about 10²⁰ times smaller than at the final layer. The ratio of
first-layer to last-layer gradient is the number I report.

**19. How do you know multiplication is the cause, and not something else?**

Two independent pieces of evidence:

1. **Plot gradient magnitude per layer on a log axis — it is a straight line.**
   A straight line in log space means a constant ratio between consecutive
   layers, and a constant ratio is exactly what repeated multiplication
   produces. No other mechanism gives that shape.
2. **Doubling depth squares the ratio.** 10 layers → 20 layers moves the
   first-layer gradient from roughly 1e-10 to 1e-19. That is the prediction the
   multiplicative model makes before you run it.

**20. What are the standard fixes, and which worked best for you?**

Four: ReLU-family activations, careful initialisation (He/Xavier),
normalisation layers, and skip connections.

**Not the one I predicted.** I recorded a prediction that residual connections
plus batch norm would win at depth. It did not. **He initialisation alone** was
the biggest practical win, carrying a network past 40 layers at 0.7692 accuracy
where plain ReLU died at depth 5. Batch norm gave the best gradient ratio —
about 0.8, nearly perfect — but degraded as depth grew.

**21. Why does He initialisation help?**

Because it sets the initial weight variance to `2/n_in`, which is chosen so that
the variance of the activations is preserved as the signal passes forward
through a ReLU. The factor of 2 compensates for ReLU zeroing half its inputs.
Preserve forward variance and you approximately preserve backward gradient
magnitude too. Xavier uses `1/n_in` and is the equivalent for tanh/sigmoid.

**22. You said residual connections failed. Explain.**

Residual connections *without* normalisation produced the opposite failure:
**exploding** gradients, roughly 1000× too large, and accuracy of 0.5108 at
depth 40 — chance. The reason is that a skip connection adds the input to the
output, so the variance of the signal grows additively at every block. With
nothing to renormalise it, the forward signal and therefore the backward
gradient blow up. This is exactly why real ResNets pair every skip with batch
norm — I reproduced the failure that motivates the pairing.

**23. What is the exploding gradient problem and how is it normally fixed?**

The same chain-rule product, but with factors greater than one, so the product
grows geometrically instead of shrinking. Symptoms are loss going to NaN and
weights diverging. Standard fixes are gradient clipping (rescale the gradient
if its norm exceeds a threshold), normalisation layers, and more careful
initialisation. I did not need clipping because batch norm handled it in the
detector; I saw the explosion only in the deliberately unnormalised residual
experiment.

**24. Why doesn't your own detector suffer from vanishing gradients?**

Three reasons, all deliberate. It uses ReLU, whose derivative is exactly 1 on
the positive side rather than ≤ 0.25. It has batch norm in every block. And it
is only four blocks deep — the problem is a function of depth, and at depth 4
there is nothing to accumulate. The gradient experiments are a separate,
purpose-built study, not a property of the detector.

**25. What is the difference between vanishing gradients and dying ReLU?**

Vanishing gradients are *multiplicative and global* — the signal shrinks as it
propagates through depth, and affects every early-layer weight. Dying ReLU is
*local and permanent* — one unit's pre-activation goes negative for every input
in the dataset, so its gradient is exactly zero forever and the unit never
recovers. Leaky ReLU exists to fix the second, not the first.

**26. Did ReLU solve the problem completely?**

No, and this surprised me — I had predicted it would. At 20 layers a pure ReLU
network still reached 4.328e-11 at layer 1. ReLU removes the ≤ 0.25 ceiling but
it does not control the *scale* of the weights, and with poor initialisation the
product still collapses. That is precisely why He initialisation was the bigger
win. I had the wrong prediction and recorded the correction.

---

# D. The attack

**27. Explain FGSM in your own words.**

`x_adv = clamp( x + ε · sign( ∂Loss/∂x ) , 0, 1 )`

Compute the gradient of the loss with respect to the *input pixels*, take its
sign to get a ±1 per pixel, scale by a small ε, add it to the image, and clamp
back into valid pixel range. It is a single step in the direction that most
increases the model's loss.

**28. How is that different from training? You're still doing backprop.**

Same backpropagation, stopped one step earlier and pointed somewhere else.
Training computes ∂Loss/∂**weights** and steps *down* to reduce loss. This
computes ∂Loss/∂**pixels** and steps *up* to increase it. The weights are frozen;
the image is the variable being optimised. In PyTorch the difference is literally
`x.requires_grad_(True)` instead of the parameters requiring grad.

**29. Why take the sign? Why not use the raw gradient?**

Because the sign bounds every pixel's change to exactly ε regardless of how
steep the gradient is at that pixel. The raw gradient would change a few pixels
enormously and most pixels not at all, and the large changes would be visible.
The uniform bound — formally an L∞ constraint — is what keeps the perturbation
invisible. Invisibility is the entire point.

**30. What is PGD and why is it stronger?**

PGD is iterated FGSM: start from a random point inside the ε-ball, take many
small steps of size α in the sign-gradient direction, and after each step
project back into the ε-ball around the original image and clamp to [0,1]. It is
stronger because the loss surface is curved — one big step can overshoot, while
many small steps follow the curvature. I used 10 steps with α = ε/4.

**31. Your FGSM accuracy goes UP at ε = 0.03–0.05. Is the model recovering?**

No — that is the one-step attack overshooting. A single large step on a curved
loss surface can land past the adversarial region and back into correctly
classified territory. The proof is that **PGD stays at zero across the entire
range.** The adversarial examples are still there; FGSM just stopped finding
them. PGD ≤ FGSM at every ε in my results.

**32. How much noise does it actually take?**

Collapse is total by **ε = 0.002** — half a brightness level out of 255. Not one
of 6,000 images survived. I report ε = 0.005 as the headline because that value
**survives 8-bit rounding**, so it is reproducible in an actual saved PNG rather
than being a floating-point artifact that disappears the moment you save the file.

**33. Does the attack survive being saved as a real image file?**

For 8-bit rounding, yes — I tested it explicitly by quantising the adversarial
image to uint8 and re-evaluating. At ε = 0.005 the verdict still flips; at
ε = 0.001 it does not, because the change rounds away. You can see both in the
live demo with the 8-bit toggle.

**I did not test JPEG**, and I do not claim it. JPEG discards high-frequency
content, which is exactly the band the perturbation occupies, so I would expect
the attack to be substantially weakened. That is the first experiment I would run
next.

**34. Why is 0% worse than 50%?**

Because 50% is a coin flip — no information. **0% is perfect anti-correlation:
the model is systematically, reliably wrong.** Invert every answer and you score
100%. The attack does not blind the detector; it inverts it. In skill-above-chance
terms it is −101%, which is *past* the floor a useless model would occupy.

**35. Does the attack work in both directions?**

Yes, and both matter. Fake → real is **evasion**: synthetic content gets through
the filter. Real → fake is **false accusation**: a genuine photograph of a real
person gets flagged as AI-generated. Both reach 0% at ε = 0.005. The second is
the one people forget, and it is arguably the more harmful.

**36. Isn't this just adding noise?**

No, and the control proves it. Random noise of the same magnitude leaves
accuracy essentially unchanged. This perturbation is *aligned with the gradient*
— it is the single worst direction for this specific model on this specific
image. Same budget, opposite outcome. That alignment is what makes it an attack
rather than corruption.

**37. Is this white-box or black-box, and why does that matter?**

White-box — every attack had full access to the weights. It matters because
white-box gives you the *upper bound* on attacker capability: it is the honest
way to measure fragility. It does not tell you what an attacker without the
weights could do. Transfer attacks are the standard black-box test and I did not
run one, so I make no claim there.

---

# E. The defence

**38. What is adversarial training?**

During each training batch, generate adversarial examples from the *current*
model, then train on those alongside the clean images with their correct labels.
The model is continuously attacked with its own gradients and continuously
corrected, so it learns features that survive the attack. Formally it is the
min-max formulation from Madry et al. — minimise the loss over the worst-case
perturbation inside the ε-ball.

**39. ⚠ At ε = 0.010 — the budget you trained on — your defence scores 0.5043. That is chance. Your defence does not work.**

*(This is the hardest question in the project. Answer it head-on.)*

Correct, and I would rather state it than hide it. I used **FGSM** adversarial
training, not PGD. FGSM-trained models are known to give far weaker robustness
than PGD-trained ones — the one-step attack during training is too easy to
defeat, so the model learns to beat that specific step rather than becoming
genuinely robust. The literature calls the extreme form *catastrophic
overfitting*.

What I did buy was robustness **below** the training budget: at ε = 0.005 the
defended model keeps about half its own clean skill where the undefended model
loses all of it and goes negative. At the training budget itself it is at
chance. The fix is Madry-style PGD adversarial training, which costs roughly
7–10× the compute per epoch, and I did not have that budget.

I also never isolated ε = 0.01 as the correct training budget — it was a single
choice, not a sweep.

**40. Why did clean accuracy drop 24.5 points?**

I had a hypothesis and I tested it rather than asserting it. The hypothesis was
that adversarial training pushes the model off high-frequency features — because
those are exactly what the attack perturbs — and onto coarser, more robust ones.
Test: blur both models and compare. **The undefended model loses 47% under blur;
the defended one loses 5%.** Confirmed. The defended model is not simply worse;
it is looking at different, lower-frequency evidence, and that evidence is less
discriminative on clean images.

**41. How do you know your defence is not gradient masking?**

Gradient masking is when a defence makes gradients uninformative, so attacks
*appear* to fail while the model is still vulnerable. Three checks, all of which
my results pass:

1. **PGD is never weaker than FGSM.** Under masking the multi-step attack
   mysteriously underperforms the one-step one. Mine does not.
2. **Accuracy decreases monotonically with ε.** No unexplained plateaus or
   recoveries in the defended curve.
3. **Accuracy goes to chance at large enough ε.** A masked model stays
   artificially high. Mine reaches 0.5043 at ε = 0.010.

A fourth standard check is a transfer attack from a separately trained model.
I did not run it; I name it as the missing one.

**42. Is the trade-off worth it?**

That is a deployment question, not a research one, and the answer depends on the
threat model. If adversaries are unlikely, 95.6% clean beats 71% clean. If
adversaries are expected, a model at 0% under attack is worse than useless —
it is confidently and systematically wrong — so 71% that degrades gracefully is
the better system. The honest summary is that robustness was purchased with
accuracy, and I report both numbers.

**43. What would you do with more compute?**

Four things, in order: PGD adversarial training instead of FGSM; a sweep over
the training ε rather than a single value; multiple random seeds with confidence
intervals; and a transfer attack from an independently trained model to get a
black-box number.

---

# F. Honesty and limitations

**44. Is this detector unusually fragile, or are all CNNs like this?**

I can only claim **extremely** fragile, not **unusually** fragile. Supporting
"unusually" would require training a reference model and showing mine breaks at
a smaller ε, and I never trained one. Every undefended CNN collapses at
ε = 8/255; mine collapses at ε = 0.002, which is about 0.5/255 — that is
suggestive, but a single model is not a comparison.

**45. You predicted the fragility in advance. Isn't that hindsight?**

No, and it is dated. On **31 October**, before any attack code existed, I
recorded the prediction: because the blur ablation showed the detector depends
on exactly the high-frequency band that FGSM perturbs, it should break at an
unusually small budget. The attack was implemented and run on **5 November** and
confirmed it. The entry is in the lab notebook with the date, which is the whole
reason I kept one.

**46. What did you get wrong?**

Four things, all recorded:

1. I predicted ReLU alone would keep gradients healthy at 20 layers. It did not
   (4.328e-11). He initialisation mattered more than the activation.
2. I predicted residual + batch norm would win at depth. It lost; plain He init
   won (0.7692 vs 0.5108).
3. An early depth experiment returned **identical results at every depth**,
   which made it uninterpretable — with no positive control I could not
   distinguish "depth killed it" from "the setup is broken." I rebuilt it with a
   5-layer ReLU control that *had* to succeed, and it did (0.7122).
4. I over-estimated the defended model's clean accuracy before running it.

**47. Why only one seed?**

Compute. Everything ran on a free Kaggle T4 with a session limit, and a single
training run of the defended model was the expensive item. One seed means I
cannot put error bars on any number, and I do not claim differences smaller than
a couple of points are meaningful.

**48. How do you know the blur result isn't a fluke?**

It is a sweep, not a single point — accuracy falls smoothly as the blur radius
increases, which is what a dependency looks like, rather than jumping at one
value, which is what an artifact looks like. And the defended model, run through
the same sweep, behaves completely differently (5% loss vs 47%), which it would
not if the procedure were simply destroying both models' inputs.

**49. Why did you hold out a test set and only touch it once?**

Because validation accuracy stops being an unbiased estimate the moment you use
it to make decisions — and I made many: architecture, image size, learning rate,
epochs, defence budget. The test set was split off on day three and evaluated
exactly once, on 7 November, after every decision was final. It scored 0.9558
against validation's 0.9522, which is the evidence that nothing leaked.

---

# G. Engineering

**50. Tell me about a problem you had to debug.**

My first training run was **120× slower than it should have been** — 16 minutes
per epoch. I profiled it and found the GPU was mostly idle, which ruled out
compute. The cause was the data pipeline: `/kaggle/input` is a network mount and
the dataset is 140,000 small JPEGs, so reading was capped around 31 images per
second. The job was **disk-bound, not compute-bound.** I restructured it to
decode the images once and hold them in GPU memory as uint8, converting to float
in chunks per batch. Epochs dropped to **8 seconds.** That one change is what
made the rest of the project possible in the time I had.

**51. Why uint8 in GPU memory rather than float?**

Memory. 45,000 images at 3×128×128 is 2.2 GB as uint8 and 8.8 GB as float32 —
the float version OOMs on a 16 GB T4 with the model also resident. I convert to
float per batch, which costs nothing because it is a GPU-local operation.

**52. How did you handle losing work?**

I lost both trained models once to a Kaggle session timeout, which cost about
forty minutes of retraining. After that I downloaded the `.pth` file the moment
training finished, and wrote a single self-contained bootstrap cell that
rediscovers the dataset path, reloads the weights and redefines every helper — so
a dead session costs one cell run rather than a rebuild.

**53. Is the demo your evidence?**

No — the demo is the illustration. It shows one image, and one image proves
nothing. The evidence is the 6,000-image sweep and the held-out test set. If
anyone asks whether one example is enough, that is the answer, and I point at the
epsilon sweep.

---

# H. Curveballs

**54. If I hand you a photo from my phone, will it work?**

Probably not reliably, and that is a genuine limitation rather than a bug. The
model was trained on 128×128 FFHQ and StyleGAN crops — one tight distribution. A
phone selfie differs in resolution, compression, lighting and framing. I would
rather name that than have it discovered live. What I claim is the *mechanism*,
measured on 6,000 held-out images from the distribution it was trained on.

**55. The face on your poster — whose is it?**

Nobody's. It is StyleGAN output, selected from the validation set by label, so no
real person is depicted. That is also the point of the exhibit: you cannot tell,
which is exactly why the detector exists.

**56. The image is still obviously the same person. So what did you break?**

**Identity is not provenance.** The attack does not change *who* is in the
photograph — a face stays the same face and a human sees no difference. It flips
the answer to a different question: *was this image photographed, or
synthesised?* That is precisely the question humans cannot answer unaided, which
is why it was delegated to a machine in the first place. Flipping it removes the
only backstop there was.

**57. Could an attacker do this without access to your model?**

Not with my method as implemented — FGSM needs gradients, which needs weights.
The documented black-box routes are transfer attacks (craft on a substitute
model, hope it transfers) and query-based attacks (estimate gradients from
repeated API calls). I did not test either, so I make no claim about them.

**58. Is publishing this irresponsible?**

Both FGSM and PGD are published, widely taught, and implemented in standard
libraries — I implemented them from scratch rather than inventing them. The
contribution here is a measurement on a specific class of system, plus a
defence. Keeping fragility unmeasured does not make systems safe; it makes them
confidently wrong in ways nobody has checked.

**59. What would you do differently if you started over?**

Record predictions from day one rather than from day eight — the dated
prediction turned out to be the most valuable single entry in the notebook.
Build positive controls into every experiment from the start, because the one
experiment that lacked one was the one that wasted a day. And budget for PGD
adversarial training, since that is the one weak result in the project.

**60. What did you learn?**

Three things that were not on the syllabus. That an uninterpretable result is
worse than a negative one, and a positive control is what separates them. That
profiling beats guessing — the 120× speedup came from measuring where the time
went, not from optimising what I assumed was slow. And that writing down a
prediction before running the experiment changes how you read the output.

---

# Things to do before you walk in

- [ ] Open https://blindspott.streamlit.app/ **ten minutes early** — free hosting
      sleeps when idle and the first visitor waits 30–60 seconds.
- [ ] Record a 30–60 second screen video of the demo working, as a fallback for
      dead wifi.
- [ ] Have these four figures open in tabs: `attack_exhibit.png`,
      `epsilon_sweep.png`, `defence_comparison.png`, and the gradient-vs-layer
      log plot.
- [ ] Know by heart: **95.58 · 0.00 · 70.98 · 60.40 · 0.5033 · 1.8e-19 · 241,794**
- [ ] In the demo, rehearse the three settings: undefended ε=0.005 8-bit on
      (flips), undefended ε=0.001 8-bit on (holds — rounds away), defended
      ε=0.005 (mostly holds).
- [ ] If asked something you did not test, say **"I did not test that, so I
      don't claim it"** — and then say what you *would* run. That sentence will
      earn you more marks than a guess.
