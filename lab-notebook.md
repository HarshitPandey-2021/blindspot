# BlindSpot — Lab Notebook

Every experiment, dated. Failures included deliberately.

---
## 2026-10-29 (Day 1) — first model + vanishing gradient
- MLP on MNIST, 784→256→10, Adam 1e-3, batch 128, 3 epochs.
  94.39% → 96.41% → 96.98% test accuracy.
- 20-layer MLP, ONE backward pass, mean |gradient| per layer:
    sigmoid  layer1 1.722e-19  last 2.325e-02  ratio 1.35e17
    relu     layer1 4.328e-11  last 6.420e-04  ratio 1.48e7
  Both straight on a log axis => exponential decay => repeated multiplication (chain rule).
  sigmoid ~8x decay per layer, relu ~2.4x.
- SURPRISE: ReLU did NOT fix it. 1e8 better than sigmoid, still far too weak to learn.
  Default PyTorch init isn't scaled for 20 layers — the weights shrink the signal too.
- Saved: results/figures/grad_profile_20layer.png

## 2026-10-30 (Day 2) — CNN vs MLP
- Same MNIST task, 3 epochs each, Adam 1e-3.
    MLP  97.10%  (203,530 params, ~8s/epoch)   290 errors / 10,000
    CNN  98.70%  (421,642 params, ~8.5s/epoch) 130 errors / 10,000
- CNN has MORE total params but only ~18.8k are convolutional; Linear(3136,128) holds the
  rest. Convolutions do the visual work cheaply; the dense head is the expensive part.
  -> informed the Day 4 decision to use global average pooling instead.

## 2026-10-30 (Day 3) — dataset
- 140k Real/Fake Faces (xhlulu). train 100,000 / valid 20,000 / test 20,000. Balanced.
  classes ['fake','real'], fake=0, real=1. Loaded at 128x128 RGB.
- No training today, deliberately: if the data load is wrong AND training runs, you cannot
  tell which broke.
- Viewed samples: could not reliably tell fake from real by eye.
- NOTE: real images look varied (candid, odd angles); fakes are all centred portraits.
  Possible shortcut learning — model might key on framing, not AI artifacts. Test on Day 5.
- Saved: results/figures/sample_faces.png

## 2026-10-31 (Day 4) — detector trained
- FIRST ATTEMPT WAS 120x TOO SLOW: 10k subset + 20k valid = 970s for one epoch.
  Profiled it — GPU idle. /kaggle/input is network-mounted and the dataset is 140k small
  JPEGs; throughput ~31 images/sec. DISK-BOUND, NOT COMPUTE-BOUND.
  Fix: preload a subset into GPU memory once as uint8, train from RAM.
  Epochs went from ~16 minutes to 8 seconds.
- Also: forgot to print TRAIN accuracy at first, so "can't learn" and "overfitting" were
  indistinguishable. Added.
- v1: 20k @96px, lr 1e-3, 10 epochs -> best valid 0.9137.
  Valid swung +/-30 points between epochs (0.70, 0.59, 0.86, 0.53) while train climbed
  smoothly. Diagnosis: lr too high, model oscillating. Train acc is averaged ACROSS an
  epoch so it hides this; valid is a single snapshot at epoch end.
- v2: 20k @96px, lr 3e-4 + cosine decay + horizontal flips, 25 epochs -> best valid 0.9050.
  DIAGNOSIS CONFIRMED: epochs 1-12 (lr>1.4e-4) still swung; epochs 13-25 climbed smoothly
  0.867 -> 0.905. Stability fixed, peak UNCHANGED => optimisation was not the bottleneck.
- v3: 45k @128px, same settings -> best valid 0.9522 (train 0.9912).
  => data quantity + resolution were the real limit. StyleGAN artifacts are high-frequency
  and 96px was destroying them. Overfit gap 3.9pts (was 7pt at 20k).
- Architecture: 4x [Conv2d(3x3,pad=1) + BatchNorm2d + ReLU + MaxPool2d(2)],
  3->32->64->128->128, then AdaptiveAvgPool2d(1), Flatten, Linear(128,2). 241,794 params.
- LOST both .pth files to a session timeout (persistence was off). Re-ran. Persistence on.
  Rule since: download the .pth the moment training ends.
- Saved: models/detector_v3.pth, results/figures/training_curve_v3.png
- Test set NOT touched.

## 2026-10-31 (Day 5) — analysis
- Confusion matrix, 6,000 validation images (overall 0.9522):
      is fake -> said fake 2,809 | said real   163    (94.52% of fakes caught)
      is real -> said fake   124 | said real 2,904    (95.90% of reals passed)
  Slightly weaker at catching fakes. The two errors are NOT equally costly:
  163 fakes through = verification bypass; 124 reals flagged = innocent users rejected.
- 287 wrong / 6,000. Confident mistakes cluster on OCCLUSIONS and ATYPICAL FACES:
  sunglasses, glasses, hats, wide-open mouth, a young child.
- SHORTCUT-LEARNING TEST (answering the Day 3 worry):
      original        0.9522
      blurred         0.5085   <- collapses to chance
      border removed  0.8553   <- mostly holds
  Blur removes fine texture but preserves layout. Collapse to random => the model reads
  HIGH-FREQUENCY TEXTURE, not composition. Day 3 worry REFUTED.
  Border removal cost ~10pts => background contributes some signal, face carries most.
- PREDICTION, RECORDED TODAY, BEFORE ANY ATTACK CODE WAS WRITTEN:
  FGSM produces a small high-frequency perturbation. This detector depends on
  high-frequency texture. So it should be unusually EASY to fool — accuracy should
  collapse at very small epsilon.
- IMPLIED LIMITATION: JPEG also removes high frequencies, so this detector would likely
  degrade badly on re-compressed images.
- Saved: confusion_matrix.png, confident_mistakes.png

## 2026-10-31 (Day 6) — vanishing gradient, depth sweep
- Gradient profile at init, sigmoid MLPs, one backward pass each:
     5 layers: layer1 1.520e-06  last 3.924e-02  ratio 2.6e4
    10 layers: layer1 3.737e-11  last 2.806e-02  ratio 7.5e8
    20 layers: layer1 1.788e-19  last 4.122e-02  ratio 2.3e20
    40 layers: layer1 3.659e-37  last 1.386e-03  ratio 3.8e39
  DOUBLING DEPTH SQUARES THE RATIO. All lines straight on a log axis.
- FAILED FIRST ATTEMPT (kept deliberately): trained all four depths, every one returned
  exactly 0.5047 = 3028/6000 = "always predict real". All collapsed, including 5 layers.
  WITH NO POSITIVE CONTROL THE RESULT WAS UNINTERPRETABLE — could not separate
  "depth killed it" from "setup broken".
  Cause: 49,152 raw inputs into a sigmoid saturates it even at 5 layers; inputs not centred.
  Also hit CUDA OOM converting 45k images to float in one go — fixed by chunking.
  Fix: 32x32 input (3,072 features), centred to [-1,1], and ALWAYS run a control first.
- SECOND ATTEMPT (8 epochs, plain SGD lr=0.1):
    CONTROL relu 5 layers: 0.7122   <- control passes, setup is sound
    sigmoid 5/10/20/40   : 0.5047   (all = always-predict-real)
    relu   10/20/40      : 0.4953   (all = always-predict-fake, 2972/6000)
  1. Sigmoid fails at EVERY depth, even 5 — predicted by the gradient profile.
  2. ReLU rescues depth 5 only. Necessary but NOT sufficient.
  3. Dead networks collapsed to CONSTANT predictors, not weak learners (checked).
- Used plain SGD, not Adam: Adam adapts step size per weight and so COMPENSATES for tiny
  gradients, masking the phenomenon. SGD multiplies by lr and nothing else.
- Saved: grad_depth_sweep.png, accuracy_by_depth.png

## 2026-10-31 (Day 7) — the four fixes
- Gradient profile, 20-layer networks:
    sigmoid             layer1 3.617e-19  last 1.514e-01  ratio 4.19e17
    relu                layer1 5.945e-11  last 5.811e-04  ratio 9.77e6
    relu + He init      layer1 2.840e-03  last 5.631e-02  ratio 19.8
    relu + He + BN      layer1 1.060e-01  last 8.394e-02  ratio 0.8    <- near-perfect flow
    residual + He       layer1 2.698e+01  last 2.582e+03  ratio 95.7   <- ~1000x TOO LARGE
    residual + BN + He  layer1 2.971e-02  last 4.004e+00  ratio 134.8  <- BN cut it ~900x
- Accuracy (8 epochs, plain SGD lr=0.1, 32x32 centred):
    config              d=5      d=10     d=20     d=40
    sigmoid             0.5047   0.5047   0.5047   0.5047
    relu                0.7122   0.4953   0.4953   0.4953
    relu + He           0.7943   0.6552   0.7840   0.7692
    relu + He + BN      0.8173   0.7972   0.7363   0.5287
    residual + He       0.7732   0.4953   0.4953   0.4953
    residual + BN + He  0.7760   0.7713   0.5995   0.5108
- FINDINGS:
  1. He init is the single biggest win: moves the depth wall from 5 to beyond 40.
  2. BN best shallow (0.8173 @ d=5) and best gradient ratio (0.8), collapses at 40.
  3. EXPLODING GRADIENTS, UNPLANNED: residual-only gradients ~1000x too large.
     y = x + f(x) accumulates signal through the stack; at lr=0.1 every update overshoots.
     Both syllabus failure modes now demonstrated from own runs.
  4. BN on the residual net rescued depth 10 (0.4953 -> 0.7713). This is exactly why real
     ResNets pair skip connections WITH normalisation.
  5. OVERALL WINNER AT DEPTH IS THE SIMPLEST FIX: relu + He init, 0.7692 at 40 layers.
     My prediction that residual+BN would win at depth was WRONG.
  6. Healthy gradient FLOW is necessary but not sufficient — residual-only had a reasonable
     ratio (95.7) and still failed, because the magnitudes were wrong.
- CAVEATS: single seed; lr fixed at 0.1 for every architecture, never tuned per-config;
  8 epochs may be too few for the larger residual+BN net. The He-init dip at d=10 then
  recovery at d=20 is probably run-to-run noise.
- Saved: grad_fixes_20layer.png, depth_wall_by_fix.png

## 2026-11-05 (Days 9-12) — attack and defence
- SANITY CHECK PASSED: baseline clean 0.9522 on 6,000 valid images, matching Day 4.
- FGSM from scratch: x_adv = clamp(x + eps * sign(dLoss/dx), 0, 1).
  Gradient w.r.t. the INPUT, not the weights. sign() bounds the per-pixel change to
  exactly eps, which is what keeps the perturbation invisible.
- Single image (val #0, eps 0.01): fake (0.9995) -> real (1.0000).
  Largest pixel change 2.6/255. Images visually identical.
- EPSILON SWEEP, undefended, 6,000 images:
    eps     (x/255)   FGSM      PGD-10
    0.000   ( 0.0)    0.9522    0.9522
    0.001   ( 0.3)    0.0090    0.0013
    0.002   ( 0.5)    0.0000    0.0000
    0.005   ( 1.3)    0.0000    0.0000
    0.010   ( 2.6)    0.0000    0.0000
    0.020   ( 5.1)    0.0050    0.0000
    0.030   ( 7.6)    0.0545    0.0000
    0.050   (12.8)    0.1517    0.0000
    0.100   (25.5)    0.0110    0.0000
  COLLAPSE TOTAL BY eps=0.002. Not one of 6,000 survived.
  DAY 5 PREDICTION CONFIRMED.
  CAVEAT: no reference model was trained, so "extremely fragile" is supported but
  "unusually fragile" is NOT — that was never measured.
- NON-MONOTONIC FGSM at eps 0.02-0.05: FGSM takes ONE large step and can overshoot the
  adversarial region on a curved loss surface. PGD stays 0.0000 at every one of those
  epsilons, proving the adversarial examples are still there. Property of the ATTACK.
- BOTH DIRECTIONS (FGSM): at 0.002/0.005/0.010 both fake->real and real->fake are 0.0000.
  At 0.020: 0.0101 and 0.0000. Evasion and false accusation are equally available.
- ADVERSARIAL TRAINING: 12 epochs, FGSM eps=0.01 generated from the current model each
  batch, clean + adversarial concatenated, started from detector_v3.pth, 45k images,
  Adam 3e-4 cosine. Clean 0.9522 -> 0.7072. COST OF THE DEFENCE: 24.5 points.
    eps     undefended FGSM   defended FGSM   defended PGD
    0.000   0.9522            0.7072          0.7072
    0.001   0.0090            0.6853          0.6853
    0.002   0.0000            0.6688          0.6673
    0.005   0.0000            0.6173          0.6095
    0.010   0.0000            0.5297          0.5080
    0.020   0.0050            0.3885          0.3193
    0.030   0.0545            0.2730          0.1712
    0.050   0.1517            0.1353          0.0340
    0.100   0.0110            0.0717          0.0013
- READING IT HONESTLY. Absolute: at the trained eps=0.01 the defended model scores 0.5080
  against PGD vs a 0.5047 always-predict-one-class baseline — essentially nothing.
  Skill above chance at eps=0.005: undefended 0.4475 -> -0.5047 (INVERTED, worse than
  random); defended 0.2025 -> 0.1048, retaining ~52% of its skill.
  Robustness is concentrated at and BELOW the trained epsilon and decays above it.
- AT 0% ACCURACY THE DETECTOR IS PERFECTLY WRONG, not merely broken. Invert every answer
  and you get 100%. The attack doesn't blind it; it inverts it.
- NO strong gradient masking: PGD <= FGSM at every eps, accuracy monotonically decreasing,
  accuracy -> ~0 at large eps. Three standard checks, all pass.
- My guide predicted "holds vs FGSM, collapses vs PGD". NOT observed. The FGSM/PGD gap is
  small below 0.01 and only opens beyond it (0.05: 0.1353 vs 0.0340) — expected behaviour
  of a stronger attack outside the trained budget, not a fake defence.
- Saved: attack_exhibit.png, epsilon_sweep.png, fgsm_vs_pgd.png, defence_comparison.png,
  models/detector_defended_final.pth

## 2026-11-05 (Day 13) — diagnostics
- 8-BIT QUANTIZATION (does the attack survive being saved as a real PNG?):
    eps     (x/255)   float     8-bit
    0.001   ( 0.3)    0.0090    0.9522   <- rounds away entirely
    0.002   ( 0.5)    0.0000    0.0000
    0.005   ( 1.3)    0.0000    0.0000
    0.010   ( 2.6)    0.0000    0.0002
    0.020   ( 5.1)    0.0050    0.0047
  => eps=0.005 (1.3/255) is the REAL-WORLD headline: survives 8-bit, still 0% accuracy.
     eps=0.001 is a float-space artifact and must not be quoted.
- FREQUENCY RELIANCE (testing the hypothesis that AT cost clean accuracy by pushing the
  model off high-frequency features):
    sigma   undefended   defended
    0       0.9522       0.7072
    0.5     0.9425       0.7035
    1.0     0.6928       0.6957
    1.5     0.5245       0.6883
    2.0     0.5085       0.6747
    3.0     0.5052       0.6562
  Fraction of accuracy lost at sigma=2.0: undefended 0.466, defended 0.046.
  majority-class baseline on this sample: 0.5047
  => HYPOTHESIS SUPPORTED. The undefended model loses 47% of its accuracy when fine
     texture is removed; the defended model loses 5%. Adversarial training moved the model
     off high-frequency features onto coarser ones — and that migration is what the
     24.5 points of clean accuracy bought.
- Corrected the defence chart title from "restores robustness" to "trades clean accuracy
  for partial robustness" — the original overclaimed.
- Saved: frequency_reliance.png
### 7 Nov — Final test-set evaluation (held out since Day 3, touched once)

Majority-class baseline: 0.5033

| Model | Clean | ε=0.002 FGSM / PGD | ε=0.005 FGSM / PGD | ε=0.010 FGSM / PGD |
|---|---|---|---|---|
| Undefended | 0.9558 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| Defended   | 0.7098 | 0.6703 / 0.6688 | 0.6118 / 0.6040 | 0.5292 / 0.5043 |

Test tracks validation within 0.4 points on every row — no validation overfitting.
PGD ≤ FGSM everywhere, monotonic in ε: no gradient masking.