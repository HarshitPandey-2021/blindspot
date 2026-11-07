import glob
import io
import os

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ──────────────────────────────────────────────────────────────────────────
#  Page
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BlindSpot",
    page_icon="◐",
    layout="centered",              # centered reads far better on a phone
    initial_sidebar_state="collapsed",
)

ACCENT = "#e879f9"
GOOD = "#4ade80"
BAD = "#fb7185"

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 780px; }}
  #MainMenu, footer, header [data-testid="stToolbar"] {{ visibility: hidden; }}

  h1, h2, h3 {{ letter-spacing: -.02em; }}

  /* tab strip: let it scroll instead of squashing on narrow screens */
  div[data-baseweb="tab-list"] {{
      overflow-x: auto; gap: .25rem; scrollbar-width: none;
  }}
  div[data-baseweb="tab-list"]::-webkit-scrollbar {{ display: none; }}
  div[data-baseweb="tab-list"] button {{ white-space: nowrap; padding: .4rem .7rem; }}
  div[data-baseweb="tab-list"] button p {{ font-size: .92rem; font-weight: 600; }}

  /* columns stack cleanly instead of crushing */
  div[data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; gap: .75rem; }}
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{ min-width: 240px; }}

  img {{ border-radius: 10px; }}

  .hero {{
      background: linear-gradient(104deg,#6d28d9 0%,#c026d3 58%,#f472b6 100%);
      border-radius: 16px; padding: 18px 20px; margin-bottom: 1rem;
  }}
  .hero h1 {{ font-size: 2.1rem; font-weight: 800; color: #fff; margin: 0; line-height: 1.05; }}
  .hero p  {{ font-size: .95rem; color: #fff; opacity: .93; margin: .5rem 0 0; }}

  .card {{ border-radius: 14px; padding: 13px 15px; margin-top: .5rem; }}
  .card .lbl {{ font-size: .68rem; letter-spacing: .12em; text-transform: uppercase;
                color: #9b8fb0; margin-bottom: 5px; }}
  .card .big {{ font-size: 1.3rem; font-weight: 800; line-height: 1.15; }}
  .card .sub {{ font-size: .84rem; color: #b9aecb; margin-top: 3px; }}

  .flipbar {{ border-radius: 12px; padding: 12px 15px; margin: .9rem 0;
              font-size: .98rem; font-weight: 700; }}

  .note {{ font-size: .85rem; color: #a99cba; line-height: 1.5; }}

  @media (max-width: 640px) {{
      .block-container {{ padding-left: .9rem; padding-right: .9rem; }}
      .hero h1 {{ font-size: 1.6rem; }}
      .hero p  {{ font-size: .85rem; }}
      .card .big {{ font-size: 1.1rem; }}
  }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
#  Model
# ──────────────────────────────────────────────────────────────────────────
CLASSES = ["AI-generated", "Real photograph"]      # fake = 0, real = 1


def _block(c0, c1):
    return nn.Sequential(
        nn.Conv2d(c0, c1, 3, padding=1), nn.BatchNorm2d(c1),
        nn.ReLU(), nn.MaxPool2d(2),
    )


def make_detector():
    return nn.Sequential(
        _block(3, 32), _block(32, 64), _block(64, 128), _block(128, 128),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 2),
    )


def _find(name):
    for p in (f"models/{name}", f"app/models/{name}", name):
        if os.path.exists(p):
            return p
    return None


@st.cache_resource(show_spinner=False)
def load_models():
    out = {}
    for key, fname in [("Undefended", "detector_v3.pth"),
                       ("Defended", "detector_defended_final.pth")]:
        path = _find(fname)
        if path is None:
            continue
        m = make_detector()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        out[key] = m
    return out


MODELS = load_models()

PREP = transforms.Compose([
    transforms.Resize(146),          # resize short side, then crop —
    transforms.CenterCrop(128),      # never squash a non-square photo
    transforms.ToTensor(),
])


def to_tensor(img: Image.Image) -> torch.Tensor:
    return PREP(img.convert("RGB")).unsqueeze(0)


def to_pil(x: torch.Tensor) -> Image.Image:
    a = (x.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(a)


def predict(model, x):
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0]
    i = int(p.argmax())
    return i, float(p[i])


def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x), y)
    model.zero_grad()
    loss.backward()
    return (x + eps * x.grad.sign()).clamp(0, 1).detach()


def quantize(x):
    """Round to a real 8-bit image and back — the honesty check."""
    return (x * 255).round().clamp(0, 255) / 255.0


# ──────────────────────────────────────────────────────────────────────────
#  Header
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>BlindSpot</h1>
  <p>A face detector at 95.6% accuracy, broken to 0.00% by noise you cannot see.</p>
</div>
""", unsafe_allow_html=True)

if not MODELS:
    st.error("No model weights found. Expected `models/detector_v3.pth` in the repo.")
    st.stop()

tab_try, tab_what, tab_res, tab_how = st.tabs(
    ["Try it", "What is this", "Results", "Under the hood"]
)


# ──────────────────────────────────────────────────────────────────────────
#  Tab 1 — the demo
# ──────────────────────────────────────────────────────────────────────────
with tab_try:

    # --- pick an image -----------------------------------------------------
    examples = []
    for d in ("app/examples", "examples", "app/demo_faces", "demo_faces"):
        examples += sorted(glob.glob(os.path.join(d, "*.png")))
        examples += sorted(glob.glob(os.path.join(d, "*.jpg")))
    examples = examples[:6]

    src = None
    if examples:
        choice = st.radio(
            "Image",
            [os.path.splitext(os.path.basename(p))[0].replace("_", " ").title()
             for p in examples] + ["Upload my own"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if choice == "Upload my own":
            up = st.file_uploader("Upload a face", type=["png", "jpg", "jpeg"],
                                  label_visibility="collapsed")
            if up:
                src = Image.open(io.BytesIO(up.read()))
        else:
            src = Image.open(examples[
                [os.path.splitext(os.path.basename(p))[0].replace("_", " ").title()
                 for p in examples].index(choice)
            ])
    else:
        up = st.file_uploader("Upload a face image", type=["png", "jpg", "jpeg"])
        if up:
            src = Image.open(io.BytesIO(up.read()))

    if src is None:
        st.info("Pick an example above, or upload a face, to run the attack.")
        st.stop()

    # --- controls, in the page, not the sidebar ---------------------------
    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        which = st.radio("Model", list(MODELS.keys()), horizontal=True)
    with c2:
        eightbit = st.toggle("Save as a real 8-bit image", value=True)

    eps = st.slider(
        "Attack strength  ε", 0.000, 0.020, 0.005, 0.001,
        format="%.3f",
        help="ε = 0.005 is about 1.3 brightness levels out of 255.",
    )
    st.markdown(
        f"<div class='note'>ε = {eps:.3f} &nbsp;→&nbsp; every pixel moves by at most "
        f"<b>{eps * 255:.1f}</b> of 255 brightness levels.</div>",
        unsafe_allow_html=True,
    )

    # --- run ---------------------------------------------------------------
    model = MODELS[which]
    x = to_tensor(src)
    i0, c0 = predict(model, x)

    if eps > 0:
        x_adv = fgsm(model, x, torch.tensor([i0]), eps)
        if eightbit:
            x_adv = quantize(x_adv)
    else:
        x_adv = x.clone()
    i1, c1v = predict(model, x_adv)

    flipped = i1 != i0

    st.markdown(
        f"""<div class="flipbar" style="background:{(BAD if flipped else GOOD)}1f;
             border:1px solid {(BAD if flipped else GOOD)}55;
             color:{(BAD if flipped else GOOD)}">
             {'⚠ &nbsp;The verdict flipped.' if flipped
              else '✓ &nbsp;The verdict held.'}
           </div>""",
        unsafe_allow_html=True,
    )

    a, b = st.columns(2)
    with a:
        st.image(to_pil(x), width="stretch")
        st.markdown(
            f"""<div class="card" style="background:{ACCENT}14;border:1px solid {ACCENT}33">
                  <div class="lbl">Original</div>
                  <div class="big" style="color:{ACCENT}">{CLASSES[i0]}</div>
                  <div class="sub">{c0:.1%} confidence</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with b:
        st.image(to_pil(x_adv), width="stretch")
        col = BAD if flipped else GOOD
        st.markdown(
            f"""<div class="card" style="background:{col}14;border:1px solid {col}33">
                  <div class="lbl">After attack</div>
                  <div class="big" style="color:{col}">{CLASSES[i1]}</div>
                  <div class="sub">{c1v:.1%} confidence</div>
                </div>""",
            unsafe_allow_html=True,
        )

    with st.expander("Show the perturbation itself"):
        d = (x_adv - x).squeeze(0).permute(1, 2, 0).numpy()
        amp = np.clip(0.5 + d / (2 * max(eps, 1e-6)), 0, 1)
        st.image((amp * 255).astype(np.uint8), width="stretch")
        st.markdown(
            "<div class='note'>Amplified roughly 100×. At true scale this is "
            "invisible — that is the whole point.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("""
<div class="note">
<b>Three things worth trying</b><br>
• Undefended, ε = 0.005, 8-bit <b>on</b> — the verdict flips.<br>
• Undefended, ε = 0.001, 8-bit <b>on</b> — it holds. The change rounds away when
  the image is saved as a real file.<br>
• Defended, ε = 0.005 — it mostly holds its ground.
</div>
""", unsafe_allow_html=True)

    st.caption(
        "One image is an illustration, not evidence. The evidence is the "
        "6,000-image sweep in the Results tab."
    )


# ──────────────────────────────────────────────────────────────────────────
#  Tab 2
# ──────────────────────────────────────────────────────────────────────────
with tab_what:
    st.markdown("""
### The short version

A neural network can tell a real photograph from an AI-generated face with
**95.6% accuracy**. Changing every pixel by at most **one brightness level out
of 255** — a change no human eye can detect — drives it to **0.00%**.

Not 50%, which would be a coin flip. **Zero.** The detector becomes
systematically, reliably wrong. Invert every answer and you would score 100%.

### Why that matters

Platforms use detectors like this to flag AI-generated images. Banks use them
to confirm an uploaded selfie is a real person. These systems exist **because
humans cannot make the judgment unaided** — most people cannot tell a
StyleGAN face from a photograph.

So if the machine's answer can be reversed by a change nobody can see, there is
no backstop. The reviewer looks at the image and sees nothing wrong, because
nothing visible *is* wrong.

### "But it's still the same face"

Correct — and that is the distinction. The attack does not change **who** is in
the photograph. It flips the answer to a different question: *was this image
photographed, or synthesised?*

**Identity is not provenance.** Provenance is precisely the question humans
cannot answer by looking, which is why it was handed to a machine in the first
place. Flipping it removes the only check there was.

### How the attack works

Training a network computes the gradient of the error with respect to the
**weights**, then nudges the weights to reduce it.

This computes the gradient with respect to the **input pixels** instead, and
nudges them the other way — to *increase* the error. Same mathematics, frozen
weights, and the image is the thing being optimised.

Taking only the **sign** of that gradient bounds every pixel to the same tiny
step regardless of how steep the gradient is. That bound is what keeps the
change invisible.
""")


# ──────────────────────────────────────────────────────────────────────────
#  Tab 3
# ──────────────────────────────────────────────────────────────────────────
with tab_res:
    st.markdown("### Final test set — held out from day three, evaluated once")

    m1, m2, m3 = st.columns(3)
    m1.metric("Clean accuracy", "95.58%")
    m2.metric("Under attack", "0.00%", "−95.6 pts", delta_color="inverse")
    m3.metric("Chance", "50.33%")

    st.markdown("""
| Model | Clean | ε=0.002 | ε=0.005 | ε=0.010 |
|---|---|---|---|---|
| Undefended | 0.9558 | 0.0000 | 0.0000 | 0.0000 |
| Defended | 0.7098 | 0.6688 | 0.6040 | 0.5043 |

<div class="note">Attack figures are PGD, the stronger of the two.
6,000 unseen images. Majority-class baseline 0.5033.</div>
""", unsafe_allow_html=True)

    for name, cap in [
        ("results/figures/epsilon_sweep.png",
         "Accuracy against attack strength. Collapse is total by ε = 0.002."),
        ("results/figures/defence_comparison.png",
         "Adversarial training: robustness bought with clean accuracy."),
        ("results/figures/attack_exhibit.png",
         "The same face, opposite verdicts. This person does not exist — "
         "it is StyleGAN output."),
    ]:
        if os.path.exists(name):
            st.markdown("")
            st.image(name, width="stretch")
            st.caption(cap)

    st.markdown("""
---
### Read honestly

**Skill above chance** is the fair measure on a balanced binary task.
Clean, the detector holds **91%** of the available skill. Under attack it holds
**−101%** — past zero, into being reliably wrong.

The defence keeps about **half its own skill** at ε = 0.005, and costs
**24.5 points** of clean accuracy. At ε = 0.010 — the budget it was *trained*
on — it is at chance, because FGSM adversarial training is substantially weaker
than PGD adversarial training.

### What this does not show

- Not that this detector is *more* fragile than others. No reference model was trained.
- Not that the attack survives JPEG. Only 8-bit rounding was tested.
- Not black-box. Every attack had full access to the weights.
- Single seed throughout.
""")


# ──────────────────────────────────────────────────────────────────────────
#  Tab 4
# ──────────────────────────────────────────────────────────────────────────
with tab_how:
    st.markdown("""
### The detector

Four blocks of `Conv2d(3×3) → BatchNorm → ReLU → MaxPool`, channels
3 → 32 → 64 → 128 → 128, image 128×128 down to 8×8, then global average
pooling and one linear layer. **241,794 parameters**, trained from scratch —
not fine-tuned — on 45,000 images for 25 epochs.

Global average pooling came out of a measurement: in an earlier comparison the
convolutions cost ~18,800 parameters while a dense head cost over 400,000.
Pooling removes the head entirely.

### The attack
""")
    st.code("x_adv = clamp( x + eps * sign( dLoss/dx ), 0, 1 )", language="text")
    st.markdown("""
FGSM and PGD both implemented from scratch. PGD is iterated FGSM, projected
back into the ε-ball after every step.

### Vanishing gradients, measured rather than cited

In a 20-layer sigmoid network the gradient reaching layer 1 is **1.8e-19** —
about 10²⁰ times weaker than at the last layer. Plotted on a log axis it is a
straight line, and a straight line in log space *is* the proof that repeated
multiplication is the cause.

Of the four standard fixes, **He initialisation alone** was the biggest
practical win, carrying a network past 40 layers where plain ReLU died at 5.
Residual connections *without* normalisation produced the opposite failure —
**exploding** gradients ~1000× too large — which is exactly why real ResNets
pair skips with batch norm.

### One engineering note

The first training run was **120× too slow** — 16 minutes an epoch. Profiling
showed the GPU idle: the job was disk-bound, not compute-bound, reading 140,000
small files off a network mount at ~31 images a second. Holding the data in GPU
memory as uint8 cut epochs to **8 seconds**.

---
<div class="note">
Harshit Pandey · Roll No. 2410013155315 · B.Tech CSE (AI)<br>
Deep Learning (NAI-701) · Faculty of Engineering &amp; Technology,
University of Lucknow<br>
Guide: Dr. Prem Shanker Yadav, Assistant Professor
</div>
""", unsafe_allow_html=True)