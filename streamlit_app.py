"""BlindSpot — fooling an AI-generated-face detector. Streamlit build."""

import base64, io, os
import numpy as np
import streamlit as st
import torch, torch.nn as nn
from PIL import Image
from torchvision import transforms

st.set_page_config(page_title="BlindSpot", page_icon="◍", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["AI-GENERATED", "REAL PHOTO"]
UNDEF, DEF = "Undefended · 95.2% accurate", "Defended · 70.7% accurate"


def find(fname, dirs):
    for d in dirs:
        p = os.path.join(HERE, d, fname)
        if os.path.exists(p):
            return p
    return None


def block(c0, c1):
    return nn.Sequential(nn.Conv2d(c0, c1, 3, padding=1), nn.BatchNorm2d(c1),
                         nn.ReLU(), nn.MaxPool2d(2))

def make_detector():
    return nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 2))


@st.cache_resource
def load_models():
    out = {}
    for key, fname in (("undef", "detector_v3.pth"),
                       ("def", "detector_defended_final.pth")):
        p = find(fname, ("models", os.path.join("app", "models"), "."))
        if p:
            m = make_detector(); m.load_state_dict(torch.load(p, map_location="cpu"))
            out[key] = m.eval()
    return out

MODELS = load_models()
if "undef" not in MODELS:
    st.error("detector_v3.pth not found in models/ — commit it to the repo."); st.stop()

TF = transforms.Compose([transforms.Resize(146), transforms.CenterCrop(128),
                         transforms.ToTensor()])


def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x), y)
    model.zero_grad(); loss.backward()
    return (x + eps * x.grad.sign()).clamp(0, 1).detach()


def arr(t, size):
    a = (t.detach().permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
    return np.array(Image.fromarray(a).resize((size, size), Image.LANCZOS))


def b64(a):
    buf = io.BytesIO(); Image.fromarray(a).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


CSS = """
<style>
#MainMenu, footer, header {visibility:hidden;}
.block-container {padding-top:2rem; max-width:1280px;}

.hero {text-align:center; padding:6px 0 10px;}
.hero h1 {font-size:3.4rem; font-weight:860; margin:0; letter-spacing:-.04em;
  background:linear-gradient(103deg,#fbcfe8 0%,#f472b6 32%,#c026d3 68%,#7c3aed 100%);
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent;}
.hero p {margin:12px auto 0; max-width:620px; font-size:1.06rem; line-height:1.6; opacity:.66;}

.stage {border-radius:24px; padding:30px 26px 24px;
  background:linear-gradient(160deg,#120b1c,#1a0f27);
  border:1px solid rgba(236,72,153,.16);}
.stage.empty {text-align:center; padding:80px 30px;}
.eicon {font-size:2.6rem; color:#c026d3; opacity:.5;}
.etitle {font-size:1.3rem; font-weight:800; margin-top:14px;}
.esub {font-size:1rem; opacity:.55; margin-top:8px;}

.srow {display:flex; align-items:center; justify-content:center; gap:26px; flex-wrap:wrap;}
.arrow {font-size:2rem; color:#e879f9; opacity:.55; margin-top:-56px;}
.shot {margin:0; text-align:center;}
.shot img {width:300px; height:300px; border-radius:18px; display:block;
  border:1px solid rgba(236,72,153,.2); box-shadow:0 18px 48px rgba(0,0,0,.5);}
.shot figcaption {font-size:.68rem; letter-spacing:.2em; font-weight:700;
  opacity:.45; margin:14px 0 10px;}

.vd {border-radius:15px; padding:14px 10px; width:300px;}
.vd .vn {display:block; font-size:1.28rem; font-weight:850;}
.vd .vc {display:block; font-size:.8rem; opacity:.72; margin-top:5px; font-weight:500;}
.vf {background:linear-gradient(142deg,#2c0613,#5c0f2b); color:#ffa7bf;
     border:1px solid rgba(255,100,140,.28);}
.vr {background:linear-gradient(142deg,#032a1b,#0a4a2f); color:#58efab;
     border:1px solid rgba(60,230,160,.24);}

.sfoot {display:flex; align-items:center; gap:18px; margin-top:28px; padding-top:22px;
  border-top:1px solid rgba(236,72,153,.12); flex-wrap:wrap;}
.noise {width:104px; height:104px; border-radius:13px; flex:none;
  border:1px solid rgba(236,72,153,.2);}
.meta {flex:none; min-width:150px;}
.meta .mn {display:block; font-size:1.9rem; font-weight:860; line-height:1;
  background:linear-gradient(95deg,#fbcfe8,#e879f9);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;}
.meta .mo {font-size:.9rem; opacity:.5;}
.meta .mt {display:block; font-size:.64rem; letter-spacing:.14em; opacity:.5; margin-top:7px;}
.vline {flex:1; min-width:260px; border-radius:15px; padding:15px 19px;
  font-size:1rem; line-height:1.6;}
.vline.flip {background:linear-gradient(135deg,#2e0a33,#52103f); color:#ffd9ef;
  border:1px solid rgba(236,72,153,.26);}
.vline.hold {background:linear-gradient(135deg,#120e22,#1a1230); color:#cfc2ea;
  border:1px solid rgba(236,72,153,.14);}

.callout {border-radius:18px; padding:17px 23px; margin:14px 0;
  border:1px solid rgba(236,72,153,.22);
  background:linear-gradient(135deg,rgba(236,72,153,.12),rgba(124,58,237,.09));}
.prose {font-size:1.03rem; line-height:1.78;}
.prose h2 {font-size:1.5rem; margin:30px 0 8px; font-weight:830;
  background:linear-gradient(95deg,#fbcfe8,#c026d3);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;}
.prose .lead {font-size:1.2rem; line-height:1.7;}
.prose code {background:rgba(236,72,153,.12); padding:2px 7px; border-radius:6px;}
.stat {display:inline-block; border-radius:18px; padding:18px 26px; margin:7px 10px 7px 0;
  border:1px solid rgba(236,72,153,.18); background:linear-gradient(142deg,#170d22,#241036);}
.stat .n {display:block; font-size:2rem; font-weight:860; line-height:1.05;
  background:linear-gradient(95deg,#fbcfe8,#e879f9);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;}
.stat .t {display:block; font-size:.7rem; opacity:.58; margin-top:6px; letter-spacing:.12em;}
.cap {text-align:center; font-size:.68rem; letter-spacing:.2em; font-weight:700;
  opacity:.42; margin:8px 0 22px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("<div class='hero'><h1>BlindSpot</h1><p>A program that spots AI-generated faces "
            "95% of the time — and the invisible noise that makes it confidently wrong.</p>"
            "</div>", unsafe_allow_html=True)


# ───── sidebar controls ─────
with st.sidebar:
    st.markdown("### Controls")
    up = st.file_uploader("Upload a face", type=["png", "jpg", "jpeg"])
    st.caption("Faces only · 128×128 crops")
    choices = [UNDEF] + ([DEF] if "def" in MODELS else [])
    which = st.radio("Model", choices, index=0)
    eps = st.slider("Attack strength (ε)", 0.0, 0.05, 0.005, 0.001,
                    help="0.005 ≈ 1 brightness level out of 255")
    quant = st.checkbox("Round to 8-bit, like a real PNG", value=True)
    st.markdown("---")
    st.caption("Both pictures show what the **model** sees: 128×128, upscaled for display. "
               "Uploads are processed in memory and never stored.")

EX_DIR = find("", ("examples", os.path.join("app", "examples"))) or ""
EXAMPLES = (sorted(os.path.join(EX_DIR, f) for f in os.listdir(EX_DIR)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
            if os.path.isdir(EX_DIR) else [])

if "chosen" not in st.session_state:
    st.session_state.chosen = EXAMPLES[0] if EXAMPLES else None

tab1, tab2, tab3, tab4 = st.tabs(["Try it", "What is this?", "The results", "Under the hood"])


def show(name, caption):
    p = find(name, (os.path.join("results", "figures"), os.path.join("app", "figures")))
    if p:
        st.image(p, use_container_width=True)
        st.markdown(f"<div class='cap'>{caption}</div>", unsafe_allow_html=True)


with tab1:
    if EXAMPLES:
        st.markdown("**Pick a face**")
        cols = st.columns(min(len(EXAMPLES), 8))
        for c, path in zip(cols, EXAMPLES):
            with c:
                st.image(path, use_container_width=True)
                if st.button("Use this", key=path, use_container_width=True):
                    st.session_state.chosen = path

    img = None
    if up is not None:
        img = Image.open(up)
    elif st.session_state.chosen:
        img = Image.open(st.session_state.chosen)

    if img is None:
        st.markdown("<div class='stage empty'><div class='eicon'>◍</div>"
                    "<div class='etitle'>Upload a face to begin</div>"
                    "<div class='esub'>Then drag <b>Attack strength</b> in the sidebar.</div>"
                    "</div>", unsafe_allow_html=True)
    else:
        model = MODELS["def"] if (which == DEF and "def" in MODELS) else MODELS["undef"]
        short = "defended" if which == DEF else "undefended"

        x = TF(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            p0 = torch.softmax(model(x), 1)[0]
        y = p0.argmax().unsqueeze(0)
        xa = fgsm(model, x, y, eps) if eps > 0 else x.clone()
        if quant:
            xa = (xa * 255).round().clamp(0, 255) / 255.
        with torch.no_grad():
            p1 = torch.softmax(model(xa), 1)[0]

        d = (xa - x)[0]
        spread = (d.max() - d.min()).item()
        pert = (d - d.min()) / spread if spread > 1e-12 else torch.full_like(d, .5)
        change = (xa - x).abs().max().item() * 255
        i0, i1 = int(p0.argmax()), int(p1.argmax())

        if eps == 0:
            line, lc = "Attack strength is zero — nothing has been changed yet.", "hold"
        elif i0 != i1:
            line, lc = (f"<b>The verdict flipped.</b> Every pixel moved by at most "
                        f"<b>{change:.1f} of 255</b> brightness levels — far below what your "
                        f"eye can resolve — and the {short} detector reversed its answer."), "flip"
        else:
            line, lc = (f"<b>The verdict held.</b> Pixels moved by up to {change:.1f}/255 and "
                        f"the {short} detector kept its answer."), "hold"

        def shot(src, cap, idx, conf):
            cls = "vf" if idx == 0 else "vr"
            return (f"<figure class='shot'><img src='{src}'><figcaption>{cap}</figcaption>"
                    f"<div class='vd {cls}'><span class='vn'>{NAMES[idx]}</span>"
                    f"<span class='vc'>{conf*100:.1f}% confident</span></div></figure>")

        st.markdown(f"""<div class='stage'><div class='srow'>
{shot(b64(arr(x[0],300)),  'ORIGINAL',         i0, p0.max().item())}
<div class='arrow'>→</div>
{shot(b64(arr(xa[0],300)), 'AFTER THE ATTACK', i1, p1.max().item())}
</div><div class='sfoot'>
<img class='noise' src='{b64(arr(pert,104))}'>
<div class='meta'><span class='mn'>{change:.1f}<span class='mo'>/255</span></span>
<span class='mt'>LARGEST PIXEL CHANGE &nbsp;·&nbsp; ε {eps:.3f}</span></div>
<div class='vline {lc}'>{line}</div></div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='callout'><b>Three things to try.</b> (1) ε <b>0.005</b> on the "
                "undefended model — the verdict usually flips. (2) Drop to <b>0.001</b> — it "
                "holds, because a change that small vanishes when the image is saved as a real "
                "file. (3) Switch to the defended model at 0.005 — it mostly stands firm.</div>",
                unsafe_allow_html=True)

with tab2:
    st.markdown("""<div class='prose'>
<p class='lead'>Some of the faces you see online were never photographed. A computer drew them.
This project is about the programs that try to tell the difference — and how easily they break.</p>
<h2>Step 1 — Build a detector</h2>
<p>Trained on <b>140,000 face pictures</b>: half real photographs, half generated by StyleGAN.
It tells them apart correctly <b>95 times out of 100</b> on faces it has never seen.</p>
</div>""", unsafe_allow_html=True)
    show("sample_faces.png", "TOP ROW: AI-GENERATED · BOTTOM ROW: REAL PHOTOGRAPHS")
    st.markdown("""<div class='prose'><h2>Step 2 — Break it</h2>
<p>Change a picture by about <b>one brightness level out of 255</b> — less than the gap between
two shades that look identical on screen — and the detector reverses its answer. Not "becomes
unsure". <b>Reverses.</b></p></div>""", unsafe_allow_html=True)
    show("attack_exhibit.png", "SAME FACE · OPPOSITE VERDICTS")
    st.markdown("""<div class='prose'>
<div class='callout'><b>How is that possible?</b> The detector doesn't look at a face the way you
do. It reads tiny pixel patterns — the faint fingerprints image generators leave behind. Those
are invisible to you, so noise that is also invisible to you can scribble all over them.</div>
<h2>Step 3 — Fix it</h2>
<p>Adversarial training: keep showing the program attacked pictures while it learns. It works —
partly, and not for free. The defended version resists the attack far better but is a worse
detector on ordinary images, dropping from 95.2% to 70.7%.</p>
<h2>Why it matters</h2>
<p>Platforms use detectors like this to flag AI images; banks use them to check the selfie you
upload is a real person. And it works <b>both ways</b> — you can push a fake through as real, or
push a <b>real person's genuine photo</b> into being flagged.</p></div>""",
                unsafe_allow_html=True)

with tab3:
    st.markdown("""<div class='prose'><div>
<span class='stat'><span class='n'>95.22%</span><span class='t'>DETECTOR ACCURACY</span></span>
<span class='stat'><span class='n'>0.00%</span><span class='t'>UNDER ATTACK</span></span>
<span class='stat'><span class='n'>1 / 255</span><span class='t'>PIXELS CHANGED BY</span></span>
<span class='stat'><span class='n'>6,000</span><span class='t'>IMAGES TESTED</span></span></div>
<h2>How much noise does it take?</h2>
<p>Collapse is total by <b>ε 0.002</b>, half a brightness level: not one of 6,000 images
survived. At <b>0.005</b>, which still survives being saved as a real 8-bit file, accuracy is
still exactly zero.</p></div>""", unsafe_allow_html=True)
    show("epsilon_sweep.png", "ACCURACY VS ATTACK STRENGTH")
    st.markdown("""<div class='prose'><div class='callout'><b>Worth pausing on.</b> At 0%
accuracy the detector is not merely broken — it is <b>perfectly wrong</b>. Invert every answer
and you get 100%. The attack doesn't blind it; it inverts it.</div>
<h2>Two attacks, same budget</h2>
<p>FGSM takes one big step; PGD takes ten small ones and is strictly stronger. The FGSM line
rising on the right is the <i>attack</i> overshooting, not the model recovering — PGD stays flat
at zero throughout.</p></div>""", unsafe_allow_html=True)
    show("fgsm_vs_pgd.png", "FGSM VS PGD")
    st.markdown("""<div class='prose'><h2>Does the defence work?</h2>
<p>Partly. Below the budget it was trained on, the defended model keeps roughly <b>half its
skill</b> where the undefended one loses all of it — and it costs 24.5 points of clean
accuracy.</p></div>""", unsafe_allow_html=True)
    show("defence_comparison.png", "DEFENDED VS UNDEFENDED")
    st.markdown("""<div class='prose'><h2>What is it actually looking at?</h2>
<p>Blurring the images — destroying fine texture, keeping the face — dropped accuracy from
95.2% to <b>50.9%</b>, a coin flip. That let me <b>predict in advance</b> that it would break at
an unusually small budget. Recorded 31 October, confirmed 5 November.</p></div>""",
                unsafe_allow_html=True)
    show("frequency_reliance.png", "ACCURACY AS FINE DETAIL IS REMOVED")
    show("confusion_matrix.png", "WHICH WAY IT FAILS")

with tab4:
    st.markdown("""<div class='prose'><h2>The detector</h2>
<p>A CNN with <b>241,794 parameters</b>. Four blocks of <code>Conv2d(3×3) → BatchNorm → ReLU →
MaxPool</code>, 128×128 down to 8×8, channels 3 → 32 → 64 → 128 → 128, then global average
pooling and one linear layer. Adam, lr 3e-4 with cosine decay, horizontal flips, 25 epochs,
45,000 images.</p></div>""", unsafe_allow_html=True)
    show("training_curve_v3.png", "TRAINING AND VALIDATION ACCURACY")
    st.markdown("""<div class='prose'><h2>The attack, in one line</h2>
<p style='font-family:ui-monospace,monospace;font-size:1.06rem;padding:16px 20px;
border-radius:14px;background:rgba(236,72,153,.1);border:1px solid rgba(236,72,153,.22)'>
x_adv = clamp( x + ε · sign( ∂Loss/∂x ) , 0, 1 )</p>
<p>Training computes the gradient of the loss with respect to the <b>weights</b>. This computes
it with respect to the <b>input pixels</b> and steps the other way. Taking the <code>sign</code>
bounds every pixel to exactly ε — and that bound is what keeps the change invisible.</p>
<h2>Vanishing gradients</h2>
<p>In a 20-layer sigmoid network the signal reaching layer 1 is <code>1.8e-19</code> — about
10²⁰ times weaker than at the last layer. Every line is straight on a log axis, which is itself
the proof.</p></div>""", unsafe_allow_html=True)
    show("grad_depth_sweep.png", "GRADIENT STRENGTH PER LAYER, FOUR DEPTHS")
    st.markdown("""<div class='prose'><p><b>He initialisation alone was the biggest practical
win</b>, carrying a network past 40 layers where plain ReLU died at 5. Residual connections
without normalization produced the opposite failure — <b>exploding</b> gradients ~1000× too
large — which is why real ResNets pair skips with batch norm.</p></div>""",
                unsafe_allow_html=True)
    show("grad_fixes_20layer.png", "GRADIENT FLOW UNDER EACH FIX")
    show("depth_wall_by_fix.png", "HOW DEEP EACH FIX LETS YOU GO")
    st.markdown("""<div class='prose'><h2>What this does not show</h2>
<div class='callout'><ul style='margin:0;padding-left:20px;line-height:1.8'>
<li>That this detector is <i>more</i> fragile than others — no reference model was trained.</li>
<li>That the attack survives JPEG. Only 8-bit rounding was tested.</li>
<li>That it works black-box. Every attack had full access to the weights.</li>
<li>That ε = 0.01 was the right training budget for the defence.</li></ul></div>
<h2>Things that went wrong</h2>
<p>The first training run was 120× too slow — profiling found it disk-bound, not compute-bound.
An early depth experiment had no positive control and was uninterpretable. And one prediction —
that residual + batch norm would win at depth — was wrong; plain He initialisation beat it.</p>
<h2>Built with</h2>
<p>PyTorch · Kaggle T4 · "140k Real and Fake Faces" · FGSM and PGD implemented from scratch.</p>
</div>""", unsafe_allow_html=True)