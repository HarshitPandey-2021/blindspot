import base64
import glob
import io
import os
import time

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ══════════════════════════════════════════════════════════════════════════
#  Page
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="BlindSpot", page_icon="◐",
                   layout="wide", initial_sidebar_state="collapsed")

# Material-3 dark lab palette
C = dict(
    surface="#10131c", lowest="#0b0e17", low="#181b25", mid="#1c1f29",
    high="#272a33", highest="#32343f",
    primary="#ffb0cd", primaryc="#f751a1", secondary="#d0bcff",
    tertiary="#4cd7f6", error="#ffb4ab",
    on="#e0e2ef", onvar="#a99cba", outline="#574048",
)

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap" rel="stylesheet">
<style>
  .stApp {{ background: {C['surface']}; }}
  .block-container {{ padding: 1rem 1.2rem 3rem; max-width: 1400px; }}
  #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none; }}

  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {C['on']}; }}
  h1, h2, h3, h4 {{ font-family: 'Plus Jakarta Sans', sans-serif !important;
                    letter-spacing: -.02em; color: {C['on']} !important; }}

  .mono {{ font-family: 'JetBrains Mono', monospace; font-size: 11px;
           letter-spacing: .04em; text-transform: uppercase; }}

  /* panels */
  .panel {{ background: {C['low']}; border-radius: 12px; padding: 14px 16px;
            margin-bottom: 12px; box-shadow: 0 2px 16px rgba(0,0,0,.35); }}
  .panel h4 {{ font-size: 15px; font-weight: 700; margin: 0 0 10px; }}

  /* telemetry pills */
  .pills {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
  .pill {{ background: {C['mid']}; border-radius: 8px; padding: 7px 11px;
           display: flex; flex-direction: column; min-width: 128px; flex: 1; }}
  .pill .k {{ font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
              letter-spacing: .09em; color: {C['onvar']}; text-transform: uppercase; }}
  .pill .v {{ font-family: 'JetBrains Mono', monospace; font-size: 13px;
              font-weight: 700; margin-top: 2px; }}

  .hero {{ background: {C['low']}; border-radius: 14px; padding: 20px 22px;
           position: relative; overflow: hidden; margin-bottom: 14px; }}
  .hero::after {{ content:''; position:absolute; right:-90px; top:-90px;
                  width:300px; height:300px; border-radius:50%;
                  background:{C['primaryc']}1a; filter: blur(60px); }}
  .hero h1 {{ font-size: 30px; font-weight: 700; line-height: 1.22; margin: 10px 0 8px;
              position: relative; z-index: 1; }}
  .hero p {{ font-size: 13px; color: {C['onvar']}; margin: 0; max-width: 780px;
             position: relative; z-index: 1; line-height: 1.55; }}
  .tag {{ display:inline-flex; align-items:center; gap:6px; background:{C['high']};
          color:{C['primary']}; border-radius:999px; padding:3px 10px; }}
  .dot {{ width:6px; height:6px; border-radius:50%; background:{C['primaryc']}; }}

  /* verdict banner */
  .verdict {{ border-radius: 12px; padding: 13px 16px; margin: 12px 0;
              display: flex; align-items: center; gap: 12px; }}
  .verdict .t {{ font-family:'Plus Jakarta Sans',sans-serif; font-size: 17px; font-weight: 800; }}
  .verdict .s {{ font-size: 12.5px; color: {C['onvar']}; margin-top: 2px; line-height:1.45; }}

  /* probability rows */
  .prow {{ margin-bottom: 12px; }}
  .prow .hd {{ display:flex; justify-content:space-between; align-items:center;
               font-family:'JetBrains Mono',monospace; font-size:11px; margin-bottom:5px; }}
  .track {{ height: 9px; border-radius: 999px; background: {C['highest']}; overflow:hidden; }}
  .fill {{ height:100%; border-radius:999px; transition: width .3s; }}
  .was {{ color:{C['onvar']}; text-decoration: line-through; opacity:.65; margin-right:6px; }}

  .formula {{ background:{C['lowest']}; border-radius:10px; padding:12px 14px;
              font-family:'JetBrains Mono',monospace; font-size:13.5px;
              overflow-x:auto; white-space:nowrap; }}

  .foot {{ font-family:'JetBrains Mono',monospace; font-size:10px; color:{C['onvar']};
           letter-spacing:.06em; padding-top:14px; border-top:1px solid {C['outline']};
           margin-top:22px; display:flex; flex-wrap:wrap; gap:14px; }}

  /* streamlit widget restyling */
  div[data-baseweb="tab-list"] {{ gap:4px; background:{C['lowest']}; padding:4px;
      border-radius:10px; overflow-x:auto; scrollbar-width:none; }}
  div[data-baseweb="tab-list"]::-webkit-scrollbar {{ display:none; }}
  div[data-baseweb="tab-list"] button {{ white-space:nowrap; border-radius:7px; padding:.35rem .8rem; }}
  div[data-baseweb="tab-list"] button p {{ font-size:13px; font-weight:600; }}
  div[data-baseweb="tab-list"] button[aria-selected="true"] {{ background:{C['high']}; }}
  div[data-baseweb="tab-list"] button[aria-selected="true"] p {{ color:{C['primary']} !important; }}

  div[data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; gap: 12px; }}
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{ min-width: 300px; }}

  .stSlider [data-baseweb="slider"] div[role="slider"] {{ background: {C['primaryc']} !important; }}
  .stRadio label p, .stCheckbox label p {{ font-size:13px; }}
  [data-testid="stMetricValue"] {{ font-family:'JetBrains Mono',monospace; }}

  table {{ font-family:'JetBrains Mono',monospace !important; font-size:12px !important; }}

  @media (max-width: 680px) {{
      .block-container {{ padding-left:.7rem; padding-right:.7rem; }}
      .hero h1 {{ font-size: 21px; }}
      .pill {{ min-width: 46%; }}
  }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════════════════
CLASSES = ["AI-generated", "Real photograph"]      # fake = 0, real = 1


def _blk(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b),
                         nn.ReLU(), nn.MaxPool2d(2))


def make_detector():
    return nn.Sequential(_blk(3, 32), _blk(32, 64), _blk(64, 128), _blk(128, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 2))


def _find(name):
    for p in (f"models/{name}", f"app/models/{name}", name):
        if os.path.exists(p):
            return p
    return None


@st.cache_resource(show_spinner=False)
def load_models():
    out = {}
    for key, fn in [("Undefended", "detector_v3.pth"),
                    ("Defended", "detector_defended_final.pth")]:
        p = _find(fn)
        if p:
            m = make_detector()
            m.load_state_dict(torch.load(p, map_location="cpu"))
            m.eval()
            out[key] = m
    return out


MODELS = load_models()
N_PARAMS = sum(p.numel() for p in make_detector().parameters())

PREP = transforms.Compose([transforms.Resize(146),
                           transforms.CenterCrop(128),
                           transforms.ToTensor()])


def to_tensor(im):
    return PREP(im.convert("RGB")).unsqueeze(0)


def to_pil(x):
    a = (x.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(a)


def b64(im, size=420):
    im = im.resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def predict(model, x):
    t0 = time.perf_counter()
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0]
    ms = (time.perf_counter() - t0) * 1000
    return p.numpy(), ms


def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x), y)
    model.zero_grad()
    loss.backward()
    return (x + eps * x.grad.sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps, steps=10):
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


def quantize(x):
    return (x * 255).round().clamp(0, 255) / 255.0


# ══════════════════════════════════════════════════════════════════════════
#  Split-lens comparison (real images, drag to scrub)
# ══════════════════════════════════════════════════════════════════════════
def split_lens(clean_im, adv_im, left_label, left_conf, right_label, right_conf, flipped):
    rc = C["error"] if flipped else C["tertiary"]
    return f"""
<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:{C['lowest']};font-family:'JetBrains Mono',monospace}}
  #stage{{position:relative;width:100%;max-width:430px;aspect-ratio:1/1;
         border-radius:12px;overflow:hidden;background:{C['lowest']};
         user-select:none;cursor:ew-resize;margin:0 auto;touch-action:none}}
  #stage img{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;
             display:block;image-rendering:auto}}
  #cleanwrap{{position:absolute;inset:0;width:50%;overflow:hidden;z-index:10}}
  #cleanwrap .inner{{position:absolute;top:0;left:0;height:100%;width:100vw}}
  .hud{{position:absolute;top:9px;z-index:25;display:flex;flex-direction:column;gap:3px;
        pointer-events:none;font-size:9.5px;letter-spacing:.04em;max-width:52%}}
  .hud.l{{left:9px;align-items:flex-start}}
  .hud.r{{right:9px;align-items:flex-end}}
  .hud .b{{background:{C['lowest']}ef;backdrop-filter:blur(6px);border-radius:7px;
           padding:4px 8px;display:flex;align-items:center;gap:5px;font-weight:700;
           text-transform:uppercase;box-shadow:0 2px 10px rgba(0,0,0,.55);
           white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%}}
  .hud .c{{background:{C['high']};border-radius:5px;padding:2px 6px;color:{C['on']};
           font-size:9.5px;white-space:nowrap}}
  .sq{{width:7px;height:7px;border-radius:50%}}
  #handle{{position:absolute;top:0;bottom:0;z-index:30;width:2px;left:50%;
           transform:translateX(-50%);background:{C['primaryc']};
           box-shadow:0 0 12px {C['primaryc']}cc}}
  #knob{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
         width:30px;height:30px;border-radius:50%;background:{C['primaryc']};
         color:#2a0016;display:flex;align-items:center;justify-content:center;
         font-weight:700;font-size:13px;box-shadow:0 3px 14px rgba(0,0,0,.6)}}
  .ret{{position:absolute;width:12px;height:12px;z-index:20;pointer-events:none}}
</style></head><body>
<div id="stage">
  <img src="data:image/png;base64,{b64(adv_im)}" alt="attacked">
  <div class="hud r">
    <div class="b" style="color:{rc}"><span class="sq" style="background:{rc}"></span>After · {right_label}</div>
    <span class="c">{right_conf:.1%} confidence</span>
  </div>
  <div id="cleanwrap"><div class="inner">
    <img src="data:image/png;base64,{b64(clean_im)}" alt="original">
  </div></div>
  <div class="hud l">
    <div class="b" style="color:{C['tertiary']}"><span class="sq" style="background:{C['tertiary']}"></span>Before · {left_label}</div>
    <span class="c">{left_conf:.1%} confidence</span>
  </div>
  <div id="handle"><div id="knob">⇔</div></div>
  <div class="ret" style="top:8px;left:8px;border-left:2px solid {C['tertiary']}66;border-top:2px solid {C['tertiary']}66"></div>
  <div class="ret" style="top:8px;right:8px;border-right:2px solid {C['primary']}66;border-top:2px solid {C['primary']}66"></div>
  <div class="ret" style="bottom:8px;left:8px;border-left:2px solid {C['tertiary']}66;border-bottom:2px solid {C['tertiary']}66"></div>
  <div class="ret" style="bottom:8px;right:8px;border-right:2px solid {C['primary']}66;border-bottom:2px solid {C['primary']}66"></div>
</div>
<script>
 (function(){{
   const st=document.getElementById('stage'), h=document.getElementById('handle'),
         cw=document.getElementById('cleanwrap'), inner=cw.querySelector('.inner');
   function set(cx){{
     const r=st.getBoundingClientRect();
     let p=Math.max(2,Math.min(98,((cx-r.left)/r.width)*100));
     h.style.left=p+'%'; cw.style.width=p+'%';
     inner.style.width=r.width+'px';
   }}
   function sync(){{ inner.style.width=st.getBoundingClientRect().width+'px'; }}
   sync(); window.addEventListener('resize',sync);
   let drag=false;
   st.addEventListener('pointerdown',e=>{{
     drag=true; st.setPointerCapture(e.pointerId); set(e.clientX); e.preventDefault();
   }});
   st.addEventListener('pointermove',e=>{{ if(drag) set(e.clientX); }});
   function stop(e){{ drag=false; try{{st.releasePointerCapture(e.pointerId);}}catch(_){{}} }}
   st.addEventListener('pointerup',stop);
   st.addEventListener('pointercancel',stop);
 }})();
</script></body></html>"""


# ══════════════════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <span class="mono tag"><span class="dot"></span>ADVERSARIAL ROBUSTNESS STUDY</span>
  <span class="mono" style="color:{C['onvar']};margin-left:10px">TARGET: BINARY FACE-PROVENANCE CNN</span>
  <span class="mono" style="color:{C['tertiary']};margin-left:10px">NORM: L∞ ≤ ε</span>
  <h1>A face detector at <span style="color:{C['tertiary']}">95.58%</span> accuracy,
      broken to <span style="color:{C['error']}">0.00%</span> by noise you cannot see.</h1>
  <p>Fast Gradient Sign and Projected Gradient Descent attacks against a CNN trained from
     scratch to tell photographs from StyleGAN faces. Every number on this page was measured,
     not estimated.</p>
  <div class="pills">
    <div class="pill"><span class="k">Conv backbone</span>
      <span class="v" style="color:{C['on']}">4× Conv2d · {N_PARAMS:,} params</span></div>
    <div class="pill"><span class="k">Input tensor</span>
      <span class="v" style="color:{C['secondary']}">128 × 128 × 3 RGB</span></div>
    <div class="pill"><span class="k">Test set</span>
      <span class="v" style="color:{C['tertiary']}">6,000 held out · scored once</span></div>
    <div class="pill"><span class="k">Runtime</span>
      <span class="v" style="color:{C['primary']}">CPU · torch {torch.__version__.split('+')[0]}</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

if not MODELS:
    st.error("No model weights found — expected `models/detector_v3.pth`.")
    st.stop()

t_try, t_what, t_res, t_how = st.tabs(
    ["Interactive studio", "What is this", "Evaluation", "Architecture & math"])


# ══════════════════════════════════════════════════════════════════════════
#  Tab 1
# ══════════════════════════════════════════════════════════════════════════
with t_try:
    left, right = st.columns([2, 1], gap="medium")

    # ---- control deck -----------------------------------------------------
    with right:
        st.markdown('<div class="panel"><h4>Target specimen</h4>', unsafe_allow_html=True)
        ex = []
        for d in ("app/examples", "examples", "app/demo_faces", "demo_faces"):
            ex += sorted(glob.glob(os.path.join(d, "*.png"))) + \
                  sorted(glob.glob(os.path.join(d, "*.jpg")))
        ex = ex[:6]
        names = [os.path.splitext(os.path.basename(p))[0].replace("_", " ").title() for p in ex]

        src = None
        if ex:
            pick = st.radio("specimen", names + ["Upload"], horizontal=True,
                            label_visibility="collapsed")
            if pick == "Upload":
                up = st.file_uploader("face", type=["png", "jpg", "jpeg"],
                                      label_visibility="collapsed")
                if up:
                    src = Image.open(io.BytesIO(up.read()))
            else:
                src = Image.open(ex[names.index(pick)])
        else:
            up = st.file_uploader("Upload a face", type=["png", "jpg", "jpeg"])
            if up:
                src = Image.open(io.BytesIO(up.read()))
        st.caption("FFHQ photographs · StyleGAN synthetics · 128×128 centre crop")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><h4>Defence model</h4>', unsafe_allow_html=True)
        which = st.radio("model", list(MODELS.keys()), horizontal=True,
                         label_visibility="collapsed")
        st.markdown(
            f'<div class="mono" style="color:{C["onvar"]};line-height:1.6;'
            f'background:{C["mid"]};border-radius:8px;padding:9px 11px;text-transform:none">'
            f'Adversarially trained (Madry-style, FGSM): <b style="color:{C["tertiary"]}">70.98%</b> '
            f'clean, <b style="color:{C["tertiary"]}">60.40%</b> under PGD at ε = 0.005. '
            f'At ε = 0.010 — its own training budget — it is at chance.</div>',
            unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><h4>Attack configuration</h4>', unsafe_allow_html=True)
        algo = st.radio("algo", ["FGSM · 1 step", "PGD · 10 steps"], horizontal=True,
                        label_visibility="collapsed")
        eps = st.slider("Attack strength  ε", 0.000, 0.020, 0.005, 0.001, format="%.3f")
        eightbit = st.toggle("Save as a real 8-bit image", value=True)
        st.markdown(
            f'<div class="mono" style="color:{C["onvar"]};text-transform:none;'
            f'background:{C["mid"]};border-radius:8px;padding:9px 11px;line-height:1.55">'
            f'ε = {eps:.3f} → every pixel moves by at most '
            f'<b style="color:{C["primary"]}">{eps*255:.2f}</b> of 255 brightness levels.</div>',
            unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if src is None:
        with left:
            st.info("Pick a specimen on the right, or upload a face, to run the attack.")
        st.stop()

    # ---- compute ----------------------------------------------------------
    model = MODELS[which]
    x = to_tensor(src)
    p0, ms0 = predict(model, x)
    i0 = int(p0.argmax())

    if eps > 0:
        y = torch.tensor([i0])
        xa = fgsm(model, x, y, eps) if algo.startswith("FGSM") else pgd(model, x, y, eps)
        if eightbit:
            xa = quantize(xa)
    else:
        xa = x.clone()
    p1, ms1 = predict(model, xa)
    i1 = int(p1.argmax())
    flipped = i1 != i0

    # ---- viewport ---------------------------------------------------------
    with left:
        st.markdown('<div class="panel"><h4>Inspection viewport</h4>', unsafe_allow_html=True)
        st.caption("Drag the divider. Both halves are the same image — only the attack differs.")
        components.html(
            split_lens(to_pil(x), to_pil(xa), CLASSES[i0], float(p0[i0]),
                       CLASSES[i1], float(p1[i1]), flipped),
            height=450, scrolling=False)

        col = C["error"] if flipped else C["tertiary"]
        st.markdown(f"""
<div class="verdict" style="background:{col}1f;border:1px solid {col}44">
  <div style="font-size:22px">{'⚠' if flipped else '✓'}</div>
  <div><div class="t" style="color:{col}">
      {'The verdict flipped.' if flipped else 'The verdict held.'}</div>
    <div class="s">{algo.split(' ·')[0]} at ε = {eps:.3f}
      {'inverted' if flipped else 'did not change'} the classification:
      {CLASSES[i0]} at {p0[i0]:.1%} → {CLASSES[i1]} at {p1[i1]:.1%}.
      {'Quantized to 8-bit, so this survives being saved as a real file.' if eightbit else
       'Floating-point only — turn on 8-bit to check it survives saving.'}</div></div>
</div>""", unsafe_allow_html=True)

        # telemetry
        st.markdown('<div class="panel"><h4>Output telemetry</h4>', unsafe_allow_html=True)
        for idx, accent in ((1, C["tertiary"]), (0, C["primary"])):
            st.markdown(f"""
<div class="prow">
  <div class="hd">
    <span style="color:{C['on']}">● {CLASSES[idx]}</span>
    <span><span class="was">{p0[idx]:.1%}</span>
      <b style="color:{accent};font-size:13px">{p1[idx]:.1%}</b></span>
  </div>
  <div class="track"><div class="fill" style="width:{p1[idx]*100:.1f}%;background:{accent}"></div></div>
</div>""", unsafe_allow_html=True)
        st.markdown(
            f'<div class="mono" style="color:{C["onvar"]};margin-top:8px">'
            f'SOFTMAX · CPU FORWARD PASS {ms1:.1f} MS · MEASURED THIS REQUEST</div></div>',
            unsafe_allow_html=True)

        # perturbation
        d = (xa - x).squeeze(0).permute(1, 2, 0).numpy()
        amp = np.clip(0.5 + d / (2 * max(eps, 1e-6)), 0, 1)
        pc, fc = st.columns([1, 1], gap="medium")
        with pc:
            st.markdown('<div class="panel"><h4>Perturbation map</h4>', unsafe_allow_html=True)
            st.image((amp * 255).astype(np.uint8), width="stretch")
            st.markdown(
                f'<div class="mono" style="color:{C["onvar"]};margin-top:6px">'
                f'MEAN |Δ| {np.abs(d).mean():.5f} · MAX |Δ| {np.abs(d).max():.5f} · '
                f'AMPLIFIED ×{1/(2*max(eps,1e-6)):.0f}</div></div>', unsafe_allow_html=True)
        with fc:
            st.markdown('<div class="panel"><h4>FGSM perturbation formula</h4>',
                        unsafe_allow_html=True)
            st.markdown(f"""<div class="formula">
<span style="color:{C['primary']}">x_adv</span> =
<span style="color:{C['secondary']}">clip</span>(<span style="color:{C['tertiary']}">x</span> +
<span style="color:{C['primary']};font-weight:700">ε</span> ·
<span style="color:{C['secondary']}">sign</span>(∇<span style="color:{C['onvar']}">x</span>
L(θ, <span style="color:{C['tertiary']}">x</span>, y)), 0, 1)</div>
<p style="font-size:12.5px;color:{C['onvar']};margin-top:10px;line-height:1.55">
One gradient ascent step along the direction that maximises loss with respect to the
<b>input pixels</b>, not the weights. Taking the sign bounds every pixel to exactly ε,
which is what keeps the change invisible. PGD iterates this and projects back into the
ε-ball each step.</p></div>""", unsafe_allow_html=True)

        st.caption("One image is an illustration, not evidence. The evidence is the "
                   "6,000-image sweep under Evaluation.")
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  Tab 2
# ══════════════════════════════════════════════════════════════════════════
with t_what:
    st.markdown("""
### The short version

A neural network can tell a real photograph from an AI-generated face with **95.58%
accuracy** on a test set it had never seen. Changing every pixel by at most **one
brightness level out of 255** — a change no human eye can detect — drives it to **0.00%**.

Not 50%, which would be a coin flip. **Zero.** The detector becomes systematically,
reliably wrong. Invert every answer and you would score 100%.

### Why it matters

Platforms use detectors like this to flag AI-generated images. Banks use them to confirm
an uploaded selfie is a real person. These systems exist **because humans cannot make the
judgment unaided** — most people cannot tell a StyleGAN face from a photograph.

So if the machine's answer can be reversed by a change nobody can see, there is no
backstop. The reviewer looks at the image and sees nothing wrong, because nothing
visible *is* wrong.

### "But it is still the same face"

Correct, and that is the distinction. The attack does not change **who** is in the
photograph. It flips the answer to a different question: *was this image photographed,
or synthesised?*

**Identity is not provenance.** Provenance is precisely the question humans cannot answer
by looking, which is why it was handed to a machine. Flipping it removes the only check
there was.

### How the attack works

Training a network computes the gradient of the error with respect to the **weights** and
nudges the weights to reduce it. This computes the gradient with respect to the **input
pixels** instead and nudges them the other way, to *increase* it. Same mathematics, frozen
weights, and the image is the thing being optimised.
""")


# ══════════════════════════════════════════════════════════════════════════
#  Tab 3
# ══════════════════════════════════════════════════════════════════════════
with t_res:
    st.markdown("### Final test set — held out from day three, evaluated once")
    a, b, c, d_ = st.columns(4)
    a.metric("Clean accuracy", "95.58%")
    b.metric("Under attack", "0.00%", "−95.6 pts", delta_color="inverse")
    c.metric("Defended, clean", "70.98%")
    d_.metric("Majority baseline", "50.33%")

    st.markdown("""
| Model | Clean | ε=0.002 FGSM / PGD | ε=0.005 FGSM / PGD | ε=0.010 FGSM / PGD |
|---|---|---|---|---|
| Undefended | 0.9558 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| Defended | 0.7098 | 0.6703 / 0.6688 | 0.6118 / 0.6040 | 0.5292 / 0.5043 |

6,000 unseen images. PGD ≤ FGSM everywhere and accuracy falls monotonically in ε —
two of the three standard checks against gradient masking.
""")

    for path, cap in [
        ("results/figures/epsilon_sweep.png",
         "Accuracy against attack strength. Collapse is total by ε = 0.002."),
        ("results/figures/defence_comparison.png",
         "Adversarial training: robustness bought with clean accuracy."),
        ("results/figures/attack_exhibit.png",
         "The same face, opposite verdicts. This person does not exist — StyleGAN output."),
    ]:
        if os.path.exists(path):
            st.image(path, width="stretch")
            st.caption(cap)

    st.markdown(f"""
---
#### Read honestly

**Skill above chance** is the fair measure on a balanced binary task. Clean, the detector
holds **91%** of the available skill. Under attack it holds **−101%** — past zero, into
being reliably wrong.

The defence keeps about **half its own skill** at ε = 0.005 and costs **24.5 points** of
clean accuracy. At ε = 0.010, the budget it was *trained* on, it sits at chance, because
FGSM adversarial training is substantially weaker than PGD adversarial training.

#### What this does not show

- Not that this detector is *more* fragile than others. No reference model was trained.
- Not that the attack survives JPEG. Only 8-bit rounding was tested.
- Not black-box. Every attack had full access to the weights.
- Single seed throughout.
""")


# ══════════════════════════════════════════════════════════════════════════
#  Tab 4
# ══════════════════════════════════════════════════════════════════════════
with t_how:
    st.markdown(f"""
### The detector

Four blocks of `Conv2d(3×3) → BatchNorm → ReLU → MaxPool`, channels 3 → 32 → 64 → 128 → 128,
image 128×128 down to 8×8, then global average pooling and one linear layer.
**{N_PARAMS:,} parameters**, trained from scratch — not fine-tuned — on 45,000 images for
25 epochs with Adam at 3e-4 and cosine decay.

Global average pooling came out of a measurement: in an earlier comparison the convolutions
cost ~18,800 parameters while a dense head cost over 400,000. Pooling removes the head.

### Vanishing gradients, measured rather than cited

In a 20-layer sigmoid network the gradient reaching layer 1 is **1.8e-19** — about 10²⁰
times weaker than at the last layer. Plotted on a log axis it is a straight line, and a
straight line in log space *is* the proof that repeated multiplication is the cause.

Of the four standard fixes, **He initialisation alone** was the biggest practical win,
carrying a network past 40 layers where plain ReLU died at 5. Residual connections *without*
normalisation produced the opposite failure — **exploding** gradients ~1000× too large —
which is exactly why real ResNets pair skips with batch norm.

### One engineering note

The first training run was **120× too slow** — 16 minutes an epoch. Profiling showed the
GPU idle: the job was disk-bound, not compute-bound, reading 140,000 small files off a
network mount at ~31 images a second. Holding the data in GPU memory as uint8 cut epochs
to **8 seconds**.
""")

st.markdown(f"""
<div class="foot">
  <span>BLINDSPOT · DEEP LEARNING (NAI-701)</span>
  <span>HARSHIT PANDEY · ROLL NO. 2410013155315 · B.TECH CSE (AI)</span>
  <span>GUIDE: DR. PREM SHANKER YADAV</span>
  <span>FACULTY OF ENGINEERING &amp; TECHNOLOGY, UNIVERSITY OF LUCKNOW</span>
  <span style="color:{C['tertiary']}">CPU INFERENCE · TORCH {torch.__version__.split('+')[0]}</span>
</div>
""", unsafe_allow_html=True)