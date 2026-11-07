"""
BlindSpot — fooling an AI-generated-face detector.

Four tabs: a live attack demo, a plain-English explainer, the experimental
results, and the engineering underneath.

Runs on CPU. Local:  python app.py   ->  http://127.0.0.1:7860

Works on Gradio 4, 5 and 6: style arguments are routed to whichever of
Blocks() / launch() accepts them in the installed version.
"""

import os
import inspect
import numpy as np
import torch
import torch.nn as nn
import gradio as gr
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cpu"
NAMES = ["AI-GENERATED", "REAL PHOTO"]          # class 0 = fake, class 1 = real


# ══════════════════════════════ model ══════════════════════════════

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
    )


def find_file(fname, dirs):
    for d in dirs:
        p = os.path.join(HERE, d, fname)
        if os.path.exists(p):
            return p
    return None


def load_model(fname):
    path = find_file(fname, ("models", os.path.join("..", "models"), "."))
    if path is None:
        return None
    m = make_detector()
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


DETECTOR = load_model("detector_v3.pth")
DEFENDED = load_model("detector_defended_final.pth")

if DETECTOR is None:
    raise FileNotFoundError("detector_v3.pth not found — put it in app/models/ or ../models/")

UNDEF_LABEL = "undefended  ·  95.2% clean"
DEF_LABEL = "defended  ·  70.7% clean"
MODEL_CHOICES = [UNDEF_LABEL] + ([DEF_LABEL] if DEFENDED is not None else [])


def fig(name):
    """Path to a result figure, or None if it isn't shipped with the app."""
    return find_file(name, ("figures", os.path.join("..", "results", "figures")))


# ══════════════════════════════ attack ══════════════════════════════

def fgsm(model, x, y, eps):
    """Gradient of the loss w.r.t. the INPUT PIXELS, one signed step uphill."""
    x = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x), y)
    model.zero_grad()
    loss.backward()
    return (x + eps * x.grad.sign()).clamp(0, 1).detach()


TF = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])


def to_img(t):
    return (t.detach().permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)


# ══════════════════════════════ cards ══════════════════════════════

def verdict_card(idx, conf, tag):
    cls = "v-fake" if idx == 0 else "v-real"
    return (f"<div class='bsx {cls}'><span class='lbl'>{tag}</span>"
            f"<span class='val'>{NAMES[idx]}</span>"
            f"<span class='cf'>{conf * 100:.1f}% confident</span></div>")


def delta_card(change, eps):
    return (f"<div class='bsx v-mid'><span class='lbl'>PIXELS MOVED BY</span>"
            f"<span class='val'>{change:.1f} / 255</span>"
            f"<span class='cf'>epsilon = {eps:.3f}</span></div>")


def banner(flipped, change, eps, which):
    if eps == 0:
        return ("<div class='bsnote hold'>Attack strength is zero, so nothing was changed. "
                "Drag the slider to start attacking.</div>")
    if flipped:
        return (f"<div class='bsnote flip'><b>The verdict flipped.</b> Every pixel moved by at "
                f"most <b>{change:.1f} out of 255</b> brightness levels — far too small for your "
                f"eye — and the {which} detector reversed its answer completely.</div>")
    return (f"<div class='bsnote hold'><b>The verdict held.</b> The attack moved pixels by up to "
            f"{change:.1f}/255 and the {which} detector kept its answer. "
            f"This is what adversarial training buys.</div>")


IDLE = "<div class='bsnote hold'>Pick an example below, or upload a face to begin.</div>"


def run(image, eps, which, quantize):
    if image is None:
        return None, None, None, "", "", "", IDLE

    use_def = DEFENDED is not None and which == DEF_LABEL
    model = DEFENDED if use_def else DETECTOR
    short = "defended" if use_def else "undefended"

    x = TF(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        p0 = torch.softmax(model(x), 1)[0]
    y = p0.argmax().unsqueeze(0)              # attack the model's own verdict

    xa = fgsm(model, x, y, eps) if eps > 0 else x.clone()
    if quantize:
        xa = (xa * 255).round().clamp(0, 255) / 255.

    with torch.no_grad():
        p1 = torch.softmax(model(xa), 1)[0]

    d = (xa - x)[0]
    spread = (d.max() - d.min()).item()
    pert = (d - d.min()) / spread if spread > 1e-12 else torch.full_like(d, 0.5)
    change = (xa - x).abs().max().item() * 255
    flipped = int(p0.argmax()) != int(p1.argmax())

    return (to_img(x[0]), to_img(pert), to_img(xa[0]),
            verdict_card(int(p0.argmax()), p0.max().item(), "BEFORE THE ATTACK"),
            delta_card(change, eps),
            verdict_card(int(p1.argmax()), p1.max().item(), "AFTER THE ATTACK"),
            banner(flipped, change, eps, short))


# ══════════════════════════════ styling ══════════════════════════════

CSS = """
.gradio-container { max-width: 1280px !important; }

#bshero { text-align:center; padding: 22px 0 6px; }
#bshero h1 {
  font-size: 3rem; font-weight: 850; margin: 0; letter-spacing: -0.03em;
  background: linear-gradient(100deg,#8b5cf6 0%,#22d3ee 46%,#fb7185 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
}
#bshero p { margin: 10px auto 0; max-width: 660px; font-size: 1.06rem;
            line-height: 1.55; opacity: .72; }

.bsx { border-radius: 14px; padding: 13px 14px; text-align: center;
       border: 1px solid rgba(255,255,255,.10); font-weight: 700; }
.bsx .lbl { display:block; font-size:.64rem; letter-spacing:.15em; opacity:.62; margin-bottom:6px; }
.bsx .val { display:block; font-size:1.14rem; }
.bsx .cf  { display:block; font-size:.78rem; font-weight:500; opacity:.74; margin-top:5px; }

.v-fake { background: linear-gradient(140deg,#3b0d1c,#5e1028); color:#ff9db0; }
.v-real { background: linear-gradient(140deg,#07301f,#0c4a30); color:#5fe9ab; }
.v-mid  { background: linear-gradient(140deg,#1b1733,#241c47); color:#c3b6ff; }

.bsnote { border-radius:16px; padding:16px 20px; margin-top:14px;
          font-size:1rem; line-height:1.6; border:1px solid rgba(255,255,255,.08); }
.bsnote.flip { background: linear-gradient(135deg,#2d0f44,#4c1233); color:#ffd6e0; }
.bsnote.hold { background: linear-gradient(135deg,#0e2338,#113040); color:#c6e8ff; }

.bsprose { font-size: 1.02rem; line-height: 1.72; }
.bsprose h2 { font-size:1.45rem; margin:26px 0 6px; font-weight:800; letter-spacing:-.01em; }
.bsprose h3 { font-size:1.1rem; margin:22px 0 4px; font-weight:750; opacity:.9; }
.bsprose .big { font-size:1.18rem; line-height:1.68; }

.bscallout { border-radius:16px; padding:18px 22px; margin:18px 0;
             border:1px solid rgba(139,92,246,.28);
             background: linear-gradient(135deg, rgba(139,92,246,.14), rgba(34,211,238,.10)); }
.bscallout b { font-weight:800; }

.bsstat { display:inline-block; border-radius:14px; padding:14px 20px; margin:6px 8px 6px 0;
          border:1px solid rgba(255,255,255,.10);
          background: linear-gradient(140deg,#1b1733,#241c47); }
.bsstat .n { display:block; font-size:1.7rem; font-weight:850; color:#c3b6ff; line-height:1.1; }
.bsstat .t { display:block; font-size:.78rem; opacity:.7; margin-top:4px; letter-spacing:.04em; }

#bsfoot { margin-top:26px; font-size:.88rem; line-height:1.68; opacity:.62; }
"""

THEME = gr.themes.Soft(primary_hue="violet", neutral_hue="slate")


def split_style_kwargs():
    """Gradio 6 moved theme/css from Blocks() to launch(). Send each one
    wherever the installed version actually accepts it."""
    blocks_kw, launch_kw = {"title": "BlindSpot"}, {}
    try:
        launch_params = inspect.signature(gr.Blocks.launch).parameters
    except (TypeError, ValueError):
        launch_params = {}
    try:
        blocks_params = inspect.signature(gr.Blocks.__init__).parameters
    except (TypeError, ValueError):
        blocks_params = {}
    for key, val in (("theme", THEME), ("css", CSS)):
        if key in launch_params:
            launch_kw[key] = val
        elif key in blocks_params:
            blocks_kw[key] = val
    return blocks_kw, launch_kw


BLOCKS_KW, LAUNCH_KW = split_style_kwargs()


# ══════════════════════════════ examples ══════════════════════════════

EX_DIR = os.path.join(HERE, "examples")
EXAMPLES = []
if os.path.isdir(EX_DIR):
    EXAMPLES = [[os.path.join(EX_DIR, f), 0.005, UNDEF_LABEL, True]
                for f in sorted(os.listdir(EX_DIR))
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]


def show_fig(name, label):
    """Add a figure to the page if it shipped with the app."""
    p = fig(name)
    if p:
        gr.Image(value=p, label=label, height=400)


# ══════════════════════════════ layout ══════════════════════════════

with gr.Blocks(**BLOCKS_KW) as demo:

    gr.HTML("<div id='bshero'><h1>BlindSpot</h1>"
            "<p>A computer program that spots AI-generated faces 95% of the time — "
            "and the invisible noise that makes it confidently wrong.</p></div>")

    with gr.Tabs():

        # ─────────────────────────── TAB 1 ───────────────────────────
        with gr.Tab("Try it"):

            gr.HTML(
                "<div class='bscallout'><b>What to do:</b> click one of the example faces "
                "at the bottom, then drag the <b>Attack strength</b> slider. Watch the two "
                "outer pictures stay identical while the verdict underneath changes.</div>"
            )

            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    inp = gr.Image(type="pil", label="Face", height=250)
                    which = gr.Radio(MODEL_CHOICES, value=UNDEF_LABEL, label="Which model to attack")
                    eps = gr.Slider(
                        0, 0.05, value=0.005, step=0.001,
                        label="Attack strength — epsilon   (0.005 ≈ 1 brightness level out of 255)")
                    quant = gr.Checkbox(
                        value=True, label="Round to 8-bit (as if saved as a real PNG file)")
                    go = gr.Button("Run the attack", variant="primary")

                with gr.Column(scale=2, min_width=520):
                    with gr.Row():
                        im_o = gr.Image(label="Original", height=236)
                        im_p = gr.Image(label="The noise, amplified so you can see it", height=236)
                        im_a = gr.Image(label="Attacked", height=236)
                    with gr.Row():
                        v_before = gr.HTML()
                        v_mid = gr.HTML()
                        v_after = gr.HTML()

            msg = gr.HTML(IDLE)

            if EXAMPLES:
                try:
                    gr.Examples(examples=EXAMPLES, inputs=[inp, eps, which, quant])
                except Exception as exc:                      # noqa: BLE001
                    print(f"[BlindSpot] examples disabled: {exc}")

            gr.HTML(
                "<div id='bsfoot'>"
                "<b>Three things to try.</b> (1) Attack strength <b>0.005</b> on the "
                "<i>undefended</i> model — the verdict usually flips. (2) Drop it to "
                "<b>0.001</b> — it usually holds, because a change that small disappears when "
                "the image is saved as a real file. (3) Switch to the <i>defended</i> model at "
                "0.005 — it mostly holds its ground.<br><br>"
                "<b>Privacy.</b> Uploaded images are processed in memory and never stored. "
                "Please don't upload anything sensitive."
                "</div>"
            )

        # ─────────────────────────── TAB 2 ───────────────────────────
        with gr.Tab("What is this?"):
            gr.HTML("""
<div class='bsprose'>

<p class='big'>Some of the faces you see online were never photographed. A computer drew
them. This project is about the programs that try to tell the difference — and how easily
they can be tricked.</p>

<h2>Step 1 — Build a detector</h2>
<p>I trained a program on <b>140,000 face pictures</b>. Half were real photographs of real
people. Half were generated by an AI called StyleGAN. The program learned to tell them
apart, and it now gets this right <b>95 times out of 100</b> on faces it has never seen.</p>
<p>That is already better than most people can do by eye. Try the grid below — the top row
is fake, the bottom row is real. Most people cannot reliably tell.</p>
</div>""")
            show_fig("sample_faces.png", "Top row: AI-generated. Bottom row: real photographs.")

            gr.HTML("""
<div class='bsprose'>
<h2>Step 2 — Break it</h2>
<p>Here is the uncomfortable part. I can change a picture by an amount <b>so small you
cannot see it</b> — about one brightness level out of 255, which is less than the
difference between two shades that look identical on your screen — and the detector
reverses its answer.</p>
<p>Not "becomes unsure". <b>Reverses.</b> It goes from "99.9% sure this is fake" to
"100% sure this is real", looking at what is, to your eyes, the same photograph.</p>
</div>""")
            show_fig("attack_exhibit.png", "Left and right are the same face. The verdicts are opposite.")

            gr.HTML("""
<div class='bsprose'>

<div class='bscallout'>
<b>How is that possible?</b> The detector doesn't look at a face the way you do. It looks
at tiny patterns in the pixels — the faint fingerprints AI image generators leave behind.
Those patterns are invisible to you, so noise that is also invisible to you can scribble
all over them. You see a face. The detector sees evidence, and the evidence has been
quietly rewritten.
</div>

<h2>Step 3 — Fix it</h2>
<p>There is a known defence called <b>adversarial training</b>: while the program is
learning, you keep showing it attacked pictures alongside normal ones, with the correct
answers, until it stops falling for them.</p>
<p>It works, partly — and it is not free. The defended version resists the attack far
better, but it is a <b>worse detector on ordinary images</b>, dropping from 95.2% to 70.7%
accuracy. That trade is one of the real findings of this project, and the demo lets you
feel it: switch to the defended model and you will see it hold firm under attack, but also
get a few clean images wrong.</p>

<h2>Why any of this matters</h2>
<p>Programs like this are running right now. Social media platforms use them to flag
AI-generated images. Banks use them to check that the selfie you uploaded is a real person
and not a generated face. If invisible noise walks straight past them, those checks are
decoration.</p>
<p>And there is a second harm that fewer people test: the attack works in <b>both</b>
directions. You can push a fake face through as real, and you can push a <b>real person's
genuine photograph</b> over the line into "fake" — getting an innocent person flagged.</p>

</div>""")

        # ─────────────────────────── TAB 3 ───────────────────────────
        with gr.Tab("The results"):
            gr.HTML("""
<div class='bsprose'>
<div>
  <span class='bsstat'><span class='n'>95.22%</span><span class='t'>DETECTOR ACCURACY</span></span>
  <span class='bsstat'><span class='n'>0.00%</span><span class='t'>UNDER ATTACK</span></span>
  <span class='bsstat'><span class='n'>1 / 255</span><span class='t'>PIXELS CHANGED BY</span></span>
  <span class='bsstat'><span class='n'>6,000</span><span class='t'>IMAGES TESTED</span></span>
</div>

<h2>How much noise does it take?</h2>
<p>Epsilon is the attack budget — the most any single pixel is allowed to move. The chart
shows what happens to accuracy as that budget grows.</p>
<p>The collapse is total by <b>epsilon 0.002</b>, which is half a brightness level out of
255. Not one of 6,000 images survived. And at <b>0.005</b> — a change that still survives
being saved as a real 8-bit image file — accuracy is still exactly zero.</p>
</div>""")
            show_fig("epsilon_sweep.png", "Accuracy vs attack strength")

            gr.HTML("""
<div class='bsprose'>
<div class='bscallout'>
<b>A result worth pausing on.</b> At 0% accuracy the detector is not merely broken — it is
<b>perfectly wrong</b>. Flip every answer it gives and you would get 100% correct. The
attack doesn't blind the detector; it inverts it into a reliable liar.
</div>

<h2>Two attacks, same budget</h2>
<p>FGSM takes one big step. PGD takes ten small ones and is strictly stronger. Notice the
FGSM line rising again on the right — that is the <i>attack</i> overshooting on a curved
surface, not the model recovering. PGD stays flat at zero the whole way, proving the
adversarial examples are still there and FGSM is simply missing them.</p>
</div>""")
            show_fig("fgsm_vs_pgd.png", "FGSM vs PGD")

            gr.HTML("""
<div class='bsprose'>
<h2>Does the defence work?</h2>
<p>Partly, and the honest answer depends on how you measure. Below the budget it was
trained on, the defended model keeps roughly <b>half its skill</b> where the undefended one
loses all of it. At and above that budget, against a full-strength attack, it buys very
little. It also costs 24.5 points of clean accuracy.</p>
</div>""")
            show_fig("defence_comparison.png", "Defended vs undefended across attack strengths")

            gr.HTML("""
<div class='bsprose'>
<h2>What is the detector actually looking at?</h2>
<p>Before building the attack I ran a test: blur the images, destroying fine texture while
leaving the face and composition intact. Accuracy collapsed from 95.2% to <b>50.9%</b> — a
coin flip.</p>
<p>That told me the detector lives on fine, high-frequency texture. It also let me
<b>predict, in advance</b>, that this detector would break at an unusually small attack
budget — because that is exactly the kind of detail the attack rewrites. The prediction was
recorded on 31 October and confirmed on 5 November.</p>
</div>""")
            show_fig("frequency_reliance.png", "Accuracy as fine detail is removed")
            show_fig("confusion_matrix.png", "Which way the detector fails")

        # ─────────────────────────── TAB 4 ───────────────────────────
        with gr.Tab("Under the hood"):
            gr.HTML("""
<div class='bsprose'>

<h2>The detector</h2>
<p>A convolutional neural network with <b>241,794 parameters</b>. Four blocks, each one
<code>Conv2d(3×3) → BatchNorm → ReLU → MaxPool</code>, taking the image from 128×128 down to
8×8 while the channel count grows 3 → 32 → 64 → 128 → 128. Then global average pooling and
a single linear layer with two outputs.</p>
<p>The global average pooling is deliberate. An earlier experiment comparing a plain network
against a convolutional one showed that the convolutions were cheap — about 18,800
parameters — while the dense layer bolted on the end cost over 400,000. Pooling removes
that cost.</p>
<p>Trained with Adam at a learning rate of 3e-4 with cosine decay, random horizontal flips,
25 epochs on 45,000 images.</p>
</div>""")
            show_fig("training_curve_v3.png", "Training and validation accuracy")

            gr.HTML("""
<div class='bsprose'>
<h2>The attack, in one line</h2>
<p style='font-family:ui-monospace,monospace; font-size:1.05rem; padding:14px 18px;
border-radius:12px; background:rgba(139,92,246,.12); border:1px solid rgba(139,92,246,.25);'>
x_adv = clamp( x + ε · sign( ∂Loss/∂x ) , 0, 1 )
</p>
<p>Training a network computes the gradient of the loss with respect to the
<b>weights</b> — "how should the model change to be less wrong?" This computes it with
respect to the <b>input pixels</b> instead, and steps the other way: "how should the
picture change to make the model more wrong?"</p>
<p>It is the same backpropagation machinery, stopped one step earlier and pointed somewhere
else. Taking the <code>sign</code> rather than the raw gradient is what bounds every pixel
to exactly ε, and that bound is what keeps the change invisible.</p>

<h2>The vanishing gradient experiments</h2>
<p>Before any of the above, I measured why deep networks are hard to train at all. In a
20-layer network using sigmoid activations, the learning signal reaching the first layer is
<code>1.8e-19</code> — roughly <b>10<sup>20</sup> times weaker</b> than the signal at the
last layer. The first layer effectively never learns.</p>
<p>Each line is straight on a logarithmic axis, which is itself the proof: a constant
shrinking factor per layer is exactly what repeated multiplication by the chain rule
predicts.</p>
</div>""")
            show_fig("grad_depth_sweep.png", "Gradient strength per layer, at four depths")

            gr.HTML("""
<div class='bsprose'>
<p>Then the four standard fixes, each measured rather than assumed. Batch normalization
gave near-perfect gradient flow (a ratio of 0.8). <b>He initialisation alone was the
biggest practical win</b>, carrying a working network past 40 layers where plain ReLU died
at 5.</p>
<p>Residual connections without normalization produced the opposite failure —
<b>exploding</b> gradients, around 1000× too large — which is precisely why real ResNets
pair skip connections with batch norm.</p>
</div>""")
            show_fig("grad_fixes_20layer.png", "Gradient flow under each fix")
            show_fig("depth_wall_by_fix.png", "How deep each fix lets you go")

            gr.HTML("""
<div class='bsprose'>

<h2>What this project does not show</h2>
<div class='bscallout'>
<p style='margin:0 0 10px'><b>These limits are stated deliberately.</b></p>
<ul style='margin:0; padding-left:20px; line-height:1.75'>
<li>That this detector is <i>more</i> fragile than others. No reference model was trained,
so "extremely fragile" is supported; "unusually fragile" is not.</li>
<li>That the attack survives JPEG compression. Only 8-bit rounding was tested.</li>
<li>That it works black-box. Every attack here had full access to the model's weights.</li>
<li>That ε=0.01 was the right training budget for the defence. That was never isolated.</li>
</ul>
</div>

<h2>Things that went wrong</h2>
<p>The first training run was 120× too slow; profiling showed it was disk-bound, not
compute-bound, and restructuring the data pipeline cut epochs from 16 minutes to 8 seconds.
An early depth experiment produced identical results at every depth and was therefore
uninterpretable — there was no positive control, so it could not distinguish "depth killed
it" from "the setup is broken". And one prediction, that residual connections plus batch
norm would win at depth, was simply wrong: plain He initialisation beat it.</p>
<p>All three are in the lab notebook with dates.</p>

<h2>Built with</h2>
<p>PyTorch · trained on Kaggle's free T4 GPU · dataset "140k Real and Fake Faces"
(70,000 real photographs from Flickr, 70,000 StyleGAN-generated) · FGSM and PGD both
implemented from scratch rather than imported.</p>

</div>""")

    ins = [inp, eps, which, quant]
    outs = [im_o, im_p, im_a, v_before, v_mid, v_after, msg]

    go.click(run, ins, outs)
    for comp in (inp, eps, which, quant):
        comp.change(run, ins, outs)


if __name__ == "__main__":
    demo.launch(**LAUNCH_KW)
