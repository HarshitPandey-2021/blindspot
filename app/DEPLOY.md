# BlindSpot demo — run locally, then deploy

The app is one file. It loads your two trained models and runs on CPU — no GPU, no Kaggle.

```
app/
├── app.py              the demo
├── requirements.txt
├── DEPLOY.md           this file
├── models/             <- copy the two .pth files here before deploying
└── examples/           <- optional sample faces (.png)
```

---

## Step 1 — get example faces out of Kaggle (optional but worth it)

Examples make the demo instant for whoever opens the link. In your live Kaggle session:

```python
import os
from PIL import Image
os.makedirs("/kaggle/working/demo_faces", exist_ok=True)

with torch.no_grad():
    P = torch.cat([torch.softmax(detector(Xva[i:i+256].float()/255.), 1)
                   for i in range(0, Xva.size(0), 256)])
conf, pred = P.max(1)

n = 0
for label, tag in [(0, "aigen"), (1, "real")]:
    ids = ((Yva == label) & (pred == label) & (conf > 0.99)).nonzero().squeeze(1)[:3]
    for j, i in enumerate(ids.tolist()):
        arr = Xva[i].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(arr).save(f"/kaggle/working/demo_faces/{tag}_{j}.png")
        n += 1
print(f"saved {n} example faces")

import shutil
shutil.make_archive("/kaggle/working/demo_faces", "zip", "/kaggle/working/demo_faces")
```

Download `demo_faces.zip`, unzip it into `C:\dev\blindspot\app\examples\`.

**The app runs fine with no examples folder** — it just won't show the example strip.

---

## Step 2 — run it locally

```powershell
cd C:\dev\blindspot\app
pip install -r requirements.txt
pip install gradio
python app.py
```

Open **http://127.0.0.1:7860**.

`gradio` is installed separately and deliberately left out of `requirements.txt`, because
Hugging Face Spaces supplies its own version and listing it there causes conflicts.

The app looks for models in `app/models/` first, then `../models/`. Locally the second
path already works, so you don't need to copy anything yet.

### Test it like this

| Setting | Expected |
|---|---|
| example face, undefended, eps **0.005**, 8-bit **on** | verdict **flips** |
| same, eps **0.001**, 8-bit **on** | verdict **holds** — the change rounds away |
| same, **defended** model, eps 0.005 | often holds; sometimes wrong even clean (it's only 70.7% accurate) |

That second row is your quantization finding, live. It is a good thing to show the examiner.

---

## Step 3 — deploy to Hugging Face Spaces

Free, permanent link, works on phones.

1. Make an account at **huggingface.co**
2. **New Space** → name `blindspot` → SDK **Gradio** → hardware **CPU basic (free)** → **Public**
3. **Do not replace the README.md that HF generates.** It contains the `sdk_version` the
   Space needs. You can edit the title later.
4. First copy your models into the app folder so you can upload one tree:

   ```powershell
   mkdir C:\dev\blindspot\app\models
   copy C:\dev\blindspot\models\detector_v3.pth             C:\dev\blindspot\app\models\
   copy C:\dev\blindspot\models\detector_defended_final.pth C:\dev\blindspot\app\models\
   ```

5. In the Space → **Files** → **Add file → Upload files**. Upload `app.py`,
   `requirements.txt`, the `models` folder and the `examples` folder.
6. Wait 3–6 minutes for the build. The **Logs** tab shows errors if anything fails.

Your link: `huggingface.co/spaces/<your-username>/blindspot`

**Free Spaces sleep when idle.** The first visitor waits ~30–60 seconds while it wakes.
Open the link yourself a few minutes before the viva.

If the build fails, copy the log and send it over.

---

## Step 4 — commit it

```powershell
cd C:\dev\blindspot
git add app
git commit -m "Add deployable Gradio demo"
git push
```

Put the Space link in your README and on your slides.

---

## Two cautions for the viva

**The demo is the illustration, not the evidence.** It shows one image. Your evidence is
the 6,000-image sweep. If asked "is one example enough?", say exactly that and point at
the epsilon sweep chart.

**Check the clean verdict before attacking.** If the detector is already wrong about an
image, attacking it proves nothing. That's why the examples come from your validation set.
A phone selfie is a different distribution from 128×128 StyleGAN/FFHQ crops and may get an
odd verdict before any attack — that's a real limitation of the detector, worth naming
rather than hiding.

**Record a 30–60 second screen video of the working demo** as a fallback for dead wifi.
