# 🐾 WhiskerBond — Pet Breed Identifier

> Given two pet photos, tells you if they're the same breed.
> Powered by EfficientNet-B0 · Oxford-IIIT Pets · Stanford Dogs · **No API key, 100% free.**

**[🚀 Live Demo on Streamlit Cloud →](https://whiskerbond.streamlit.app)**

---

## How It Works

```
Image 1 ──┐
           ├── EfficientNet-B0 (pretrained, ImageNet)
Image 2 ──┘         │
                     ↓
           Remove classifier head
                     │
                     ↓
           1280-dim feature embedding per image
                     │
                     ↓
           Cosine similarity + ImageNet top-5 classification
                     │
                     ↓
           Same / Different + Confidence + Reasoning
```

- **Model**: EfficientNet-B0 pretrained on ImageNet (~20MB, downloads once)
- **Breed labels**: Oxford-IIIT Pet Dataset (37 breeds) + Stanford Dogs (120 breeds)
- **Similarity**: Cosine similarity between 1280-dim embeddings
- **Decision**: ≥0.78 → Same breed · ≤0.60 → Different · in-between → label tie-breaker

---

## Deploy to Streamlit Cloud (step-by-step)

### 1. Fork / push this repo to GitHub

Make sure your repo contains these files at the root:
```
streamlit_app.py      ← entry point (Streamlit Cloud looks for this)
requirements.txt      ← CPU-only torch + dependencies
packages.txt          ← system libs (libgl1 etc.)
.streamlit/
  config.toml         ← theme + server settings
models/
utils/
```

### 2. Go to share.streamlit.io

- Sign in with your GitHub account
- Click **"New app"**

### 3. Fill in the form

| Field | Value |
|-------|-------|
| Repository | `your-github-username/whiskerbond` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |

### 4. Click Deploy

Streamlit Cloud will:
1. Install `packages.txt` (system deps)
2. Install `requirements.txt` (Python deps — takes ~3-4 min first time)
3. Start the app

The EfficientNet-B0 weights (~20MB) download on **first user visit** and are cached.
Subsequent visits are fast.

### 5. Share your URL

Your app will be live at:
```
https://your-username-whiskerbond-streamlit-app-xxxxxx.streamlit.app
```

---

## Run Locally

```bash
# Clone
git clone https://github.com/your-username/whiskerbond
cd whiskerbond

# Install
pip install -r requirements.txt

# Run
streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser.

> **Windows users:** You can also double-click `run_app.bat`

---

## Project Structure

```
whiskerbond/
├── streamlit_app.py          ← Main app (Streamlit Cloud entry point)
├── requirements.txt          ← CPU-only torch, pinned for cloud
├── packages.txt              ← System packages for Linux cloud env
├── run_app.bat               ← Windows one-click launcher
├── run_app.sh                ← Mac/Linux launcher
├── .streamlit/
│   └── config.toml           ← Theme and server config
├── models/
│   ├── embedder.py           ← EfficientNet-B0 wrapper → 1280-dim embeddings
│   └── comparator.py         ← Core logic: embed → compare → verdict
├── utils/
│   ├── breed_labels.py       ← Oxford-IIIT + Stanford Dogs breed label maps
│   └── image_utils.py        ← Image loading (files + URLs)
├── tests/
│   └── test_comparator.py    ← Test suite with real breed image pairs
├── scripts/
│   └── setup_weights.py      ← One-time weight pre-download script
└── demo_cli.py               ← Command-line interface
```

---

## Datasets

| Dataset | Breeds | Images | License |
|---------|--------|--------|---------|
| [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) | 37 (dogs + cats) | 7,393 | CC BY-SA 4.0 |
| [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) | 120 | 20,580 | Research use |

---

## Limitations

- Similar-looking breeds (Golden vs Labrador) may score high even when different
- Mixed breeds may get classified as one parent breed
- Accuracy depends on image quality — blurry/dark photos reduce confidence

---

*Built for the WhiskerBond intern assignment · Free & open source 🐾*
