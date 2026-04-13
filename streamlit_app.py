"""
WhiskerBond – Pet Breed Identifier
====================================
Entry point for Streamlit Cloud deployment.

Local run:
  streamlit run streamlit_app.py

Streamlit Cloud:
  Set main file path to: streamlit_app.py
"""

import sys
import os
import json
import time

# ── Make sure the project root is always on the path ─────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from PIL import Image

from utils.image_utils import load_image, validate_image
from models.comparator import compare
from models.embedder import PetEmbedder


# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="WhiskerBond – Breed Identifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─── Cache the model so it loads ONCE and stays in memory ────────────────────
# Without this, Streamlit reloads the model on every interaction — very slow.
@st.cache_resource(show_spinner="🧠 Loading EfficientNet-B0 weights (~20MB, once only)…")
def load_model() -> PetEmbedder:
    """Download and cache the EfficientNet-B0 embedder. Runs once per session."""
    embedder = PetEmbedder()
    embedder._load()
    return embedder


# Pre-warm the model on app startup so the first comparison is fast
_embedder = load_model()


# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .wb-header { text-align: center; padding: 10px 0 4px 0; }
  .wb-header h1 { font-size: 2.6rem; font-weight: 800; margin-bottom: 0; }
  .wb-sub { text-align: center; color: #666; font-size: 1rem; margin-bottom: 24px; }

  div[data-testid="stFileUploadDropzone"] {
    border: 2.5px dashed #aaa !important;
    border-radius: 14px !important;
    background: #fafafa !important;
    min-height: 200px !important;
    transition: border-color 0.2s;
  }
  div[data-testid="stFileUploadDropzone"]:hover {
    border-color: #4a90d9 !important;
    background: #f0f7ff !important;
  }

  .verdict-same {
    background: #d4edda; color: #155724;
    padding: 22px 32px; border-radius: 14px;
    font-size: 1.6rem; font-weight: 800;
    text-align: center; border: 2px solid #b1dfbb;
    margin: 16px 0;
  }
  .verdict-diff {
    background: #f8d7da; color: #721c24;
    padding: 22px 32px; border-radius: 14px;
    font-size: 1.6rem; font-weight: 800;
    text-align: center; border: 2px solid #f1b0b7;
    margin: 16px 0;
  }
  .reason-box {
    background: #f1f3f5; border-radius: 10px;
    padding: 14px 20px; border-left: 5px solid #4a90d9;
    color: #333; font-size: 0.95rem; margin: 12px 0;
    line-height: 1.6;
  }
  .howto {
    background: #fffbe6; border: 1px solid #ffe08a;
    border-radius: 10px; padding: 12px 18px;
    font-size: 0.88rem; color: #555; margin-bottom: 18px;
  }
  .sim-bar-wrap {
    background: #e9ecef; border-radius: 100px;
    height: 14px; margin: 6px 0 2px 0;
  }
  .sim-bar-fill { border-radius: 100px; height: 14px; }
  div[data-testid="stMetric"] {
    background: #f8f9fa; border-radius: 10px;
    padding: 12px 16px; border: 1px solid #e9ecef;
  }
  div[data-testid="stMetricValue"] {
    font-size: 1.6rem !important; font-weight: 700 !important;
  }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Sample images (Wikipedia public domain) ──────────────────────────────────
SAMPLES = {
   "Golden Retriever":  "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Golden_Retriever_Dukedestiny01_drvd.jpg/240px-Golden_Retriever_Dukedestiny01_drvd.jpg",
    "Labrador":          "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Labrador_on_Quantock_%282175262184%29.jpg/240px-Labrador_on_Quantock_%282175262184%29.jpg",
    "Beagle":            "https://a-us.storyblok.com/f/1016262/1104x676/e36872ce32/beagle.png",
    "Siberian Husky":    "https://a-z-animals.com/media/Siberian-Husky-Canis-familiaris-at-beach.jpg",
    "Pug":               "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Mops_oct09_cropped2.jpg/240px-Mops_oct09_cropped2.jpg",
    "Siamese Cat":       "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Siam_lilacpoint.jpg/240px-Siam_lilacpoint.jpg",
}    


# ─── Session state ────────────────────────────────────────────────────────────
for key, default in [
    ("img1", None), ("img1_label", ""),
    ("img2", None), ("img2_label", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="wb-header"><h1>🐾 WhiskerBond</h1></div>
<div class="wb-sub">
  Pet Breed Identifier
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="howto">
  <b>How to use:</b>
  &nbsp;①&nbsp; Drag &amp; drop your pet photos into the boxes below, or click to browse.
  &nbsp;②&nbsp; Or paste an image URL.
  &nbsp;③&nbsp; Or click one of the <b>sample breed buttons</b> below to try instantly — no images needed.
  &nbsp;Then hit <b>Compare Breeds</b>.
</div>
""", unsafe_allow_html=True)


# ─── Sample picker ────────────────────────────────────────────────────────────
with st.expander("🖼️  Try with built-in sample breeds — click any to load", expanded=True):
    st.caption("First click → Slot 1.  Second click → Slot 2.  Try: Golden Retriever + Labrador, or Beagle + Pug.")
    sample_cols = st.columns(len(SAMPLES))
    for col, (name, url) in zip(sample_cols, SAMPLES.items()):
        with col:
            st.image(url, use_container_width=True)
            if st.button(name, key=f"sample_{name}", use_container_width=True):
                try:
                    with st.spinner(f"Loading {name}…"):
                        img = load_image(url)
                    if st.session_state.img1 is None:
                        st.session_state.img1 = img
                        st.session_state.img1_label = name
                    else:
                        st.session_state.img2 = img
                        st.session_state.img2_label = name
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load {name}: {e}")

st.divider()


# ─── Image slots ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

def render_slot(col, slot_num: int):
    img_key   = f"img{slot_num}"
    label_key = f"img{slot_num}_label"

    with col:
        st.markdown(f"### 📷 Pet Image {slot_num}")

        if st.session_state[img_key] is not None:
            st.image(
                st.session_state[img_key],
                use_container_width=True,
                caption=st.session_state[label_key] or f"Image {slot_num}",
            )
            if st.button(f"✕  Remove image {slot_num}", key=f"clear{slot_num}"):
                st.session_state[img_key] = None
                st.session_state[label_key] = ""
                st.rerun()
            return

        # Drag & drop / file browse
        uploaded = st.file_uploader(
            f"Drag & drop or click to browse",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key=f"uploader{slot_num}",
            help="Supports JPG, PNG, WebP, BMP",
        )
        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            st.session_state[img_key] = img
            st.session_state[label_key] = uploaded.name
            st.rerun()

        # URL input
        st.markdown(
            "<div style='text-align:center;color:#bbb;font-size:0.82rem;margin:10px 0'>— or paste a URL —</div>",
            unsafe_allow_html=True,
        )
        url_val = st.text_input(
            "Image URL",
            key=f"url_input{slot_num}",
            placeholder="https://example.com/my_pet.jpg",
            label_visibility="collapsed",
        )
        if url_val and st.button(f"Load URL → slot {slot_num}", key=f"load_url{slot_num}"):
            with st.spinner("Fetching image…"):
                try:
                    img = load_image(url_val.strip())
                    st.session_state[img_key] = img
                    st.session_state[label_key] = url_val.strip()
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load: {e}")

render_slot(col1, 1)
render_slot(col2, 2)

st.divider()


# ─── Compare button ───────────────────────────────────────────────────────────
img1 = st.session_state.img1
img2 = st.session_state.img2
both_ready = img1 is not None and img2 is not None

n_loaded = sum([img1 is not None, img2 is not None])
if not both_ready:
    st.info(
        f"📌  {n_loaded}/2 images loaded — "
        f"{'load one more image to compare' if n_loaded == 1 else 'load two images to get started'}"
    )

compare_btn = st.button(
    "🔍  Compare Breeds",
    type="primary",
    use_container_width=True,
    disabled=not both_ready,
)


# ─── Results ──────────────────────────────────────────────────────────────────
if compare_btn and both_ready:
    try:
        validate_image(img1)
        validate_image(img2)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.spinner("🔬  Extracting embeddings and comparing breeds…"):
        t0 = time.time()
        try:
            result = compare(img1, img2)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.exception(e)
            st.stop()
        elapsed = time.time() - t0

    st.divider()
    st.subheader("📊 Results")

    # Image previews with detected breed as caption
    r1, r2 = st.columns(2, gap="large")
    with r1:
        st.image(img1, use_container_width=True,
                 caption=f"Image 1 → {result.breed1} ({result.species1})")
    with r2:
        st.image(img2, use_container_width=True,
                 caption=f"Image 2 → {result.breed2} ({result.species2})")

    # Verdict
    if result.same_breed:
        st.markdown(
            f'<div class="verdict-same">✅  SAME BREED &nbsp;·&nbsp; {result.breed1}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="verdict-diff">❌  DIFFERENT BREEDS &nbsp;·&nbsp; {result.breed1} &nbsp;vs&nbsp; {result.breed2}</div>',
            unsafe_allow_html=True,
        )

    # Metrics
    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Match Confidence",   f"{result.confidence * 100:.1f}%")
    m2.metric("Similarity Score",   f"{result.similarity_score:.3f} / 1.0")
    m3.metric("Breed 1 Confidence", f"{result.confidence1 * 100:.0f}%")
    m4.metric("Breed 2 Confidence", f"{result.confidence2 * 100:.0f}%")

    # Similarity bar
    st.write("")
    sim = result.similarity_score
    bar_color  = "#28a745" if sim >= 0.78 else ("#ffc107" if sim >= 0.60 else "#dc3545")
    zone_label = "Same breed zone 🟢" if sim >= 0.78 else ("Uncertain zone 🟡" if sim >= 0.60 else "Different breed zone 🔴")
    st.markdown(f"**Visual similarity: {sim:.4f}** — {zone_label}")
    st.markdown(f"""
    <div class="sim-bar-wrap">
      <div class="sim-bar-fill" style="width:{int(sim*100)}%;background:{bar_color};"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#888;margin-top:3px;">
      <span>0.0 — Completely different</span>
      <span style="font-weight:700;color:{bar_color};">{sim:.3f}</span>
      <span>1.0 — Identical</span>
    </div>
    """, unsafe_allow_html=True)

    # Reason
    st.markdown(
        f'<div class="reason-box">💬 <b>Why this verdict:</b><br>{result.verdict_reason}</div>',
        unsafe_allow_html=True,
    )

    # Per-breed breakdown
    st.write("")
    bc1, bc2 = st.columns(2, gap="large")

    def breed_card(col, num, breed, species, conf, top5):
        with col:
            st.markdown(f"#### Image {num}: {breed}")
            st.caption(f"Species: {species.title()}")
            st.progress(conf, text=f"Detection confidence: {conf*100:.0f}%")
            with st.expander("Top-5 predictions"):
                for label, prob, _ in top5:
                    marker = "▶ " if label == breed else "   "
                    bold   = "**" if label == breed else ""
                    st.markdown(f"`{marker}`{bold}{label}{bold} — {prob*100:.2f}%")

    breed_card(bc1, 1, result.breed1, result.species1, result.confidence1, result.top5_1)
    breed_card(bc2, 2, result.breed2, result.species2, result.confidence2, result.top5_2)

    # Similar breeds
    if result.similar_breeds:
        st.write("")
        st.markdown("**🐕 Other breeds the model also considered:**")
        sc = st.columns(len(result.similar_breeds))
        for col, breed in zip(sc, result.similar_breeds):
            col.info(breed)

    # Threshold explainer
    with st.expander("📊 How does the similarity score work?"):
        st.markdown(f"""
| Score | Zone | Meaning |
|-------|------|---------|
| ≥ 0.78 | 🟢 Same breed | Feature embeddings are close — same breed |
| 0.60 – 0.78 | 🟡 Uncertain | Borderline — breed label used as tie-breaker |
| < 0.60 | 🔴 Different | Embeddings far apart — different breeds |

**Your score: `{sim:.4f}`**

The score is **cosine similarity** between two 1280-dimensional feature vectors
extracted by EfficientNet-B0 (pretrained on ImageNet). Thresholds calibrated
against the Oxford-IIIT Pets benchmark.
        """)

    # JSON download
    st.write("")
    result_dict = result.to_dict()
    result_dict["inference_time_sec"] = round(elapsed, 2)
    st.download_button(
        "💾  Download Full Result (JSON)",
        data=json.dumps(result_dict, indent=2),
        file_name="whiskerbond_result.json",
        mime="application/json",
        use_container_width=True,
    )
    st.caption(f"⚡ Completed in {elapsed:.2f}s")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "WhiskerBond 🐾"
)
