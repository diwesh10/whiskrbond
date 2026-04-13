#!/usr/bin/env python3
"""
scripts/setup_weights.py
------------------------
Downloads and validates the EfficientNet-B0 pretrained weights.
Run this once after cloning the repo to pre-cache the model weights.

Usage:
  python scripts/setup_weights.py

What it does:
  1. Downloads EfficientNet-B0 weights from PyTorch Hub (~20MB)
  2. Saves to ~/.cache/torch/hub/checkpoints/ (standard torch cache)
  3. Runs a quick sanity check (dummy forward pass)
  4. Prints the embedding dimension and confirms everything is working

You only need to run this once. After that, the model loads from cache.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("\n" + "="*55)
    print("  WhiskerBond — Model Setup")
    print("="*55)

    print("\n📦  Step 1: Importing dependencies...")
    try:
        import torch
        import torchvision
        import numpy
        import sklearn
        import PIL
        print(f"  ✅ PyTorch       {torch.__version__}")
        print(f"  ✅ torchvision   {torchvision.__version__}")
        print(f"  ✅ NumPy         {numpy.__version__}")
        print(f"  ✅ scikit-learn  {sklearn.__version__}")
        print(f"  ✅ Pillow        {PIL.__version__}")
    except ImportError as e:
        print(f"\n❌  Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)

    print("\n📥  Step 2: Downloading EfficientNet-B0 weights (~20MB)...")
    print("   (This happens only once — cached at ~/.cache/torch/hub/)")
    t0 = time.time()

    try:
        from models.embedder import PetEmbedder
        embedder = PetEmbedder()
        embedder._load()  # Force weight download
        print(f"   ✅ Weights downloaded and cached in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"\n❌  Failed to download weights: {e}")
        print("   Check your internet connection and try again.")
        sys.exit(1)

    print("\n🧪  Step 3: Sanity check — dummy forward pass...")
    try:
        import numpy as np
        from PIL import Image

        dummy = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        emb = embedder.embed(dummy)
        top5 = embedder.classify(dummy, top_k=5)

        print(f"   ✅ Embedding shape:  {emb.shape}  (expected: (1280,))")
        print(f"   ✅ Embedding norm:   {np.linalg.norm(emb):.4f}  (expected: ~1.0)")
        print(f"   ✅ Top prediction:   {top5[0][0]}  ({top5[0][1]*100:.1f}%)")
    except Exception as e:
        print(f"\n❌  Sanity check failed: {e}")
        sys.exit(1)

    print("\n🧪  Step 4: Full pipeline test (two synthetic images)...")
    try:
        from models.comparator import compare

        img_a = Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        )
        img_b = Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        )

        result = compare(img_a, img_b)
        print(f"   ✅ Comparator works")
        print(f"   ✅ Similarity score: {result.similarity_score:.4f}")
        print(f"   ✅ Verdict:          {'Same' if result.same_breed else 'Different'} breed")
    except Exception as e:
        print(f"\n❌  Pipeline test failed: {e}")
        sys.exit(1)

    print("\n" + "="*55)
    print("  ✅  Setup complete! WhiskerBond is ready to use.")
    print()
    print("  CLI demo:   python demo_cli.py <image1> <image2>")
    print("  Web app:    streamlit run app/streamlit_app.py")
    print("  Tests:      python tests/test_comparator.py")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
