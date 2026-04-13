"""
embedder.py
-----------
Extracts 1280-dimensional embeddings from pet images using
EfficientNet-B0 pretrained on ImageNet.

The key idea:
  - Remove the final classification head from EfficientNet-B0
  - Use the 1280-dim feature vector BEFORE the classifier as the embedding
  - These embeddings capture visual semantics learned from 1.2M images
  - Same breed → similar embeddings → high cosine similarity
  - Different breed → different embeddings → low cosine similarity

Weight download: ~20MB, happens once on first run, cached at:
  ~/.cache/torch/hub/checkpoints/
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union


# ─── Image preprocessing (must match ImageNet training) ───────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = T.Compose([
    T.Resize(256),           # Resize shortest edge to 256
    T.CenterCrop(224),       # Crop to 224×224 (EfficientNet-B0 input)
    T.ToTensor(),            # [0,255] uint8 → [0.0,1.0] float
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # ImageNet normalisation
])

EMBEDDING_DIM = 1280  # EfficientNet-B0 penultimate layer output size


class PetEmbedder:
    """
    Wraps EfficientNet-B0 to produce breed embeddings.

    Usage:
        embedder = PetEmbedder()
        emb = embedder.embed("dog.jpg")          # numpy array shape (1280,)
        top5 = embedder.classify("dog.jpg")      # list of (breed, score) pairs
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._model = None
        self._full_model = None  # Full classifier for breed prediction

    def _load(self):
        """Lazy-load weights on first use."""
        if self._model is not None:
            return

        # Load full EfficientNet-B0 with ImageNet weights
        # This downloads ~20MB on first run and caches it
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        full = models.efficientnet_b0(weights=weights)
        full.eval()
        full.to(self.device)

        # Embedding model: remove the final Linear classifier
        # EfficientNet architecture:
        #   features (Conv + MBConv blocks)
        #   avgpool (AdaptiveAvgPool2d → 1×1)
        #   classifier (Dropout + Linear 1280→1000)
        # We want everything EXCEPT the classifier
        self._model = nn.Sequential(
            full.features,
            full.avgpool,
            nn.Flatten(),    # [B, 1280, 1, 1] → [B, 1280]
        )
        self._model.eval()

        # Keep full model for breed classification (top-k predictions)
        self._full_model = full

        # Load ImageNet class labels
        self._imagenet_labels = weights.meta["categories"]

    def _preprocess(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Load and preprocess an image into a model-ready tensor."""
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"Expected path or PIL Image, got {type(image)}")

        return TRANSFORM(img).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]

    @torch.no_grad()
    def embed(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Compute the 1280-dimensional embedding for an image.

        Returns:
            np.ndarray of shape (1280,) — L2-normalised embedding vector
        """
        self._load()
        tensor = self._preprocess(image)
        embedding = self._model(tensor).cpu().numpy().squeeze()  # (1280,)

        # L2-normalise so cosine similarity == dot product
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    @torch.no_grad()
    def classify(self, image: Union[str, Path, Image.Image], top_k: int = 5) -> list:
        """
        Get top-k ImageNet class predictions with confidence scores.

        Returns:
            list of (class_name, probability) sorted by probability descending
        """
        self._load()
        tensor = self._preprocess(image)
        logits = self._full_model(tensor)               # [1, 1000]
        probs = torch.softmax(logits, dim=1)[0]         # [1000]
        top_probs, top_indices = torch.topk(probs, top_k)

        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            label = self._imagenet_labels[idx]
            results.append((label, float(prob), int(idx)))
        return results


# ─── Module-level singleton ───────────────────────────────────────────────────
_embedder_instance = None

def get_embedder() -> PetEmbedder:
    """Return a shared PetEmbedder instance (loaded once per process)."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = PetEmbedder()
    return _embedder_instance
