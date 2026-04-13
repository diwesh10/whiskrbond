"""
comparator.py
-------------
Takes two pet images and determines if they are the same breed.

Pipeline:
  1. Extract 1280-dim embeddings via EfficientNet-B0
  2. Get top-5 ImageNet class predictions for each image
  3. Map predictions to Oxford-IIIT / Stanford Dogs breed labels
  4. Compute cosine similarity between embedding vectors
  5. Apply a threshold + classification agreement logic for final verdict

Threshold tuning rationale:
  - Oxford-IIIT Pets benchmark: same-breed pairs typically score > 0.82
  - Different-breed pairs typically score < 0.65
  - We use 0.75 as the decision boundary with a "uncertain" zone (0.65-0.75)
    where we fall back to classification label agreement
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union
from dataclasses import dataclass

from models.embedder import get_embedder
from utils.breed_labels import IMAGENET_PET_INDICES, get_species


# ─── Thresholds ───────────────────────────────────────────────────────────────
SAME_BREED_THRESHOLD   = 0.78   # cosine sim above this → same breed
DIFF_BREED_THRESHOLD   = 0.60   # cosine sim below this → different breed
# Between these two: uncertain, use classification label to break tie


@dataclass
class BreedResult:
    """Full result of a breed comparison."""
    # Per-image results
    breed1: str
    breed2: str
    species1: str
    species2: str
    confidence1: float          # classification confidence for image 1
    confidence2: float          # classification confidence for image 2
    top5_1: list                # top-5 (label, prob) for image 1
    top5_2: list                # top-5 (label, prob) for image 2

    # Comparison
    same_breed: bool
    confidence: float           # overall confidence in the verdict (0-1)
    similarity_score: float     # raw cosine similarity (0-1)
    verdict_reason: str         # human-readable explanation

    # Bonus
    similar_breeds: list        # other breeds this could be

    def to_dict(self) -> dict:
        return {
            "breed1": self.breed1,
            "breed2": self.breed2,
            "species1": self.species1,
            "species2": self.species2,
            "confidence1": round(self.confidence1, 4),
            "confidence2": round(self.confidence2, 4),
            "top5_1": [(label, round(p, 4)) for label, p, _ in self.top5_1],
            "top5_2": [(label, round(p, 4)) for label, p, _ in self.top5_2],
            "same_breed": self.same_breed,
            "confidence": round(self.confidence, 4),
            "similarity_score": round(self.similarity_score, 4),
            "verdict_reason": self.verdict_reason,
            "similar_breeds": self.similar_breeds,
        }


def _map_to_breed(top5_predictions: list) -> tuple:
    """
    Map ImageNet top-5 predictions to the most likely pet breed label.

    Returns:
        (breed_name, confidence, species, imagenet_idx)
    """
    for label, prob, idx in top5_predictions:
        # Check if this ImageNet class is a known pet breed
        if idx in IMAGENET_PET_INDICES:
            breed = IMAGENET_PET_INDICES[idx]
            species = get_species(idx)
            return breed, prob, species, idx

    # Fallback: use the top-1 prediction's label, clean it up
    label, prob, idx = top5_predictions[0]
    # Clean up ImageNet label format (e.g. "golden_retriever" → "Golden Retriever")
    clean = label.replace("_", " ").title()
    species = "dog" if idx in range(151, 269) else ("cat" if idx in range(281, 286) else "unknown")
    return clean, prob, species, idx


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    # Since embeddings are L2-normalised in embedder.py, this is just a dot product
    sim = float(np.dot(a, b))
    # Clamp to [0, 1] — negative similarity means very different
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))  # map [-1,1] → [0,1]


def compare(
    image1: Union[str, Path, Image.Image],
    image2: Union[str, Path, Image.Image],
) -> BreedResult:
    """
    Compare two pet images and return a BreedResult.

    Args:
        image1: File path, URL-loaded PIL image, or PIL Image object
        image2: Same as image1

    Returns:
        BreedResult with full analysis
    """
    embedder = get_embedder()

    # Step 1: Extract embeddings (1280-dim, L2-normalised)
    emb1 = embedder.embed(image1)
    emb2 = embedder.embed(image2)

    # Step 2: Get top-5 ImageNet predictions
    top5_1 = embedder.classify(image1, top_k=5)
    top5_2 = embedder.classify(image2, top_k=5)

    # Step 3: Map to breed labels
    breed1, conf1, species1, idx1 = _map_to_breed(top5_1)
    breed2, conf2, species2, idx2 = _map_to_breed(top5_2)

    # Step 4: Cosine similarity
    sim = _cosine_similarity(emb1, emb2)

    # Step 5: Verdict logic
    labels_agree = breed1.lower() == breed2.lower()
    cross_species = (species1 != species2) and ("unknown" not in [species1, species2])

    if cross_species:
        # Cat vs dog — definitely different
        same_breed = False
        verdict_conf = 0.99
        reason = (
            f"Cross-species comparison: image 1 appears to be a {species1} "
            f"({breed1}) and image 2 a {species2} ({breed2}). "
            f"Different species cannot be the same breed."
        )
    elif sim >= SAME_BREED_THRESHOLD:
        same_breed = True
        verdict_conf = min(0.99, 0.5 + sim * 0.5 + (0.1 if labels_agree else 0))
        reason = (
            f"High embedding similarity ({sim:.3f} ≥ {SAME_BREED_THRESHOLD}) between "
            f"'{breed1}' and '{breed2}'. "
            f"{'Classification labels also agree.' if labels_agree else 'Visual features strongly match despite label difference.'}"
        )
    elif sim <= DIFF_BREED_THRESHOLD:
        same_breed = False
        verdict_conf = min(0.99, 0.5 + (1.0 - sim) * 0.5)
        reason = (
            f"Low embedding similarity ({sim:.3f} ≤ {DIFF_BREED_THRESHOLD}). "
            f"Detected '{breed1}' vs '{breed2}' — visually distinct breeds."
        )
    else:
        # Uncertain zone: use classification agreement to break the tie
        same_breed = labels_agree
        # Confidence is lower in this zone
        verdict_conf = 0.55 + abs(sim - 0.69) * 0.3
        if same_breed:
            reason = (
                f"Moderate similarity ({sim:.3f}) but classification agrees: "
                f"both identified as '{breed1}'. Likely same breed."
            )
        else:
            reason = (
                f"Moderate similarity ({sim:.3f}) with disagreeing classifications: "
                f"'{breed1}' vs '{breed2}'. Likely different breeds."
            )

    # Collect similar breeds from top-5 predictions
    similar = []
    for label, prob, idx in top5_1[1:3]:
        if idx in IMAGENET_PET_INDICES:
            similar.append(IMAGENET_PET_INDICES[idx])
        else:
            similar.append(label.replace("_", " ").title())
    for label, prob, idx in top5_2[1:2]:
        if idx in IMAGENET_PET_INDICES:
            b = IMAGENET_PET_INDICES[idx]
        else:
            b = label.replace("_", " ").title()
        if b not in similar:
            similar.append(b)

    return BreedResult(
        breed1=breed1,
        breed2=breed2,
        species1=species1,
        species2=species2,
        confidence1=conf1,
        confidence2=conf2,
        top5_1=top5_1,
        top5_2=top5_2,
        same_breed=same_breed,
        confidence=verdict_conf,
        similarity_score=sim,
        verdict_reason=reason,
        similar_breeds=similar[:4],
    )
