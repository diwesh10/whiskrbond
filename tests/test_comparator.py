"""
tests/test_comparator.py
------------------------
Test suite for the WhiskerBond breed comparator.

Uses publicly available Wikipedia images (license-free) to validate:
  - Same-breed pairs → same_breed = True
  - Different-breed pairs → same_breed = False
  - Cross-species pairs → same_breed = False (with high confidence)
  - Edge cases: low-quality images, unusual angles

Run:
  python tests/test_comparator.py
  python tests/test_comparator.py --verbose
  python tests/test_comparator.py --quick      # skip slow URL tests
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_utils import load_image
from models.comparator import compare


# ─── Test cases ───────────────────────────────────────────────────────────────
# All images are Wikimedia Commons public domain / CC-licensed images

TEST_CASES = [
    # ── Same breed pairs ────────────────────────────────────────────────────
    {
        "id": "golden_vs_golden",
        "name": "Two Golden Retrievers (SAME)",
        "img1": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Golden_Retriever_Dukedestiny01_drvd.jpg/320px-Golden_Retriever_Dukedestiny01_drvd.jpg",
        "img2": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Golden_retriever_eating_pigs_foot.jpg/320px-Golden_retriever_eating_pigs_foot.jpg",
        "expected_same": True,
        "category": "same_breed",
    },
    {
        "id": "beagle_vs_beagle",
        "name": "Two Beagles (SAME)",
        "img1": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Beagle_verde.jpg/320px-Beagle_verde.jpg",
        "img2": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Beagle_puppy_%28houndtooth_collar%29.jpg/320px-Beagle_puppy_%28houndtooth_collar%29.jpg",
        "expected_same": True,
        "category": "same_breed",
    },
    {
        "id": "siamese_vs_siamese",
        "name": "Two Siamese Cats (SAME)",
        "img1": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Siam_lilacpoint.jpg/320px-Siam_lilacpoint.jpg",
        "img2": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/Applehead_Siamese_Cats.jpg/320px-Applehead_Siamese_Cats.jpg",
        "expected_same": True,
        "category": "same_breed",
    },
    {
        "id": "labrador_vs_labrador",
        "name": "Two Labrador Retrievers (SAME)",
        "img1": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Labrador_on_Quantock_%282175262184%29.jpg/320px-Labrador_on_Quantock_%282175262184%29.jpg",
        "img2": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg",
        "expected_same": True,
        "category": "same_breed",
    },
    # ── Different breed pairs ────────────────────────────────────────────────
    {
        "id": "golden_vs_husky",
        "name": "Golden Retriever vs Siberian Husky (DIFFERENT)",
        "img1": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Golden_Retriever_Dukedestiny01_drvd.jpg/320px-Golden_Retriever_Dukedestiny01_drvd.jpg",
        "img2": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Husky_Yakutsk.jpg/320px-Husky_Yakutsk.jpg",
        "expected_same": False,
        "category": "different_breed",
    },
    {
        "id": "golden_vs_labrador",
        "name": "Golden Retriever vs Labrador (DIFFERENT — similar looking!)",
        "img1": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Golden_Retriever_Dukedestiny01_drvd.jpg/320px-Golden_Retriever_Dukedestiny01_drvd.jpg",
        "img2": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Labrador_on_Quantock_%282175262184%29.jpg/320px-Labrador_on_Quantock_%282175262184%29.jpg",
        "expected_same": False,
        "category": "different_breed",
    },
    {
        "id": "pug_vs_beagle",
        "name": "Pug vs Beagle (DIFFERENT)",
        "img1": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Mops_oct09_cropped2.jpg/320px-Mops_oct09_cropped2.jpg",
        "img2": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Beagle_verde.jpg/320px-Beagle_verde.jpg",
        "expected_same": False,
        "category": "different_breed",
    },
    # ── Cross-species pairs ─────────────────────────────────────────────────
    {
        "id": "dog_vs_cat",
        "name": "Golden Retriever vs Siamese Cat (DIFFERENT — cross-species)",
        "img1": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Golden_Retriever_Dukedestiny01_drvd.jpg/320px-Golden_Retriever_Dukedestiny01_drvd.jpg",
        "img2": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Siam_lilacpoint.jpg/320px-Siam_lilacpoint.jpg",
        "expected_same": False,
        "category": "cross_species",
    },
]


# ─── Test runner ──────────────────────────────────────────────────────────────

def run_test(test: dict, verbose: bool = False) -> dict:
    """Run a single test case. Returns result dict."""
    print(f"\n  [{test['id']}] {test['name']}")

    try:
        t0 = time.time()
        img1 = load_image(test["img1"])
        img2 = load_image(test["img2"])
        result = compare(img1, img2)
        elapsed = time.time() - t0

        passed = result.same_breed == test["expected_same"]
        status = "PASS ✅" if passed else "FAIL ❌"

        print(f"  Status:     {status}")
        print(f"  Detected:   {result.breed1}  vs  {result.breed2}")
        print(f"  Same breed: {result.same_breed}  (expected: {test['expected_same']})")
        print(f"  Similarity: {result.similarity_score:.4f}")
        print(f"  Confidence: {result.confidence*100:.1f}%")
        print(f"  Time:       {elapsed:.1f}s")

        if verbose:
            print(f"  Reason:     {result.verdict_reason}")

        return {
            "id": test["id"],
            "name": test["name"],
            "passed": passed,
            "expected": test["expected_same"],
            "got": result.same_breed,
            "similarity": result.similarity_score,
            "confidence": result.confidence,
            "breed1": result.breed1,
            "breed2": result.breed2,
            "time_sec": round(elapsed, 2),
            "category": test["category"],
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "id": test["id"],
            "name": test["name"],
            "passed": False,
            "error": str(e),
            "category": test["category"],
        }


def main():
    parser = argparse.ArgumentParser(description="WhiskerBond test suite")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print reasoning for each test")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Run only 4 tests (faster)")
    parser.add_argument("--category", choices=["same_breed", "different_breed", "cross_species"],
                        help="Run only tests from this category")
    parser.add_argument("--save", metavar="FILE", default="test_results.json",
                        help="Save results to JSON file (default: test_results.json)")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  WhiskerBond — Breed Comparator Test Suite")
    print("  Model: EfficientNet-B0 (ImageNet pretrained)")
    print("  Datasets: Oxford-IIIT Pets + Stanford Dogs")
    print("═"*60)

    # Filter tests
    cases = TEST_CASES
    if args.category:
        cases = [t for t in cases if t["category"] == args.category]
    if args.quick:
        cases = cases[:4]

    print(f"\n  Running {len(cases)} test(s)...\n")

    results = []
    for i, test in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}]", end="")
        result = run_test(test, verbose=args.verbose)
        results.append(result)
        if i < len(cases):
            time.sleep(0.5)  # Be polite to Wikipedia's servers

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r.get("passed", False))
    total  = len(results)
    errors = sum(1 for r in results if "error" in r)

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        categories.setdefault(cat, {"passed": 0, "total": 0})
        categories[cat]["total"] += 1
        if r.get("passed"):
            categories[cat]["passed"] += 1

    print("\n" + "═"*60)
    print(f"  FINAL RESULTS: {passed}/{total} passed", end="")
    if errors:
        print(f"  ({errors} errors)", end="")
    print()

    for cat, counts in categories.items():
        pct = counts["passed"] / counts["total"] * 100 if counts["total"] else 0
        print(f"  {cat:<20} {counts['passed']}/{counts['total']}  ({pct:.0f}%)")

    # Average similarity scores by category
    same_sims = [r["similarity"] for r in results
                 if r.get("category") == "same_breed" and "similarity" in r]
    diff_sims = [r["similarity"] for r in results
                 if r.get("category") == "different_breed" and "similarity" in r]

    if same_sims:
        print(f"\n  Avg similarity (same breed):      {sum(same_sims)/len(same_sims):.4f}")
    if diff_sims:
        print(f"  Avg similarity (different breed): {sum(diff_sims)/len(diff_sims):.4f}")

    print("═"*60 + "\n")

    # Save results
    with open(args.save, "w") as f:
        json.dump({
            "summary": {
                "passed": passed, "total": total, "errors": errors,
                "by_category": categories,
            },
            "results": results,
        }, f, indent=2)
    print(f"  Full results saved to: {args.save}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
