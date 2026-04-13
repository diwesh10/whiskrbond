#!/usr/bin/env python3
"""
WhiskerBond - CLI Demo
======================
Compare two pet images from the command line.

Usage:
  python demo_cli.py <image1> <image2> [--json]

Examples:
  python demo_cli.py golden1.jpg golden2.jpg
  python demo_cli.py https://example.com/dog1.jpg husky.jpg
  python demo_cli.py cat1.png cat2.png --json

Arguments:
  image1, image2  File paths or HTTP(S) URLs
  --json          Output raw JSON instead of pretty-printed results
  --help          Show this message
"""

import sys
import os
import json
import time
import argparse

# Make sure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.image_utils import load_image, validate_image
from models.comparator import compare


# ─── Colour codes for terminal output ────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def bar(value: float, width: int = 25) -> str:
    """Render a simple ASCII progress bar."""
    filled = int(value * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {value*100:.1f}%"


def print_banner():
    print(f"\n{BOLD}{'═'*58}")
    print(f"   🐾  WhiskerBond – Pet Breed Identifier")
    print(f"   Powered by EfficientNet-B0 + Oxford-IIIT / Stanford Dogs")
    print(f"{'═'*58}{RESET}")


def print_result(result, img1_src: str, img2_src: str):
    """Pretty-print the comparison result to the terminal."""

    print(f"\n{BOLD}📸  Input images{RESET}")
    print(f"  {DIM}Image 1:{RESET} {img1_src}")
    print(f"  {DIM}Image 2:{RESET} {img2_src}")

    print(f"\n{BOLD}{'─'*58}")
    print(f"  ANALYSIS RESULTS")
    print(f"{'─'*58}{RESET}")

    # Verdict
    if result.same_breed:
        v_color = GREEN
        verdict  = "✅  SAME BREED"
    else:
        v_color = RED
        verdict  = "❌  DIFFERENT BREED"

    print(f"\n  {v_color}{BOLD}{verdict}{RESET}")
    print(f"  {BOLD}Confidence:{RESET}        {bar(result.confidence)}")
    print(f"  {BOLD}Similarity score:{RESET}  {result.similarity_score:.4f}  "
          f"{DIM}(cosine, 0=different, 1=identical){RESET}")

    # Per-image breakdown
    print(f"\n{BOLD}  Image 1 — {result.breed1}{RESET}  ({result.species1})")
    print(f"    Classification confidence: {bar(result.confidence1)}")
    print(f"    {DIM}Top-5 predictions:{RESET}")
    for label, prob, _ in result.top5_1:
        marker = "▶" if label == result.breed1 else " "
        print(f"      {marker} {label:<35} {prob*100:5.1f}%")

    print(f"\n{BOLD}  Image 2 — {result.breed2}{RESET}  ({result.species2})")
    print(f"    Classification confidence: {bar(result.confidence2)}")
    print(f"    {DIM}Top-5 predictions:{RESET}")
    for label, prob, _ in result.top5_2:
        marker = "▶" if label == result.breed2 else " "
        print(f"      {marker} {label:<35} {prob*100:5.1f}%")

    # Reasoning
    print(f"\n{BOLD}  💬  Why this verdict:{RESET}")
    print(f"     {result.verdict_reason}")

    # Similar breeds
    if result.similar_breeds:
        print(f"\n{BOLD}  🐕  Similar breeds to consider:{RESET}")
        for i, breed in enumerate(result.similar_breeds, 1):
            print(f"     {i}. {breed}")

    print(f"\n{BOLD}{'═'*58}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        description="WhiskerBond: Compare two pet images for breed similarity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("image1", help="First pet image (file path or URL)")
    parser.add_argument("image2", help="Second pet image (file path or URL)")
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )
    parser.add_argument(
        "--save-json", metavar="FILE", help="Save JSON result to a file"
    )

    args = parser.parse_args()

    if not args.json:
        print_banner()

    # Load images
    try:
        if not args.json:
            print(f"\n⏳  Loading images...")

        img1 = load_image(args.image1)
        img2 = load_image(args.image2)
        validate_image(img1)
        validate_image(img2)

        if not args.json:
            print(f"    Image 1: {img1.size[0]}×{img1.size[1]}px")
            print(f"    Image 2: {img2.size[0]}×{img2.size[1]}px")
            print(f"\n⏳  Loading EfficientNet-B0 (downloads ~20MB on first run)...")
            print(f"    Extracting embeddings and running comparison...")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌  Error loading images: {e}", file=sys.stderr)
        sys.exit(1)

    # Run comparison
    try:
        t0 = time.time()
        result = compare(img1, img2)
        elapsed = time.time() - t0

    except Exception as e:
        print(f"\n❌  Comparison failed: {e}", file=sys.stderr)
        raise

    # Output
    result_dict = result.to_dict()
    result_dict["inference_time_sec"] = round(elapsed, 2)

    if args.json:
        print(json.dumps(result_dict, indent=2))
    else:
        print(f"\n✅  Done in {elapsed:.2f}s")
        print_result(result, args.image1, args.image2)

    # Optionally save JSON
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(result_dict, f, indent=2)
        if not args.json:
            print(f"💾  Full result saved to: {args.save_json}\n")

    # Exit code: 0 = same breed, 1 = different breed
    sys.exit(0 if result.same_breed else 1)


if __name__ == "__main__":
    main()
