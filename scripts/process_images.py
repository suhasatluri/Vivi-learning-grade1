#!/usr/bin/env python3
"""
Process raw shape-card photos:
  1. Analyse each image with Claude Vision to get metadata
  2. Auto-crop to the white card region
  3. Resize to 480x340, compress to ≤20 KB
  4. Save to assets/shape-cards/{Category}/{name}.jpg
  5. Build assets/shape-cards/cards.json
"""

import base64, json, os, sys, time
from pathlib import Path
from io import BytesIO

# Load .env file (no external dependency needed)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import anthropic
from PIL import Image, ImageFilter
import numpy as np

# ── paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw-images"
OUT_DIR = ROOT / "assets" / "shape-cards"
CARDS_JSON = OUT_DIR / "cards.json"

VALID_CATEGORIES = {"Animals", "Vehicles", "Nature", "Objects", "People", "Food"}
TARGET_W, TARGET_H = 480, 340
MAX_BYTES = 20 * 1024  # 20 KB

client = anthropic.Anthropic()

# ── helpers ──────────────────────────────────────────────────────────

def encode_image_for_api(path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) sized for the API (≤1568px long edge)."""
    img = Image.open(path)
    img = img.convert("RGB")
    # Downscale for API efficiency
    max_dim = 1568
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg"


def analyse_image(path: Path) -> dict:
    """Call Claude Vision and return structured metadata."""
    b64, media = encode_image_for_api(path)
    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                {"type": "text", "text": (
                    "This is a photo of a children's shape-art flash card. "
                    "The card shows a picture made from geometric shapes and has a label at the bottom.\n\n"
                    "Return ONLY a JSON object (no markdown fences) with these fields:\n"
                    '  "name": short lowercase slug (e.g. "fish", "fire-truck"),\n'
                    '  "label": display name exactly as printed on the card,\n'
                    '  "emoji": one emoji that best represents the subject,\n'
                    '  "category": exactly one of Animals | Vehicles | Nature | Objects | People | Food,\n'
                    '  "shapes_used": array of geometric shape names used (e.g. "circle", "triangle", "rectangle", "semicircle", "oval", "square", "trapezoid", "diamond", "hexagon", "crescent", "star")\n'
                )}
            ]
        }]
    )
    text = msg.content[0].text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    data = json.loads(text)

    # Validate / fix category
    if data.get("category") not in VALID_CATEGORIES:
        data["category"] = "Objects"
    return data


def auto_crop_card(path: Path) -> Image.Image:
    """Detect the white card region and crop to it."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    # White-ish mask: all channels > 180 and low variance across channels
    channel_min = arr.min(axis=2)
    channel_max = arr.max(axis=2)
    bright = channel_min > 160
    low_spread = (channel_max.astype(int) - channel_min.astype(int)) < 50
    mask = (bright & low_spread).astype(np.uint8) * 255

    # Clean up the mask with morphological operations
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.filter(ImageFilter.MedianFilter(7))
    mask = np.array(mask_img)

    # Find bounding box of the largest white region
    rows = np.any(mask > 128, axis=1)
    cols = np.any(mask > 128, axis=0)
    if not rows.any() or not cols.any():
        return img  # fallback: return full image

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add small padding inward to trim card edge artifacts
    h, w = arr.shape[:2]
    pad = int(min(h, w) * 0.01)
    rmin = min(rmin + pad, rmax)
    rmax = max(rmax - pad, rmin)
    cmin = min(cmin + pad, cmax)
    cmax = max(cmax - pad, cmin)

    cropped = img.crop((cmin, rmin, cmax, rmax))
    return cropped


def resize_and_compress(img: Image.Image, out_path: Path):
    """Resize to TARGET dimensions and compress to ≤ MAX_BYTES."""
    # Resize to fit within 480x340, maintaining aspect ratio, then pad
    img = img.convert("RGB")

    # Fit within target box
    img.thumbnail((TARGET_W, TARGET_H), Image.LANCZOS)

    # Create white canvas and paste centered
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), (255, 255, 255))
    x = (TARGET_W - img.width) // 2
    y = (TARGET_H - img.height) // 2
    canvas.paste(img, (x, y))

    # Binary search for quality that fits under MAX_BYTES
    lo, hi = 10, 85
    best_buf = None
    while lo <= hi:
        mid = (lo + hi) // 2
        buf = BytesIO()
        canvas.save(buf, format="JPEG", quality=mid, optimize=True)
        if buf.tell() <= MAX_BYTES:
            best_buf = buf
            lo = mid + 1
        else:
            hi = mid - 1

    if best_buf is None:
        # Even quality=10 is too big — scale down more
        for scale in [0.8, 0.6, 0.5, 0.4]:
            smaller = canvas.resize(
                (int(TARGET_W * scale), int(TARGET_H * scale)), Image.LANCZOS
            )
            padded = Image.new("RGB", (TARGET_W, TARGET_H), (255, 255, 255))
            px = (TARGET_W - smaller.width) // 2
            py = (TARGET_H - smaller.height) // 2
            padded.paste(smaller, (px, py))
            buf = BytesIO()
            padded.save(buf, format="JPEG", quality=20, optimize=True)
            if buf.tell() <= MAX_BYTES:
                best_buf = buf
                break
        if best_buf is None:
            # Last resort
            buf = BytesIO()
            canvas.save(buf, format="JPEG", quality=10, optimize=True)
            best_buf = buf

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(best_buf.getvalue())


# ── main ─────────────────────────────────────────────────────────────

def main():
    images = sorted([
        f for f in RAW_DIR.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])

    if not images:
        print("No images found in raw-images/")
        sys.exit(1)

    print(f"Found {len(images)} images to process\n")

    cards = []
    failures = []
    category_counts = {}
    seen_names = set()

    consecutive_fails = 0
    last_error = ""

    for i, img_path in enumerate(images, 1):
        tag = f"[{i}/{len(images)}] {img_path.name}"
        try:
            # Step 1: Analyse with Claude Vision
            print(f"{tag}  analysing...", end="", flush=True)
            meta = analyse_image(img_path)
            name = meta["name"]

            # Deduplicate names
            if name in seen_names:
                suffix = 2
                while f"{name}-{suffix}" in seen_names:
                    suffix += 1
                name = f"{name}-{suffix}"
                meta["name"] = name
            seen_names.add(name)

            print(f"  {meta['emoji']}  {meta['label']} ({meta['category']})", end="", flush=True)

            # Step 2: Crop
            cropped = auto_crop_card(img_path)

            # Step 3: Resize + compress + save
            cat = meta["category"]
            out_path = OUT_DIR / cat / f"{name}.jpg"
            resize_and_compress(cropped, out_path)
            size_kb = out_path.stat().st_size / 1024
            print(f"  → {size_kb:.1f} KB  ✓")

            # Step 4: Record
            cards.append({
                "id": name,
                "label": meta["label"],
                "emoji": meta["emoji"],
                "category": cat,
                "img": f"assets/shape-cards/{cat}/{name}.jpg",
                "shapes": meta["shapes_used"],
            })
            category_counts[cat] = category_counts.get(cat, 0) + 1
            consecutive_fails = 0

        except Exception as e:
            err_str = str(e)
            print(f"  ✗ FAILED: {err_str}")
            failures.append((img_path.name, err_str))
            consecutive_fails += 1
            # Abort early if same error repeats (model/auth issue)
            if consecutive_fails >= 3 and ("404" in err_str or "auth" in err_str.lower()):
                print(f"\n  ⚠ Aborting: {consecutive_fails} consecutive failures with same error type.")
                break

        # Small delay to avoid rate limits
        if i < len(images):
            time.sleep(0.5)

    # Write cards.json
    cards.sort(key=lambda c: (c["category"], c["id"]))
    CARDS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(CARDS_JSON, "w") as f:
        json.dump(cards, f, indent=2, ensure_ascii=False)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  PROCESSING COMPLETE")
    print("=" * 55)
    print(f"  Total images:     {len(images)}")
    print(f"  Succeeded:        {len(cards)}")
    print(f"  Failed:           {len(failures)}")
    print("-" * 55)
    print("  Category breakdown:")
    for cat in sorted(category_counts):
        print(f"    {cat:15s}  {category_counts[cat]:3d}")
    print("-" * 55)
    if failures:
        print("  Failures:")
        for fname, err in failures:
            print(f"    {fname}: {err}")
        print("-" * 55)
    print(f"  Output:  {CARDS_JSON}")
    print("=" * 55)


if __name__ == "__main__":
    main()
