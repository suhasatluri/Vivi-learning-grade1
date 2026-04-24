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
    """Detect the white card region and crop tightly to it.

    Strategy:
    1. Downsample + heavy Gaussian blur so colored shapes merge into the
       white card background, leaving a uniform bright blob vs darker table.
    2. Otsu threshold to separate card from table.
    3. Two-pass density scan: find card rows first (row density > threshold),
       then find card columns within only the card rows. This handles the
       card not filling the full image in either dimension.
    4. Find the longest contiguous run of card rows/cols = card bounds.
    """
    img = Image.open(path).convert("RGB")

    # --- Downsample for speed ---
    MAX_DIM = 600
    scale = 1.0
    if max(img.size) > MAX_DIM:
        scale = MAX_DIM / max(img.size)
        small = img.resize(
            (int(img.width * scale), int(img.height * scale)), Image.LANCZOS
        )
    else:
        small = img

    # --- Heavy blur to merge shapes into card background ---
    blur_radius = max(small.width, small.height) // 20
    blurred = small.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.array(blurred, dtype=np.float32)
    h, w = arr.shape[:2]
    gray = arr.mean(axis=2)

    # --- Otsu threshold ---
    flat = gray.flatten()
    total = len(flat)
    best_thresh, best_var = 0, 0
    for t in range(50, 220, 2):
        fg = flat[flat >= t]
        bg = flat[flat < t]
        if len(fg) == 0 or len(bg) == 0:
            continue
        var = (len(fg) / total) * (len(bg) / total) * (fg.mean() - bg.mean()) ** 2
        if var > best_var:
            best_var = var
            best_thresh = t

    mask = gray >= best_thresh

    # --- Helper: longest contiguous run of True ---
    def longest_run(arr_bool):
        best_start, best_len = 0, 0
        cur_start, cur_len = 0, 0
        for i, v in enumerate(arr_bool):
            if v:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
            else:
                cur_len = 0
        if best_len == 0:
            return 0, len(arr_bool) - 1
        return best_start, best_start + best_len - 1

    # --- Two-pass density detection with union for robustness ---
    # Pass A: rows first, then columns within those rows
    DENSITY_THRESH = 0.50
    row_density_a = mask.mean(axis=1)
    card_rows_a = row_density_a > DENSITY_THRESH
    rmin_a, rmax_a = longest_run(card_rows_a)
    card_strip_a = mask[rmin_a:rmax_a + 1, :]
    col_density_a = card_strip_a.mean(axis=0)
    card_cols_a = col_density_a > DENSITY_THRESH
    cmin_a, cmax_a = longest_run(card_cols_a)

    # Pass B: columns first, then rows within those columns
    col_density_b = mask.mean(axis=0)
    card_cols_b = col_density_b > DENSITY_THRESH
    cmin_b, cmax_b = longest_run(card_cols_b)
    card_strip_b = mask[:, cmin_b:cmax_b + 1]
    row_density_b = card_strip_b.mean(axis=1)
    card_rows_b = row_density_b > DENSITY_THRESH
    rmin_b, rmax_b = longest_run(card_rows_b)

    # Union: take the most inclusive bounds from both passes
    rmin = min(rmin_a, rmin_b)
    rmax = max(rmax_a, rmax_b)
    cmin = min(cmin_a, cmin_b)
    cmax = max(cmax_a, cmax_b)

    # --- Extend edges outward until we hit a steep brightness drop ---
    # This recovers shadowed card edges that the density threshold missed.
    # The card-to-table transition has a sharp brightness drop; within-card
    # shadows are more gradual.
    row_profile = gray.mean(axis=1)
    col_profile = gray.mean(axis=0)
    DROP_THRESH = 8  # brightness drop per pixel that signals card edge

    # Extend top edge upward
    while rmin > 0:
        drop = row_profile[rmin] - row_profile[rmin - 1]
        if drop > DROP_THRESH:
            break
        rmin -= 1

    # Extend bottom edge downward
    while rmax < h - 1:
        drop = row_profile[rmax] - row_profile[rmax + 1]
        if drop > DROP_THRESH:
            break
        rmax += 1

    # Extend left edge leftward
    while cmin > 0:
        drop = col_profile[cmin] - col_profile[cmin - 1]
        if drop > DROP_THRESH:
            break
        cmin -= 1

    # Extend right edge rightward
    while cmax < w - 1:
        drop = col_profile[cmax] - col_profile[cmax + 1]
        if drop > DROP_THRESH:
            break
        cmax += 1

    # --- Inward padding to trim rounded corners and shadow fringe ---
    card_h = rmax - rmin
    card_w = cmax - cmin
    pad_x = int(card_w * 0.015)
    pad_y = int(card_h * 0.015)
    rmin += pad_y
    rmax -= pad_y
    cmin += pad_x
    cmax -= pad_x

    # Clamp
    rmin = max(rmin, 0)
    cmin = max(cmin, 0)
    rmax = min(rmax, h - 1)
    cmax = min(cmax, w - 1)

    if rmax <= rmin or cmax <= cmin:
        return img

    # --- Scale back to original image coords ---
    orig_left = int(cmin / scale)
    orig_top = int(rmin / scale)
    orig_right = int(cmax / scale)
    orig_bottom = int(rmax / scale)

    orig_left = max(orig_left, 0)
    orig_top = max(orig_top, 0)
    orig_right = min(orig_right, img.width)
    orig_bottom = min(orig_bottom, img.height)

    return img.crop((orig_left, orig_top, orig_right, orig_bottom))


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

def reprocess_images():
    """Re-crop and re-compress all images using existing cards.json metadata (no API calls)."""
    if not CARDS_JSON.exists():
        print("No cards.json found — run without --reprocess first.")
        sys.exit(1)

    cards = json.loads(CARDS_JSON.read_text())
    total = len(cards)
    print(f"Re-processing {total} images with improved cropping\n")
    failures = []

    for i, card in enumerate(cards, 1):
        source = card.get("_source")
        if not source:
            print(f"[{i}/{total}]  {card['id']} — no _source field, skipping")
            failures.append((card["id"], "missing _source"))
            continue

        img_path = RAW_DIR / source
        if not img_path.exists():
            print(f"[{i}/{total}]  {card['id']} — source {source} not found, skipping")
            failures.append((source, "file not found"))
            continue

        try:
            print(f"[{i}/{total}]  {card['emoji']}  {card['label']}...", end="", flush=True)
            cropped = auto_crop_card(img_path)
            out_path = OUT_DIR / card["category"] / f"{card['id']}.jpg"
            resize_and_compress(cropped, out_path)
            size_kb = out_path.stat().st_size / 1024
            print(f"  → {size_kb:.1f} KB  ✓")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failures.append((source, str(e)))

    print(f"\nDone. {total - len(failures)} succeeded, {len(failures)} failed.")
    if failures:
        for fname, err in failures:
            print(f"  {fname}: {err}")


def main():
    if "--reprocess" in sys.argv:
        reprocess_images()
        return

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
                "_source": img_path.name,
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
