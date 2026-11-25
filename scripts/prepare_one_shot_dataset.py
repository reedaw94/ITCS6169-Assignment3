
from pathlib import Path
import csv
import random
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "train"
MANIFEST_DIR = ROOT / "manifests"
OUTPUT = MANIFEST_DIR / "one_shot_manifest.csv"

SEED = 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED_CLASSES = [
    "Bedroom", "Coast", "Flower", "Forest",
    "Highway", "Industrial", "InsideCity", "Kitchen",
    "LivingRoom", "Mountain", "Office", "OpenCountry",
    "Store", "Street", "Suburb", "TallBuilding"
]


def main():
    rng = random.Random(SEED)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for label, class_name in enumerate(EXPECTED_CLASSES):
        class_dir = TRAIN_DIR / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        images = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        if not images:
            raise RuntimeError(
                f"No images found for class: {class_name}"
            )

        selected = rng.choice(images)

        rows.append({
            "class_id": label,
            "class_name": class_name,
            "image_path": str(selected.relative_to(ROOT))
        })

    counts = Counter(row["class_name"] for row in rows)

    assert len(rows) == 16
    assert len(counts) == 16
    assert all(count == 1 for count in counts.values())

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "class_name", "image_path"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Random seed: {SEED}")
    print(f"Selected images: {len(rows)}")
    print(f"Classes represented: {len(counts)}")
    print(f"Manifest saved: {OUTPUT}")

    print("\nOne-shot training samples:")
    for row in rows:
        print(
            f"{row['class_id']:2d} "
            f"{row['class_name']:15s} "
            f"{row['image_path']}"
        )


if __name__ == "__main__":
    main()
