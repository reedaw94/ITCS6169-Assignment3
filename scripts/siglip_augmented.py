
from pathlib import Path
import csv
import torch
from PIL import Image, ImageEnhance
from transformers import AutoModel, AutoImageProcessor

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "one_shot_manifest.csv"
TEST_DIR = ROOT / "test"
RESULTS_DIR = ROOT / "results"

MODEL_NAME = "google/siglip-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def augment(image):
    width, height = image.size

    # Deterministic transformations of the same labeled image.
    views = [
        image,
        image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        ImageEnhance.Brightness(image).enhance(1.10),
        ImageEnhance.Contrast(image).enhance(1.10),
    ]

    # Mild center crop.
    crop_ratio = 0.90
    dx = int(width * (1 - crop_ratio) / 2)
    dy = int(height * (1 - crop_ratio) / 2)

    views.append(
        image.crop((dx, dy, width - dx, height - dy))
    )

    return views


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SigLIP:", MODEL_NAME)
    print("Device:", DEVICE)

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    @torch.inference_mode()
    def encode(images):
        vectors = []

        for start in range(0, len(images), BATCH_SIZE):
            batch = images[start:start + BATCH_SIZE]

            inputs = processor(
                images=batch,
                return_tensors="pt"
            ).to(DEVICE)

            features = model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )

            if hasattr(features, "pooler_output"):
                features = features.pooler_output

            features = torch.nn.functional.normalize(
                features.float(), dim=-1
            )

            vectors.append(features.cpu())

        return torch.cat(vectors, dim=0)

    with MANIFEST.open(newline="", encoding="utf-8") as f:
        references = list(csv.DictReader(f))

    assert len(references) == 16
    assert len({r["class_name"] for r in references}) == 16

    class_names = [r["class_name"] for r in references]

    print("Building augmented prototypes...")

    prototypes = []

    for row in references:
        image = load_image(ROOT / row["image_path"])
        views = augment(image)

        features = encode(views)
        prototype = features.mean(dim=0, keepdim=True)

        prototype = torch.nn.functional.normalize(
            prototype, dim=-1
        )

        prototypes.append(prototype)

    prototypes = torch.cat(prototypes, dim=0)

    test_paths = sorted(
        p for p in TEST_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    print(f"Encoding {len(test_paths)} test images...")

    test_features = []

    for start in range(0, len(test_paths), BATCH_SIZE):
        paths = test_paths[start:start + BATCH_SIZE]
        images = [load_image(p) for p in paths]
        test_features.append(encode(images))

    test_features = torch.cat(test_features, dim=0)

    scores = test_features @ prototypes.T
    predictions = scores.argmax(dim=1).tolist()

    output = RESULTS_DIR / "siglip_augmented_predictions.csv"

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "predicted_class",
            "similarity"
        ])

        for path, pred, score in zip(
            test_paths, predictions, scores
        ):
            writer.writerow([
                str(path.relative_to(ROOT)),
                class_names[pred],
                float(score[pred])
            ])

    print("Predictions saved:", output)

    if all(p.parent.name in class_names for p in test_paths):
        correct = sum(
            path.parent.name == class_names[pred]
            for path, pred in zip(test_paths, predictions)
        )

        print(f"Test accuracy: {correct / len(test_paths):.2%}")
        print(f"Correct: {correct}/{len(test_paths)}")


if __name__ == "__main__":
    main()
