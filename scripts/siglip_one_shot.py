
from pathlib import Path
import csv

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "one_shot_manifest.csv"
TEST_DIR = ROOT / "test"
RESULTS_DIR = ROOT / "results"

MODEL_NAME = "google/siglip-base-patch16-224"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH_SIZE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print("Loading SigLIP:", MODEL_NAME)

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    with MANIFEST.open(newline="", encoding="utf-8") as f:
        references = list(csv.DictReader(f))

    assert len(references) == 16
    assert len({r["class_name"] for r in references}) == 16

    class_names = [r["class_name"] for r in references]

    @torch.inference_mode()
    def encode(paths):
        embeddings = []

        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start:start + BATCH_SIZE]
            images = [load_image(p) for p in batch_paths]

            inputs = processor(
                images=images,
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

            embeddings.append(features.cpu())

        return torch.cat(embeddings, dim=0)

    reference_paths = [
        ROOT / row["image_path"]
        for row in references
    ]

    print("Encoding 16 reference images...")
    prototypes = encode(reference_paths)

    test_paths = sorted(
        p for p in TEST_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if not test_paths:
        raise RuntimeError("No test images found.")

    print(f"Encoding {len(test_paths)} test images...")
    test_embeddings = encode(test_paths)

    similarities = test_embeddings @ prototypes.T
    predictions = similarities.argmax(dim=1).numpy()

    output = RESULTS_DIR / "siglip_one_shot_predictions.csv"

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "predicted_class",
            "confidence_similarity"
        ])

        for path, pred, scores in zip(
            test_paths, predictions, similarities
        ):
            writer.writerow([
                str(path.relative_to(ROOT)),
                class_names[int(pred)],
                float(scores[int(pred)])
            ])

    print("Predictions saved:", output)

    if all(p.parent.name in class_names for p in test_paths):
        true_labels = [p.parent.name for p in test_paths]
        predicted_labels = [
            class_names[int(pred)] for pred in predictions
        ]

        correct = sum(
            actual == predicted
            for actual, predicted in zip(
                true_labels, predicted_labels
            )
        )

        accuracy = correct / len(test_paths)

        print(f"Test accuracy: {accuracy:.2%}")
        print(f"Correct: {correct}/{len(test_paths)}")
    else:
        print("Test labels not identified from folder structure.")


if __name__ == "__main__":
    main()
