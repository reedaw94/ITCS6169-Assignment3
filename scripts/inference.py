
from pathlib import Path
import argparse
import csv

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = ROOT / "models" / "siglip_fusion_prototypes.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def main():
    parser = argparse.ArgumentParser(
        description="One-shot SigLIP scene classification"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "test",
        help="Image file or directory of images"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "inference_predictions.csv"
    )

    args = parser.parse_args()

    print("Device:", DEVICE)
    print("Loading classifier checkpoint:", CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    model_name = checkpoint["model_name"]
    class_names = checkpoint["class_names"]

    image_prototypes = checkpoint["image_prototypes"]
    text_prototypes = checkpoint["text_prototypes"]

    text_weight = checkpoint["text_weight"]
    image_weight = 1.0 - text_weight

    print("Foundation model:", model_name)
    print("Classes:", len(class_names))
    print("Text weight:", text_weight)

    processor = AutoImageProcessor.from_pretrained(
        model_name
    )

    model = AutoModel.from_pretrained(
        model_name
    ).to(DEVICE)

    model.eval()

    if args.input.is_file():
        paths = [args.input]
    else:
        paths = sorted(
            p for p in args.input.rglob("*")
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
        )

    if not paths:
        raise RuntimeError("No input images found.")

    print(f"Classifying {len(paths)} images...")

    predictions = []

    with torch.inference_mode():

        for start in range(0, len(paths), BATCH_SIZE):

            batch_paths = paths[start:start + BATCH_SIZE]

            images = [
                load_image(path)
                for path in batch_paths
            ]

            inputs = processor(
                images=images,
                return_tensors="pt"
            ).to(DEVICE)

            features = model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )

            if hasattr(features, "pooler_output"):
                features = features.pooler_output

            features = F.normalize(
                features.float(),
                dim=-1
            ).cpu()

            image_scores = features @ image_prototypes.T
            text_scores = features @ text_prototypes.T

            scores = (
                image_weight * image_scores
                + text_weight * text_scores
            )

            predicted_ids = scores.argmax(dim=1)

            for i, path in enumerate(batch_paths):

                class_id = predicted_ids[i].item()

                predictions.append({
                    "image_path": str(path),
                    "predicted_class": class_names[class_id],
                    "fused_score": float(scores[i, class_id])
                })

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "predicted_class",
                "fused_score"
            ]
        )

        writer.writeheader()
        writer.writerows(predictions)

    print("Predictions saved:", args.output)

    if all(p.parent.name in class_names for p in paths):

        correct = sum(
            path.parent.name == pred["predicted_class"]
            for path, pred in zip(paths, predictions)
        )

        print(f"Accuracy: {correct / len(paths):.2%}")
        print(f"Correct: {correct}/{len(paths)}")

    print("Inference complete.")


if __name__ == "__main__":
    main()
