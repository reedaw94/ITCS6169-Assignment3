
from pathlib import Path
import csv
import json

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "one_shot_manifest.csv"
TEST_DIR = ROOT / "test"
RESULTS_DIR = ROOT / "results"

MODEL_NAME = "google/siglip-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
TEXT_WEIGHT = 0.25

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Fixed, generic class descriptions.
# These do not describe any additional labeled examples.
DESCRIPTIONS = {
    "Bedroom": "A photograph of a bedroom with a bed and bedroom furniture.",
    "Coast": "A photograph of a coastline with ocean water and a shore.",
    "Flower": "A photograph of flowers, petals, and plants.",
    "Forest": "A photograph of a forest with many trees and vegetation.",
    "Highway": "A photograph of a highway with road lanes and vehicles.",
    "Industrial": "A photograph of an industrial area with factories or machinery.",
    "InsideCity": "A photograph of an urban city scene with buildings and streets.",
    "Kitchen": "A photograph of a kitchen with cabinets and cooking appliances.",
    "LivingRoom": "A photograph of a living room with sofas and furniture.",
    "Mountain": "A photograph of mountains and mountainous landscape.",
    "Office": "A photograph of an office with desks and office equipment.",
    "OpenCountry": "A photograph of open countryside with fields and landscape.",
    "Store": "A photograph of a store with shelves and merchandise.",
    "Street": "A photograph of a street with buildings, roads, or pedestrians.",
    "Suburb": "A photograph of a suburban neighborhood with houses.",
    "TallBuilding": "A photograph of tall buildings and skyscrapers."
}


def load_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def augment(image):
    width, height = image.size

    views = [
        image,
        image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        ImageEnhance.Brightness(image).enhance(1.10),
        ImageEnhance.Contrast(image).enhance(1.10),
    ]

    dx = int(width * 0.05)
    dy = int(height * 0.05)

    views.append(
        image.crop((dx, dy, width - dx, height - dy))
    )

    return views


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print("Model:", MODEL_NAME)
    print("Text weight:", TEXT_WEIGHT)

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    with MANIFEST.open(newline="", encoding="utf-8") as f:
        references = list(csv.DictReader(f))

    class_names = [r["class_name"] for r in references]

    assert len(references) == 16
    assert len(set(class_names)) == 16
    assert set(class_names) == set(DESCRIPTIONS)

    @torch.inference_mode()
    def encode_images(images):
        vectors = []

        for start in range(0, len(images), BATCH_SIZE):
            batch = images[start:start + BATCH_SIZE]

            inputs = processor.image_processor(
                images=batch,
                return_tensors="pt"
            ).to(DEVICE)

            features = model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )

            if hasattr(features, "pooler_output"):
                features = features.pooler_output

            features = F.normalize(features.float(), dim=-1)
            vectors.append(features.cpu())

        return torch.cat(vectors, dim=0)

    
  
    @torch.inference_mode()
    def encode_texts(texts):
        inputs = processor.tokenizer(
            texts,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)

        text_inputs = {
            "input_ids": inputs["input_ids"]
        }

        if "attention_mask" in inputs:
            text_inputs["attention_mask"] = inputs["attention_mask"]

        features = model.get_text_features(**text_inputs)

        if hasattr(features, "pooler_output"):
            features = features.pooler_output

        return F.normalize(
            features.float(),
            dim=-1
        ).cpu()

    print("Building augmented image prototypes...")

    image_prototypes = []

    for row in references:
        image = load_image(ROOT / row["image_path"])
        views = augment(image)

        features = encode_images(views)
        prototype = F.normalize(
            features.mean(dim=0, keepdim=True),
            dim=-1
        )

        image_prototypes.append(prototype)

    image_prototypes = torch.cat(image_prototypes, dim=0)

    print("Encoding class descriptions...")

    texts = [DESCRIPTIONS[name] for name in class_names]
    text_prototypes = encode_texts(texts)

    print("Text embedding shape:", text_prototypes.shape)

    if not torch.isfinite(text_prototypes).all():
        raise RuntimeError(
            "Text embeddings contain NaN or infinite values."
        )

    print("Text embeddings validated successfully.")

     # Save the final one-shot classifier prototypes.
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        models_dir / "siglip_fusion_prototypes.pt"
    )

    torch.save(
        {
            "model_name": MODEL_NAME,
            "class_names": class_names,
            "image_prototypes": image_prototypes.cpu(),
            "text_prototypes": text_prototypes.cpu(),
            "text_weight": TEXT_WEIGHT,
            "image_weight": 1.0 - TEXT_WEIGHT,
            "reference_augmentation_views": 5,
            "labeled_training_images": 16,
            "seed": 42,
            "descriptions": DESCRIPTIONS
        },
        checkpoint_path
    )

    print("Classifier checkpoint saved:", checkpoint_path)

    test_paths = sorted(
        p for p in TEST_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    test_paths = sorted(
        p for p in TEST_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if not test_paths:
        raise RuntimeError("No test images found.")

    print(f"Encoding {len(test_paths)} test images...")

    test_vectors = []

    for start in range(0, len(test_paths), BATCH_SIZE):
        paths = test_paths[start:start + BATCH_SIZE]
        images = [load_image(path) for path in paths]
        test_vectors.append(encode_images(images))

    test_vectors = torch.cat(test_vectors, dim=0)

    image_scores = test_vectors @ image_prototypes.T
    text_scores = test_vectors @ text_prototypes.T

    fused_scores = (
        (1.0 - TEXT_WEIGHT) * image_scores
        + TEXT_WEIGHT * text_scores
    )

    predictions = fused_scores.argmax(dim=1).tolist()

    output = RESULTS_DIR / "siglip_fusion_predictions.csv"

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "image_path",
            "predicted_class",
            "image_similarity",
            "text_similarity",
            "fused_score"
        ])

        for i, (path, pred) in enumerate(
            zip(test_paths, predictions)
        ):
            writer.writerow([
                str(path.relative_to(ROOT)),
                class_names[pred],
                float(image_scores[i, pred]),
                float(text_scores[i, pred]),
                float(fused_scores[i, pred])
            ])

    print("Predictions saved:", output)

    if all(path.parent.name in class_names for path in test_paths):
        correct = sum(
            path.parent.name == class_names[pred]
            for path, pred in zip(test_paths, predictions)
        )

        accuracy = correct / len(test_paths)

        print(f"Test accuracy: {accuracy:.2%}")
        print(f"Correct: {correct}/{len(test_paths)}")

        metrics = {
            "experiment": "siglip_image_text_fusion",
            "model": MODEL_NAME,
            "labeled_training_images": 16,
            "augmentation_views": 5,
            "text_weight": TEXT_WEIGHT,
            "test_images": len(test_paths),
            "correct": correct,
            "accuracy": accuracy
        }

        metrics_path = RESULTS_DIR / "siglip_fusion_metrics.json"

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print("Metrics saved:", metrics_path)


if __name__ == "__main__":
    main()
