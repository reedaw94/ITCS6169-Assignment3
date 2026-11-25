
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "experiment6"
OUTPUT.mkdir(parents=True, exist_ok=True)

FUSION_FILE = RESULTS / "siglip_fusion_predictions.csv"
TTA_FILE = RESULTS / "siglip_fusion_tta_predictions.csv"

CLASSES = [
    "Bedroom",
    "Coast",
    "Flower",
    "Forest",
    "Highway",
    "Industrial",
    "InsideCity",
    "Kitchen",
    "LivingRoom",
    "Mountain",
    "Office",
    "OpenCountry",
    "Store",
    "Street",
    "Suburb",
    "TallBuilding"
]


def load_predictions(path, prediction_name):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {path}"
        )

    df = pd.read_csv(path)

    required = {"image_path", "predicted_class"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing required columns in {path.name}"
        )

    df = df[["image_path", "predicted_class"]].copy()

    df["image_path"] = (
        df["image_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )

    if df["image_path"].duplicated().any():
        raise ValueError(
            f"Duplicate image paths found in {path.name}"
        )

    df = df.rename(
        columns={"predicted_class": prediction_name}
    )

    return df


print("=" * 60)
print("EXPERIMENT 6: ERROR ANALYSIS")
print("=" * 60)

fusion = load_predictions(
    FUSION_FILE,
    "fusion_prediction"
)

tta = load_predictions(
    TTA_FILE,
    "tta_prediction"
)

# Merge on image path, not row position.
df = fusion.merge(
    tta,
    on="image_path",
    how="outer",
    validate="one_to_one",
    indicator=True
)

if not (df["_merge"] == "both").all():
    raise ValueError(
        "Fusion and TTA prediction files contain "
        "different image sets."
    )

df = df.drop(columns=["_merge"])

# Ground-truth labels are stored in the parent
# folder of each test image.
df["true_class"] = df["image_path"].apply(
    lambda p: Path(p).parent.name
)

for column in [
    "true_class",
    "fusion_prediction",
    "tta_prediction"
]:
    invalid = set(df[column]) - set(CLASSES)

    if invalid:
        raise ValueError(
            f"Unexpected classes in {column}: {invalid}"
        )

df["fusion_correct"] = (
    df["true_class"] == df["fusion_prediction"]
)

df["tta_correct"] = (
    df["true_class"] == df["tta_prediction"]
)

df = df.sort_values("image_path").reset_index(drop=True)

total = len(df)
fusion_correct = int(df["fusion_correct"].sum())
tta_correct = int(df["tta_correct"].sum())

print(f"\nTotal test images: {total}")
print(
    f"Fusion accuracy: "
    f"{fusion_correct / total:.2%} "
    f"({fusion_correct}/{total})"
)
print(
    f"TTA accuracy: "
    f"{tta_correct / total:.2%} "
    f"({tta_correct}/{total})"
)

# ==================================================
# 1. CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    df["true_class"],
    df["fusion_prediction"],
    labels=CLASSES
)

cm_df = pd.DataFrame(
    cm,
    index=CLASSES,
    columns=CLASSES
)

cm_df.to_csv(
    OUTPUT / "fusion_confusion_matrix.csv"
)

fig, ax = plt.subplots(figsize=(13, 11))

im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(range(len(CLASSES)))
ax.set_yticks(range(len(CLASSES)))

ax.set_xticklabels(CLASSES, rotation=90)
ax.set_yticklabels(CLASSES)

ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title(
    "SigLIP Image-Text Fusion: Confusion Matrix"
)

for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        if cm[i, j] > 0:
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if cm[i, j] > cm.max() / 2
                    else "black"
                )
            )

fig.colorbar(im, ax=ax, label="Image count")
fig.tight_layout()

fig.savefig(
    OUTPUT / "fusion_confusion_matrix.png",
    dpi=250
)

plt.close(fig)

# ==================================================
# 2. PER-CLASS ACCURACY
# ==================================================

per_class = (
    df.groupby("true_class")
    .agg(
        total=("fusion_correct", "size"),
        correct=("fusion_correct", "sum")
    )
    .reindex(CLASSES)
)

per_class["accuracy"] = (
    per_class["correct"] / per_class["total"]
)

per_class.to_csv(
    OUTPUT / "fusion_per_class_accuracy.csv"
)

fig, ax = plt.subplots(figsize=(11, 8))

sorted_classes = per_class.sort_values(
    "accuracy",
    ascending=True
)

ax.barh(
    sorted_classes.index,
    sorted_classes["accuracy"] * 100
)

ax.set_xlim(0, 100)
ax.set_xlabel("Accuracy (%)")
ax.set_title(
    "SigLIP Image-Text Fusion: Per-Class Accuracy"
)

for i, value in enumerate(
    sorted_classes["accuracy"] * 100
):
    ax.text(
        min(value + 1, 95),
        i,
        f"{value:.1f}%",
        va="center",
        fontsize=9
    )

fig.tight_layout()

fig.savefig(
    OUTPUT / "fusion_per_class_accuracy.png",
    dpi=250
)

plt.close(fig)

# ==================================================
# 3. MOST COMMON CONFUSIONS
# ==================================================

errors = df.loc[
    ~df["fusion_correct"]
].copy()

errors.to_csv(
    OUTPUT / "fusion_misclassified_images.csv",
    index=False
)

confusions = (
    errors.groupby(
        ["true_class", "fusion_prediction"]
    )
    .size()
    .reset_index(name="count")
    .sort_values(
        ["count", "true_class"],
        ascending=[False, True]
    )
)

confusions.to_csv(
    OUTPUT / "fusion_top_confusions.csv",
    index=False
)

# ==================================================
# 4. COMPARE FUSION VS TTA
# ==================================================

conditions = [
    df["fusion_correct"] & df["tta_correct"],
    df["fusion_correct"] & ~df["tta_correct"],
    ~df["fusion_correct"] & df["tta_correct"],
    ~df["fusion_correct"] & ~df["tta_correct"]
]

choices = [
    "Both correct",
    "TTA introduced error",
    "TTA corrected error",
    "Both incorrect"
]

df["comparison"] = np.select(
    conditions,
    choices,
    default="Unknown"
)

df.to_csv(
    OUTPUT / "fusion_vs_tta_comparison.csv",
    index=False
)

comparison_counts = (
    df["comparison"]
    .value_counts()
    .reindex(choices, fill_value=0)
)

comparison_counts.to_csv(
    OUTPUT / "fusion_vs_tta_summary.csv",
    header=["count"]
)

# ==================================================
# 5. CLASSIFICATION REPORT
# ==================================================

report = classification_report(
    df["true_class"],
    df["fusion_prediction"],
    labels=CLASSES,
    output_dict=True,
    zero_division=0
)

with open(
    OUTPUT / "fusion_classification_report.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(report, f, indent=2)

# ==================================================
# 6. SUMMARY
# ==================================================

summary = {
    "experiment": "experiment6_error_analysis",
    "test_images": total,
    "fusion_correct": fusion_correct,
    "fusion_accuracy": fusion_correct / total,
    "tta_correct": tta_correct,
    "tta_accuracy": tta_correct / total,
    "fusion_errors": total - fusion_correct,
    "tta_errors": total - tta_correct,
    "comparison": comparison_counts.to_dict()
}

with open(
    OUTPUT / "summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)

for class_name, row in per_class.iterrows():
    print(
        f"{class_name:15s} "
        f"{int(row['correct']):2d}/"
        f"{int(row['total']):2d} "
        f"({row['accuracy']:.1%})"
    )

print("\n" + "=" * 60)
print("TOP 10 CLASS CONFUSIONS")
print("=" * 60)

print(
    confusions.head(10).to_string(index=False)
)

print("\n" + "=" * 60)
print("FUSION VS TTA")
print("=" * 60)

print(comparison_counts.to_string())

print("\nFiles saved to:")
print(OUTPUT)

print("\nExperiment 6 complete.")
