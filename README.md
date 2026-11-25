# One-Shot Scene Classification Using SigLIP and Augmented Image-Text Prototypes

## Project Overview

- Course: ITCS 6169
- Assignment: Assignment 3 - One-Shot Image Classification
- University: University of North Carolina at Charlotte
- Author: Andrew Reed

This project investigates one-shot scene classification using pretrained
vision-language foundation models. The objective is to classify images
across 16 scene categories using only one labeled training image per class.

The best-performing approach combines augmented SigLIP visual prototypes
with semantic text embeddings, achieving 83.00% classification accuracy
on the provided 400-image test dataset.

## Dataset

- Available training images: 2,400
- Scene categories: 16
- Labeled training images used: 16
- Labeled images per category: 1
- Test images: 400
- Test images per category: 25
- Reference selection seed: 42

The fixed one-shot manifest is stored in:

manifests/one_shot_manifest.csv

The original datasets are not distributed through this repository.

## Methodology

The proposed classification pipeline uses the pretrained
google/siglip-base-patch16-224 foundation model.

All encoder parameters remain frozen.

### Augmented Visual Prototypes

Five deterministic views are generated from each labeled reference image:

1. Original image
2. Horizontal reflection
3. Brightness enhancement of 10%
4. Contrast enhancement of 10%
5. Center crop retaining approximately 90% of each dimension

The normalized image embeddings are averaged and renormalized to
construct a visual prototype for each class.

### Semantic Text Prototypes

Each class is represented by a natural-language description.
The descriptions are encoded using the SigLIP text encoder.

### Image-Text Fusion

The final classification score combines visual and semantic similarity:

    Final score = 0.75 * image similarity
                + 0.25 * text similarity

The class with the highest combined similarity is selected.

No additional labeled training images or gradient-based fine-tuning
are used.

## Experimental Results

| Experiment | Correct | Accuracy |
|---|---:|---:|
| CLIP visual prototypes | 306/400 | 76.50% |
| SigLIP visual prototypes | 319/400 | 79.75% |
| SigLIP augmented prototypes | 324/400 | 81.00% |
| SigLIP image-text fusion | 332/400 | **83.00%** |
| SigLIP fusion with test-time augmentation | 329/400 | 82.25% |

The proposed method improves upon the CLIP baseline by 6.50
percentage points.

## Error Analysis

The best-performing classifier produced 68 classification errors.

| Class | Accuracy |
|---|---:|
| OpenCountry | 8% |
| Bedroom | 40% |
| InsideCity | 56% |
| Industrial | 68% |

OpenCountry, Bedroom, and InsideCity accounted for 49 of the
68 classification errors.

Six categories achieved 100% accuracy:

- Coast
- Flower
- Highway
- Kitchen
- Office
- Suburb

Test-time augmentation corrected one error but introduced four
additional errors.

Detailed analysis is available in results/experiment6/.

## Repository Structure

    ITCS6169-Assignment3/
    |-- manifests/
    |   `-- one_shot_manifest.csv
    |-- scripts/
    |   |-- prepare_one_shot_dataset.py
    |   |-- clip_one_shot.py
    |   |-- siglip_one_shot.py
    |   |-- siglip_augmented.py
    |   |-- siglip_fusion.py
    |   |-- siglip_fusion_tta.py
    |   `-- experiment6_error_analysis.py
    |-- results/
    |-- models/
    |-- requirements.txt
    |-- README.md
    `-- .gitignore

## Experimental Environment

- Operating system: Windows
- Python: 3.14
- Framework: PyTorch
- GPU: NVIDIA GeForce GTX 1080 (8 GB)
- Acceleration: CUDA
- Foundation models: CLIP and SigLIP

## Installation

Clone the repository:

    git clone https://github.com/reedaw94/ITCS6169-Assignment3.git
    cd ITCS6169-Assignment3

Install the required dependencies:

    py -3.14 -m pip install -r requirements.txt

A compatible CUDA-enabled PyTorch installation is recommended.
CPU inference is also supported by the classification scripts.

## Dataset Preparation

Place the provided dataset in the following directories:

    train/
    test/
    test2/

Training and labeled test images should be organized into their
respective class folders.

The original one-shot manifest is included to preserve the exact
reference images used in the experiments.

To regenerate the manifest using seed 42:

    py -3.14 scripts/prepare_one_shot_dataset.py

The supplied manifest should be used when reproducing the reported
experimental results.

## Reproducing the Best Result

Run:

    py -3.14 scripts/siglip_fusion.py

Expected result using the original dataset and environment:

    Test accuracy: 83.00%
    Correct: 332/400

The pretrained SigLIP model is downloaded automatically through
Hugging Face if it is not already cached.

## Reproducing All Experiments

    py -3.14 scripts/clip_one_shot.py
    py -3.14 scripts/siglip_one_shot.py
    py -3.14 scripts/siglip_augmented.py
    py -3.14 scripts/siglip_fusion.py
    py -3.14 scripts/siglip_fusion_tta.py
    py -3.14 scripts/experiment6_error_analysis.py

## Evaluation Limitations

Multiple experimental configurations were evaluated on the same
labeled test dataset. The reported results represent exploratory
experimental comparisons rather than an independent evaluation
on a completely untouched test set.

The SigLIP implementation also reported special-token configuration
warnings. Text embeddings were checked for finite values and expected
dimensions, and inference completed successfully. These checks do
not independently establish tokenizer configuration correctness.

## References

1. Radford et al. (2021). Learning Transferable Visual Models
   From Natural Language Supervision.
   https://arxiv.org/abs/2103.00020

2. Zhai et al. (2023). Sigmoid Loss for Language Image Pre-Training.
   https://arxiv.org/abs/2303.15343

3. CLIP pretrained model:
   https://huggingface.co/openai/clip-vit-base-patch32

4. SigLIP pretrained model:
   https://huggingface.co/google/siglip-base-patch16-224


## Standalone Checkpoint-Based Inference

The saved one-shot classifier is located at:

    models/siglip_fusion_prototypes.pt

The checkpoint contains the visual prototypes, semantic text
prototypes, class names, and fusion parameters.

To classify the labeled test dataset:

    py -3.14 scripts/inference.py --input test

To classify an unlabeled image directory:

    py -3.14 scripts/inference.py --input test2

To classify an individual image:

    py -3.14 scripts/inference.py --input example.jpg

The original training images are not required for checkpoint-based
inference. The pretrained SigLIP encoder is downloaded automatically
if it is not already cached.

