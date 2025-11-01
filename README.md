# 🧠 Fine-Tuning BERT on Sentiment Analysis (SST-2)

> **Author:** Gulrukhsor Akhmadjanova
> **Task:** Machine Learning Engineer Assignment — Fine-Tuning BERT (`bert-base-uncased`)
> **Goal:** Predict sentiment (positive / negative) on the SST-2 dataset using Hugging Face Transformers.

---

## 📌 Overview

This project fine-tunes a **pretrained BERT model** on the **Stanford Sentiment Treebank (SST-2)** dataset — a benchmark for sentiment classification.
The model learns to classify sentences into **positive** or **negative** sentiment categories.
All training and evaluation steps follow modern NLP standards using the **🤗 Transformers**, **Datasets**, and **Evaluate** libraries.

---

## 🗂️ Project Structure

```
train.py                 # Main training script (GitHub)
requirements.txt         # Python dependencies
README.md                # Project documentation
```

> **Note:** Evaluation is performed entirely in **Google Colab** (`task1_uzcosmos.ipynb`). All outputs (checkpoints, metrics, logs) are saved in Colab.

---

## 📊 Dataset: SST-2

| Property            | Description                                                                             |
| ------------------- | --------------------------------------------------------------------------------------- |
| **Dataset Name**    | GLUE — SST-2 (Stanford Sentiment Treebank)                                              |
| **Task Type**       | Binary Sentiment Classification                                                         |
| **Labels**          | `0` → Negative, `1` → Positive                                                          |
| **Train Size**      | ~67,000 samples                                                                         |
| **Validation Size** | ~1,800 samples                                                                          |
| **Source**          | [🤗 Hugging Face Datasets: glue/sst2](https://huggingface.co/datasets/glue/viewer/sst2) |

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your_username>/bert-sst2.git
cd bert-sst2
```

### 2️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install transformers datasets evaluate scikit-learn sentencepiece wandb tensorboard
pip install -q accelerate  # optional for faster training
```

Or install all packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**requirements.txt content:**

```
transformers
datasets
evaluate
scikit-learn
sentencepiece
wandb
tensorboard
accelerate
torch
numpy
matplotlib
tqdm
```

---

## 🏋️ Training

The training script (`train.py`) supports:

* Hugging Face **Trainer API** (default)
* Optional **custom PyTorch training loop** (manual optimization)

### Standard Training Command

```bash
python train.py --output_dir ./sst2_outputs --epochs 3 \
--per_device_train_batch_size 16 --per_device_eval_batch_size 64 --lr 2e-5
```

> ⚠️ On Colab Free GPU, reduce `per_device_train_batch_size` to 8 or 4 if you run out of memory.

### Optional Custom Loop

```bash
python train.py --use_custom_loop
```

> Note: Using `--use_custom_loop` **restarts training** from scratch.

All checkpoints are saved in `sst2_outputs/` in Google Colab.

---

## 📊 Evaluation in Google Colab

Evaluation runs **entirely in Colab** via `task1_uzcosmos.ipynb` and includes:

* Loading the latest checkpoint from `sst2_outputs/`
* Computing **Accuracy** and **F1-score**
* Generating **classification report** and **confusion matrix**
* Saving metrics to `results/eval_metrics.npz`

> No evaluation scripts are included in GitHub; all evaluation outputs live in Colab.

---

## 📈 Features

* Automatic detection of latest model checkpoint
* Accuracy & F1-score evaluation
* Classification report & confusion matrix visualization
* Optional custom training loop
* Metrics saved for further analysis

---

## 📌 Notes

* All model outputs and evaluation metrics are saved **inside Google Colab**.
* To reproduce results locally, download the outputs from Colab.
* GitHub repo contains **training scripts and dependencies only**.

---

## 📖 References

* **BERT**: [https://huggingface.co/bert-base-uncased](https://huggingface.co/bert-base-uncased)
* **SST-2 Dataset (GLUE)**: [https://huggingface.co/datasets/glue](https://huggingface.co/datasets/glue)

# Road Detection from Aerial Images 🛣️

> **Author:** Gulrukhsor Akhmadjanova
> **Task:** Machine Learning Engineer Assignment — Binary Semantic Segmentation
> **Objective:** Detect roads in high-resolution aerial images using deep learning.

---

## Project Overview

This repository contains a full implementation of a **binary semantic segmentation pipeline** to detect roads in aerial/satellite images.
The model predicts a per-pixel mask where:

* `1` → Road
* `0` → Background

The project demonstrates **data preprocessing, model building, training, evaluation, and visualization**, using a U-Net architecture with a pre-trained backbone.

---

## Dataset

You can use either of the following public datasets:

1. **Massachusetts Roads Dataset** – Aerial imagery of Massachusetts with road masks.
2. **DeepGlobe Road Extraction Dataset** – Satellite imagery with corresponding road masks.

**Dataset Requirements:**

* Images and masks should have the same resolution.
* Masks should be binary (0 = background, 1 = road).
* Dataset should be structured as:

```
dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

---

## Features

* **Preprocessing**:

  * Resizing and normalization
  * Data augmentation (flip, rotation, brightness, noise)

* **Model**:

  * U-Net with pre-trained backbone (e.g., ResNet34)
  * Configurable hyperparameters

* **Loss & Metrics**:

  * Binary Cross Entropy + Dice Loss
  * IoU (Intersection over Union)
  * Dice Coefficient

* **Training**:

  * Configurable epochs, batch size, learning rate
  * Validation tracking and model checkpointing

* **Evaluation**:

  * Test set evaluation with metrics
  * Visualizations: input image, ground truth mask, predicted mask

* **Code Quality**:

  * Modular and reusable
  * Configurable via dictionary (`CONFIG`)
  * Reproducible with random seed

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/road-detection.git
cd road-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> `requirements.txt` includes:
>
> ```
> ```

torch
torchvision
segmentation-models-pytorch
albumentations
opencv-python
matplotlib
numpy
tqdm
shapely
kaggle

````

---

## Configuration

All hyperparameters and paths are configurable in the `CONFIG` dictionary:

```python
CONFIG = {
    "seed": 42,
    "image_dir": "data/train/images",
    "mask_dir": "data/train/masks",
    "img_size": 256,
    "batch_size": 8,
    "epochs": 5,
    "learning_rate": 1e-4,
    "val_split": 0.15,
    "test_split": 0.15,
    "encoder_name": "resnet34",
    "encoder_weights": "imagenet",
    "device": "cuda",
    "model_save_path": "best_model.pth",
}
````

---

## Usage

1. **Prepare Dataset**
   Place images and masks into `train`, `val`, and `test` folders as shown above.

2. **Training**

```python
from train import train_loop, model, train_loader, val_loader

history = train_loop(
    model, train_loader, val_loader,
    epochs=CONFIG["epochs"], lr=CONFIG["learning_rate"],
    save_path=CONFIG["model_save_path"]
)
```

3. **Evaluation**

```python
from evaluate import evaluate_model, visualize_predictions

evaluate_model(model, test_loader)
visualize_predictions(model, test_loader, n=5)
```

4. **Visualize Results**

* Input image
* Ground truth mask
* Predicted mask

---

## Metrics

* **Loss:** BCE + Dice Loss
* **IoU (Intersection over Union)**
* **Dice Coefficient**

Metrics are logged for **training**, **validation**, and **test sets**.

---

## Reproducibility

* Random seeds are fixed for deterministic behavior.
* Modular, readable code for easy modification.
* All hyperparameters are configurable via `CONFIG`.

---

## Optional Bonuses

* Post-processing: morphological operations on predicted masks
* Vectorization: convert masks to polygons for GIS applications
* Tile-based training for high-resolution images

---

## License

This project is for educational and portfolio purposes.
Do not use the datasets for commercial purposes without proper permissions.





