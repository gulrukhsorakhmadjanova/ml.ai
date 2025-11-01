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
* **Hugging Face Transformers**: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
* **Google Colab**: [https://colab.research.google.com](https://colab.research.google.com)
