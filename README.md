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


---

## 📊 Dataset: SST-2

| Property | Description |
|-----------|--------------|
| **Dataset Name** | GLUE — SST-2 (Stanford Sentiment Treebank) |
| **Task Type** | Binary Sentiment Classification |
| **Labels** | `0` → Negative, `1` → Positive |
| **Train Size** | ~67,000 samples |
| **Validation Size** | ~1,800 samples |
| **Source** | [🤗 Hugging Face Datasets: glue/sst2](https://huggingface.co/datasets/glue/viewer/sst2) |

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your_username>/bert-sst2.git
cd bert-sst2

