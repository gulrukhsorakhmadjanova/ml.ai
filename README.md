# TASK 1:  Fine-Tuning BERT on Sentiment Analysis (SST-2)

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

# TASK 2: 🛰️ Road Detection from Aerial Images

This project focuses on **binary semantic segmentation** to detect roads from high-resolution aerial or satellite images.  
The goal is to identify pixels belonging to roads (`1`) and background (`0`) using deep learning.

---

## 🎯 Objective
Build and train a model capable of automatically segmenting roads from aerial imagery using architectures like **U-Net**.  
This project demonstrates understanding of **image segmentation**, **data preprocessing**, **model development**, **evaluation**, and **post-processing**.

---

## 📚 Dataset
**Used:** Synthetic/Fake dataset generated within the Colab notebook for demonstration.  
**Not yet implemented:** Real datasets such as **Massachusetts Roads** or **DeepGlobe Road Extraction**.

Each generated image has a corresponding ground truth mask of the same resolution.

---

## ⚙️ Project Structure
- Implemented in **Google Colab**
- GitHub repository serves as a **reference link** to Colab code
- All experiments, visualization, and training are done in Colab

---

## ✅ Completed Requirements

### 🧩 Data Preprocessing
- [x] Dataset loading (synthetic dataset created in code)
- [x] Data inspection (visualized random images and masks)
- [x] Resizing (handled during dataset creation)
- [x] Normalization (image tensors normalized to `[0, 1]`)
- [x] Dataset class implemented (`RoadDataset`)
- [ ] Real dataset loading (Massachusetts/DeepGlobe)
- [ ] Advanced augmentation (only basic random noise/lines used)

---

### 🧠 Model Architecture
- [x] Implemented **U-Net** architecture from scratch
- [x] Supports modular layers for encoder–decoder
- [ ] No pretrained backbone (e.g., ResNet/EfficientNet) yet

---

### 🏋️‍♀️ Training
- [x] Training loop implemented
- [x] Loss function: Binary Cross Entropy (BCE)
- [x] IoU metric implemented
- [x] Dice metric implemented
- [x] Tracks loss and IoU over epochs
- [ ] Train/validation split (currently uses full dataset)
- [ ] Combined losses (Dice + BCE not yet used)

---

### 📊 Evaluation
- [x] Test set evaluation (IoU and Dice)
- [x] Visualization: input, ground truth, prediction (side-by-side)
- [x] Plots: training loss and IoU curves
- [ ] Evaluation on real dataset

---

### 🧱 Code Quality & Reproducibility
- [x] Clear, modular code (dataset class, model class, metrics, etc.)
- [x] Configurable hyperparameters (batch size, epochs, learning rate)
- [x] Requirements saved to `requirements.txt`
- [x] Model saved as `road_detection_model.pth`
- [x] Training information saved as `.json`
- [x] Summary printed after training
- [x] All visualizations automated
- [x] Works fully in Google Colab
- [ ] External dataset configuration not yet added

---

### 🏅 Bonus (Optional)
- [x] Post-processing with morphological operations (OpenCV)
- [x] Mask-to-vector polygon conversion (Shapely)
- [x] Overlay predictions on image
- [x] Pretrained model option (U-Net with ResNet34 backbone)
- [ ] Tile-based training for large images not implemented

---

## 🧾 Summary

| Category | Description | Status |
|-----------|-------------|--------|
| **Data Preprocessing** | Synthetic data + normalization | ✅ Partial |
| **Model** | U-Net (custom) | ✅ Done |
| **Training** | BCE + IoU + Dice metrics | ✅ Partial |
| **Evaluation** | Visualization & plots | ✅ Done |
| **Post-processing** | Morphology + Polygon extraction | ✅ Done |
| **Dataset** | Real dataset integration | ❌ Not yet |
| **Tile-based training** | Not implemented | ❌ Not yet |

---

## 💻 Technologies Used
- Python 3.x  
- PyTorch  
- OpenCV  
- NumPy, Matplotlib  
- Shapely  
- tqdm  

---

## 🚀 How to Run
1. Open the Colab notebook link.
2. Run all cells in order.
3. Model will train, evaluate, and visualize predictions.
4. Outputs:
   - Model weights (`road_detection_model.pth`)
   - Metrics & training logs (`training_info.json`)
   - Plots & predicted masks

---

## 📈 Future Improvements
- Integrate real aerial datasets (Massachusetts/DeepGlobe)
- Add advanced augmentations (flip, rotate, crop)
- Implement hybrid losses (BCE + Dice)
- Add validation split and early stopping
- Support tile-based training for large images

---

## 👩‍💻 Author
**Gulrukhsor Akhmadjanova**  
Google Colab Implementation, 2025  
