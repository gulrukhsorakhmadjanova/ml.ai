SST-2 Sentiment Analysis with BERT

This project fine-tunes a BERT (bert-base-uncased) model on the SST-2 dataset for binary sentiment classification (positive vs negative). Training is done via train.py, and evaluation is performed entirely in Google Colab (task1_uzcosmos.ipynb).

All outputs (checkpoints, metrics, logs) are saved inside Google Colab.

Installation

Install required Python packages:

pip install --upgrade pip
pip install transformers datasets evaluate scikit-learn sentencepiece wandb tensorboard
pip install -q accelerate  # optional for faster training


Or install from requirements.txt:

pip install -r requirements.txt


requirements.txt content:

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

Training with train.py

The script supports:

Hugging Face Trainer API (default)

Optional custom PyTorch loop (manual training)

Standard training:
python train.py --output_dir ./sst2_outputs --epochs 3 \
--per_device_train_batch_size 16 --per_device_eval_batch_size 64 --lr 2e-5


⚠️ On Colab Free GPU, reduce per_device_train_batch_size to 8 or 4 to avoid out-of-memory errors.

Optional custom training loop:
python train.py --use_custom_loop


Note: Using --use_custom_loop restarts training from scratch.

All checkpoints are saved to sst2_outputs/ in Google Colab.

Evaluation in Google Colab

Evaluation is handled entirely in Google Colab (task1_uzcosmos.ipynb):

Automatically loads the latest checkpoint from sst2_outputs/

Computes Accuracy and F1-score

Generates a classification report and confusion matrix

Saves metrics to results/eval_metrics.npz

No evaluation scripts are included in GitHub; all evaluation outputs live in Colab.

Features

Automatic detection of latest model checkpoint

Accuracy & F1-score evaluation

Classification report & confusion matrix visualization

Optional custom training loop

Metrics saved for further analysis

Notes

All model outputs and evaluation metrics are saved inside Google Colab.

To reproduce results locally, download outputs from Colab.

GitHub repo contains training scripts and dependencies only.

References

BERT: https://huggingface.co/bert-base-uncased

SST-2 Dataset (GLUE): https://huggingface.co/datasets/glue

Hugging Face Transformers: https://huggingface.co/docs/transformers

Google Colab: https://colab.research.google.com
