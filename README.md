# Alzheimers MobileNet

Approximate project date: June 2023  
Original environment: Deepnote  
Repository organized/pushed: 2026  
Status: archived and repaired for local/Deepnote reruns

This project trains image classifiers on an Alzheimer MRI image dataset with four labels:

- MildDemented
- ModerateDemented
- NonDemented
- VeryMildDemented

The notebook is the historical Deepnote work from June 2023. In Deepnote, the later working notebook became the default project version, so this repository keeps that version as the single main notebook. The supporting script keeps the same project idea but makes the workflow easier to rerun locally or from Deepnote by removing hardcoded `/work` paths, keeping the large dataset out of Git, and fixing the MobileNetV2 input-channel issue.

## Repository layout

```text
.
├── model.ipynb
├── README.md
├── requirements.txt
└── src/
    └── train_alzheimers.py
```

- `model.ipynb`: main Deepnote notebook, restored from the later working project version.
- `src/train_alzheimers.py`: reusable training script for the CNN and MobileNetV2 workflows.
- `requirements.txt`: Python dependencies needed to run the repaired script.

## Dataset

Place the dataset at:

```text
dataset/OriginalDataset/
  MildDemented/
  ModerateDemented/
  NonDemented/
  VeryMildDemented/
```

The dataset is intentionally ignored by Git because it is large.

If you are running this from Deepnote and the dataset is still mounted at the original project path, use:

```bash
python src/train_alzheimers.py --dataset-dir /work/dataset/OriginalDataset
```

If you are running locally, put the dataset under this repo as `dataset/OriginalDataset`.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the small CNN:

```bash
python src/train_alzheimers.py --model cnn --epochs 10
```

Train the MobileNetV2 transfer-learning model:

```bash
python src/train_alzheimers.py --model mobilenet --epochs 10
```

Use a custom dataset path:

```bash
python src/train_alzheimers.py --dataset-dir /work/dataset/OriginalDataset --model cnn
```

## Models

The script supports two model options:

- `cnn`: a small baseline convolutional neural network trained on grayscale `128x128` images.
- `mobilenet`: a MobileNetV2 transfer-learning model trained on RGB `224x224` images.

The MobileNetV2 path uses ImageNet weights and freezes the base model before training the classification head.

## Outputs

After training, the script prints:

- Keras model summary
- validation loss
- validation accuracy
- confusion matrix
- classification report

Model files are not saved automatically yet. If you want to keep a trained model, add a `model.save(...)` call or redirect saved models into `models/`, which is already ignored by Git.

## Notes

The original Deepnote notebook used `/work/dataset/OriginalDataset`. The cleaned script defaults to the repo-local `dataset/OriginalDataset` but still accepts the Deepnote path through `--dataset-dir`.

MobileNetV2 requires RGB images with three channels. The original notebook used grayscale arrays shaped like `(128, 128, 1)`, which caused the transfer-learning model to fail. The repaired MobileNetV2 path loads RGB images and uses the matching preprocessing function.

This repository was organized and pushed in 2026 for archival purposes, while the original project work dates back to June 2023.
