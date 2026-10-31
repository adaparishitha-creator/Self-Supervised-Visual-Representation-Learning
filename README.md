# Self-Supervised Visual Representation Learning

**ITCS 6169 – Computer Vision | Assignment 2**

## 1. Project Overview

This repository contains the training code, evaluation code, model architecture, configuration, experiment results, and trained checkpoint for Assignment 2: Self-Supervised Visual Representation Learning.

Five experiments were conducted in Google Colab using PyTorch and GPU acceleration.

The best-performing model uses **ResNet-18 with MoCo v2 self-supervised pretraining followed by advanced supervised fine-tuning**.

**Best Model Performance:**
- Validation Accuracy: **81.67%**
- Test Accuracy: **76.00%**
- Correct Test Predictions: **304 out of 400**

The reported test accuracy can be reproduced using provided checkpoint without the retraining.

## 2. Repository Files

| File | Description |
|---|---|
| `Computer_Vision_Assignment2_Final.ipynb` | Complete Google Colab notebook containing all five experiments |
| `train.py` | Training pipeline for Experiment 5 |
| `evaluate.py` | Evaluation script for reproducing the final test accuracy |
| `models.py` | ResNet-18 classification model definition |
| `config.json` | Model configuration and expected results |
| `requirements.txt` | Python dependencies |
| `experiment_comparison.csv` | Accuracy comparison across five experiments |
| `checkpoints.txt` | Google Drive checkpoint link |

## 3. Download the Trained Model Checkpoint

The best-performing model checkpoint is available on Google Drive because its size exceeds GitHub's browser-upload limit.

**[Download Best Model Checkpoint – Google Drive](https://drive.google.com/file/d/12TTHbYcGauO4UjNm1bplJgQyf14OAVZL/view?usp=sharing)**

After downloading:

1. Create a folder named `checkpoints` in repository root.
2. Rename the downloaded file to `best_model.pth` if necessary.
3. Place it inside  `checkpoints` folder.

Required checkpoint path:

```text
checkpoints/best_model.pth
```

The checkpoint contains the trained weights of the final ResNet-18 classifier.

## 4. Reproduce the Reported Test Accuracy

Follow these steps to reproduce the **76.00% test accuracy**.

### Step 1: Clone the Repository

```bash
git clone https://github.com/adaparishitha-creator/Self-Supervised-Visual-Representation-Learning.git
cd Self-Supervised-Visual-Representation-Learning
```

Alternatively, download the repository as a ZIP file and extract it.

### Step 2: Create a Python Virtual Environment

Python 3.10 is recommended.

**Windows PowerShell:**

```powershell
py -3.10 -m venv .venv
```

Install the dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Step 3: Prepare the Dataset

Extract the assignment's `train.zip` and `test.zip` files.

Both datasets contain images organized into 16 class-specific folders.

Recommended directory structure:

```text
Assignment2/
|
|-- Self-Supervised-Visual-Representation-Learning/
|   |-- Computer_Vision_Assignment2_Final.ipynb
|   |-- train.py
|   |-- evaluate.py
|   |-- models.py
|   |-- config.json
|   |-- requirements.txt
|   |-- experiment_comparison.csv
|   |-- checkpoints/
|       |-- best_model.pth
|
|-- train/
|   |-- Bedroom/
|   |-- Coast/
|   |-- ...
|   |-- TallBuilding/
|
|-- test/
    |-- Bedroom/
    |-- Coast/
    |-- ...
    |-- TallBuilding/
```

The class names must remain consistent with those specified in `config.json`.

### Step 4: Run the Evaluation Script

From the repository root, execute:

**Windows PowerShell:**

```powershell
.\.venv\Scripts\python.exe evaluate.py --data-dir "../test"
```

**Linux/macOS:**

```bash
python evaluate.py --data-dir "../test"
```

The evaluation script:

1. Loads `config.json`.
2. Reconstructs the ResNet-18 classifier using `models.py`.
3. Loads the trained weights from `checkpoints/best_model.pth`.
4. Preprocesses the test images.
5. Performs inference on all 400 test images.
6. Calculates the final classification accuracy.

The final configuration uses 224 × 224 images and standard inference without test-time augmentation.

### Step 5: Verify the Expected Output

```text
Device: cpu
Test images: 400
Correct predictions: 304
Test accuracy: 76.00%
Expected reported accuracy: 76.00%
```

**Verification:** The evaluation was successfully executed on a local Windows CPU environment and reproduced the exact test accuracy reported in Google Colab.

A GPU is not required for evaluation using the provided checkpoint.

## 5. Reproduce the Training Pipeline

The complete training implementation for Experiment 5 is provided in `train.py`.

Training consists of two main stages:

1. **Self-supervised pretraining:** MoCo v2 learns image representations using contrastive learning.
2. **Supervised fine-tuning:** The pretrained ResNet-18 encoder is fine-tuned for 16-class scene classification.

### Step 1: Prepare the Training Dataset

Extract `train.zip` and ensure the training images are organized into 16 class folders.

### Step 2: View Training Arguments

**Windows:**

```powershell
.\.venv\Scripts\python.exe train.py --help
```

### Step 3: Start Training

**Windows:**

```powershell
.\.venv\Scripts\python.exe train.py --train-dir "../train" --output-dir "training_outputs"
```

**Linux/macOS:**

```bash
python train.py --train-dir "../train" --output-dir "training_outputs"
```

The `training_outputs` directory stores outputs from the new training run separately from the supplied checkpoint.

**Important:** Full training is computationally expensive and is recommended on a GPU-enabled environment. Retraining is not necessary to reproduce the reported test accuracy.

Independent training runs may produce slightly different results due to random initialization, data augmentation, and optimization.

## 6. Reproducing the Google Colab Experiments

All five experiments were implemented and executed using the Google Colab with GPU acceleration.

The complete experimental notebook is available in this repository:

`Computer_Vision_Assignment2_Final.ipynb`

### Steps to Run the Notebook

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Computer_Vision_Assignment2_Final.ipynb`.
3. Select **Runtime → Change runtime type → GPU**.
4. Upload the assignment's `train.zip` and `test.zip` files.
5. Run the dataset preparation and preprocessing cells.
6. Execute the experiment-specific training and evaluation sections in sequence.

### Experiments Included

**Experiment 1 – Supervised ConvNeXt-Atto**

Establishes a supervised classification baseline.

**Experiment 2 – SimCLR + Fine-Tuning**

Uses contrastive self-supervised pretraining followed by supervised classification.

**Experiment 3 – Joint SimCLR + Classification**

Combines self-supervised and supervised learning objectives.

**Experiment 4 – Optimized SSL-Based Classification**

Evaluates an optimized self-supervised classification approach.

**Experiment 5 – ResNet-18 + MoCo v2 + Advanced Fine-Tuning**

Uses MoCo v2 self-supervised pretraining and supervised fine-tuning with advanced optimization techniques.

For Experiment 5:
- MoCo v2 pretraining: 100 epochs
- Best fine-tuning epoch: 28
- Early stopping: Epoch 36
- Best validation accuracy: 81.67%
- Final test accuracy: 76.00%

Standard inference and horizontal-flip test-time augmentation were compared using validation performance. Standard inference was selected for the final evaluation.

The notebook contains the complete experimental workflow, including model training, validation, evaluation, and visualizations.

## 7. Experiment Results

| Experiment | Validation Accuracy (%) | Test Accuracy (%) |
|---|---:|---:|
| E1 – Supervised ConvNeXt-Atto | 56.04 | 54.25 |
| E2 – SimCLR + Fine-Tuning | 67.08 | 63.25 |
| E3 – Joint SimCLR + Classification | 68.33 | 68.50 |
| E4 – Optimized SSL-Based Classification | 75.00 | 71.75 |
| **E5 – ResNet-18 + MoCo v2 + Advanced Fine-Tuning** | **81.67** | **76.00** |

The results are also provided in `experiment_comparison.csv`.

Experiment 5 achieved highest validation and test accuracies among the evaluated approaches.

## 8. Reproducibility Notes

- All models were initialized without the pretrained weights.
- No external training datasets were used.
- The dataset contains 2,400 training images and 400 test images across 16 classes.
- A stratified 80/20 split was used for training and validation.
- Model selection was based on validation performance.
- The held-out test dataset was used for the final evaluation.
- The best model checkpoint is available through Google Drive.
- `config.json` contains the final model configuration.
- `evaluate.py` was verified locally and reproduced **76.00% test accuracy**.
- GPU acceleration is recommended for training but is not required for the evaluation.

---
