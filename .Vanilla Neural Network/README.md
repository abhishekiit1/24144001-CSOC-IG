Medical Appointment No-Show Prediction
This repository contains code and reports for a project aimed at predicting whether patients will show up for medical appointments using machine learning models.

Project Overview
The dataset used is the Medical Appointment No-Show dataset (Kaggle V2 - May 2016), which contains over 100,000 medical appointments with various patient details and whether they showed up or not.

Two main implementations are provided:

Pure Python Neural Network for binary classification.
PyTorch Neural Network for binary classification.
Both models handle class imbalance using weighted loss functions without oversampling or data augmentation.

Files Description
KaggleV2-May-2016.csv - Original dataset file.
X_train.npy, X_test.npy, y_train.npy, y_test.npy - Preprocessed data splits saved as NumPy arrays.
pure_py_implementation.ipynb - Jupyter notebook for pure Python neural network implementation.
pytorch_implementation.ipynb - Jupyter notebook for PyTorch neural network implementation.
pure_py_metrics.png, pure_python_cm.png - Performance plots and confusion matrix for pure Python model.
pytorch_metrics.png, pytorch_cm.png - Performance plots and confusion matrix for PyTorch model.
csoc_week1_report.tex, csoc_week1_report.pdf - LaTeX report source and compiled PDF.
README.md - This file.
Setup Instructions
Clone this repository.

Create a Python environment (recommended):

python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
run -> pip install -r requirements.txt
