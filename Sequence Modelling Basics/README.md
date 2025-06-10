#Sequence Modeling for Sentiment Classification using RNNs and LSTMs

This repository contains my work for the NLP Track (Sequence Modeling) under the COPS Summer of Code 2025, Intelligence Guild, IIT (BHU) Varanasi. The goal of the project is to build and compare sequence-based deep learning models for binary sentiment classification using the Amazon Reviews dataset.

---

Project Structure

├── code/
│ ├── data/ # (Kept empty due to size; locally contains cleaned_sampled_train.csv & test.csv)
│ ├── word2idx.pkl # Vocabulary mapping (20k most frequent tokens)
│ ├── 1_data_sampling.ipynb # Data cleaning + balanced stratified sampling
│ ├── 2_dataloader_and_embeddings.ipynb # Tokenization, padding, vocab creation, tensor conversion
│ ├── 3_test_prep.ipynb # Preprocessing test data identically
│ ├── 4_simple_rnn.ipynb # Vanilla RNN model
│ ├── 5_Istm.ipynb # Standard LSTM model
│ ├── 6_bidirectional_Istm.ipynb # Bidirectional LSTM model
│ ├── 7_glove_freeze.ipynb # LSTM with frozen GloVe embeddings
│ ├── 8_glove_unfreeze.ipynb # LSTM with trainable GloVe embeddings
│
├── report/
│ ├── report.pdf # Final project report (with plots, metrics, analysis)
│ └── report.tex # LaTeX source of the report


---

Models Implemented

| Model                 | Description                                   |
|----------------------|-----------------------------------------------|
| Vanilla RNN          | Basic RNN with tanh activation                |
| LSTM                 | Single-layer LSTM with randomly initialized embeddings |
| BiLSTM               | Bidirectional LSTM                            |
| GloVe (Frozen)       | LSTM using pre-trained GloVe embeddings (frozen) |
| GloVe (Unfrozen)     | LSTM with trainable GloVe embeddings          |

Each model was evaluated on training, validation, and test sets using:

- Accuracy
- Precision / Recall / F1 Score
- ROC AUC
- PR AUC
- Confusion Matrix

Plots include loss/accuracy vs epoch, ROC and PR curves, and gradient behavior (for RNN).

---
Dataset

- Amazon Reviews dataset: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/data
- Cleaned and balanced samples were used:  
  - `cleaned_sampled_train.csv` (800k samples)
  - `cleaned_sampled_test.csv` (400K samples)

> Due to GitHub size limits, the actual datasets are not included here. Please download from [Amazon Reviews Kaggle page](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products) and follow `1_data_sampling.ipynb` to reproduce.

---

How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sequence-modeling-sentiment.git
   cd sequence-modeling-sentiment/code
2. Create a virtual environment and install dependencies:
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r ../requirements.txt
3. Run the notebooks in order
4. For evaluation, use the metrics printed inside each notebook and visualizations in the report.
5.  Report
The full analysis, figures, and model comparison is available in report/report.pdf.
It includes:

Training curves

ROC & PR curves

Confusion matrices

Gradient norms (for RNN)

Comparative metric table

Reflections on evaluation practices and model behavior

---
Requirements
See requirements.txt

Author
Abhishek Kumar Chaubey
Roll Number: 24144001
COPS Summer of Code 2025 – Intelligence Guild (NLP Track)
