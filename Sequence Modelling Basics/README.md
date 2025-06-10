# Sequence Modeling for Sentiment Classification using RNNs and LSTMs

This repository contains my work for the **NLP Track (Sequence Modeling)** under the **COPS Summer of Code 2025**, Intelligence Guild, IIT (BHU) Varanasi. The objective of the project is to build and compare deep learning models based on RNN architectures for binary sentiment classification using the **Amazon Reviews dataset**.

---

## 📁 Project Structure

```
.
├── code/
│   ├── data/                         # (Kept empty due to GitHub size limits; holds train/test CSVs locally)
│   ├── word2idx.pkl                  # Vocabulary mapping (20k most frequent tokens)
│   ├── 1_data_sampling.ipynb         # Data cleaning + stratified sampling
│   ├── 2_dataloader_and_embeddings.ipynb  # Tokenization, padding, DataLoader setup
│   ├── 3_test_prep.ipynb             # Preprocess Kaggle test set
│   ├── 4_simple_rnn.ipynb            # Vanilla RNN model
│   ├── 5_Istm.ipynb                  # Basic LSTM model
│   ├── 6_bidirectional_Istm.ipynb    # BiLSTM model
│   ├── 7_glove_freeze.ipynb          # LSTM with frozen GloVe embeddings
│   ├── 8_glove_unfreeze.ipynb        # LSTM with trainable GloVe embeddings
│
├── report/
│   ├── report.pdf                    # Final project report with analysis and visualizations
│   └── report.tex                    # LaTeX source of the report
```

---

##Models Implemented

| Model              | Description                                              |
|-------------------|----------------------------------------------------------|
| **Vanilla RNN**    | Basic RNN with tanh activation                          |
| **LSTM**           | Single-layer LSTM with randomly initialized embeddings  |
| **BiLSTM**         | Bidirectional LSTM                                       |
| **GloVe (Frozen)** | LSTM with frozen pre-trained GloVe embeddings           |
| **GloVe (Unfrozen)** | LSTM with trainable GloVe embeddings                  |

Each model is evaluated on the training, validation, and test sets using:

- Accuracy  
- Precision / Recall / F1 Score  
- ROC AUC  
- PR AUC  
- Confusion Matrix

Additional plots include training loss/accuracy curves, ROC & PR curves, and gradient norms (for RNN).

---

##Dataset

- **Dataset Source**: [Amazon Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/data)
- Balanced and cleaned subset used:
  - `cleaned_sampled_train.csv` (300k samples)
  - `cleaned_sampled_test.csv` (400k samples)

>Due to GitHub size restrictions, the dataset files are not included here. Please download the raw dataset and follow `1_data_sampling.ipynb` to reproduce the processed CSVs.

---

##How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/sequence-modeling-sentiment.git
   cd sequence-modeling-sentiment/code
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   pip install -r ../requirements.txt
   ```

3. **Run the notebooks in order:**
   - `1_data_sampling.ipynb`
   - `2_dataloader_and_embeddings.ipynb`
   - `3_test_prep.ipynb`
   - `4_simple_rnn.ipynb`
   - `5_Istm.ipynb`
   - `6_bidirectional_Istm.ipynb`
   - `7_glove_freeze.ipynb`
   - `8_glove_unfreeze.ipynb`

4. **Final evaluation** is based on model performance over validation/test sets (detailed in report).

---

##Final Report

The final report is located at `report/report.pdf`. It includes:

- Model descriptions
- Training & validation curves
- ROC and PR curves
- Confusion matrices
- Gradient analysis (for RNN)
- Comparative performance table
- Reflections on model behavior and evaluation metrics

---

##Requirements
install using:
```bash
pip install -r requirements.txt
```

---

##Author

**Abhishek Kumar Chaubey**  
Roll Number: `24144001`  
COPS Summer of Code 2025 – Intelligence Guild (NLP Track)

