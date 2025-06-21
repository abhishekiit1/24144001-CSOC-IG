# English–Spanish Neural Machine Translation

This repository contains implementations and evaluation of different sequence-to-sequence models for **English to Spanish** translation as part of the CSOC 2025 NLP Track Project. The task involves training and comparing multiple models on the OPUS Books dataset using PyTorch and Hugging Face Transformers.

---

##  Folder Structure

```
.
├── code/
│   ├── seq2seq_without_attention.ipynb      # Basic Encoder–Decoder (LSTM) model
│   ├── seq2seq_with_attention.ipynb         # LSTM model with Attention Mechanism
│   ├── transformers_implementation.ipynb    # Transformer model using MarianMT
│
├── report/
│   ├── seq2seq_report.tex                   # LaTeX source for report
│   ├── seq2seq_report.pdf                   # Final compiled project report
```

---

##  Project Description

The goal of this project is to explore and compare different neural machine translation approaches:

###  Implemented Models:
- **Vanilla Seq2Seq (LSTM)**: Encoder–decoder model without attention.
- **Seq2Seq with Attention**: Adds Bahdanau-style attention mechanism for better performance on longer sequences.
- **Transformer-based MarianMT**: Pretrained Helsinki-NLP MarianMT model fine-tuned on the OPUS Books English–Spanish dataset.

---

##  Evaluation Metrics

- **BLEU Score**: Calculated on both validation and test sets.
- **Loss Curves**: Plotted for training and validation over epochs.
---

##  Dataset

- **OPUS Books (en-es)** from Hugging Face Datasets
- Split into training (80%), validation (10%), and test (10%) sets
- Preprocessing done using Hugging Face Tokenizer (MarianTokenizer)

---

## 🛠 How to Run

1. Clone this repo:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Navigate to the `code` directory and open any `.ipynb` notebook in Jupyter or VS Code.

3. Required packages:
    ```
    pip install torch transformers datasets sacrebleu matplotlib
    ```

4. Train or evaluate models by running the cells in order.

---

##  Notes

- Due to time constraints and GPU limitations, only **2 epochs** of fine-tuning were done for the Transformer model.
- Transformer was trained on **entire dataset**, while the other models were trained on only **15k samples** for faster prototyping.
- Loss logs were extracted manually for visualization due to lack of automatic log saving during training.

---

##  Report

The detailed report discussing:
- Implementation strategy,
- Model architecture,
- Evaluation metrics,
- Observations and limitations

...is available in the `report/` folder as both `.tex` and compiled `.pdf`.

---

##  Author

- **Name**: Abhishek Chuabey  
- **IIT BHU | CSOC 2025 NLP Track**

---

## Acknowledgements

- Club of Programmers, IIT (BHU) Varanasi
- Hugging Face Datasets and Transformers
- CSOC 2025 Organizers and Mentors

---
