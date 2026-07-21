# 🧠 Next Word Prediction using LSTM

> A complete **Next Word Prediction** pipeline built from scratch using **PyTorch**, including custom tokenization, preprocessing, dataset generation, training, and inference.

This project was created to understand how language models work **before Transformers**, by implementing an LSTM-based language model from scratch.

---

# ✨ Features

- ✅ Custom Byte Pair Encoding (Word-Level) tokenizer built from scratch
- ✅ Custom preprocessing pipeline
- ✅ Reddit Jokes dataset for training
- ✅ Dynamic training sample generation
- ✅ Sequence padding
- ✅ LSTM Language Model
- ✅ Training checkpointing
- ✅ Inference script for text generation
- ✅ Modular project structure

---

# 📂 Project Structure

```text
.
├── config.py          # Hyperparameters and configuration
├── model.py           # LSTM model
├── preparing.py       # Dataset preparation & vocabulary creation
├── utils.py           # Utility functions
├── trainning.py       # Model training
├── inference.py       # Generate next words
└── README.md
```

---

# 📖 Project Pipeline

## 1️⃣ Tokenization

Instead of using an existing tokenizer, a **custom Byte Pair Encoding inspired tokenizer** was implemented.

> **Note**
>
> This is **not the official Byte Pair Encoding algorithm**.
>
> The implementation reproduces the **merge mechanism** of BPE at the **word level**, but does **not operate on raw bytes** like the original algorithm.

The goal was to understand **how BPE progressively merges tokens** without relying on external libraries.

---

## 2️⃣ Dataset

The model is trained on approximately **10,000 words** sampled from the **Reddit Jokes Dataset**.

Each sentence is cleaned before being converted into numerical tokens.

---

## 3️⃣ Training Sample Generation

Instead of predicting every word in the sentence simultaneously, the dataset is transformed into multiple training examples.

Example sentence:

```text
I love deep learning
```

Generated samples:

| Input | Target |
|-------|--------|
| I | love |
| I love | deep |
| I love deep | learning |

Each training example teaches the model to predict **the next word** given all previous words.

---

## 4️⃣ Padding

Since every training example has a different sequence length, all sequences are padded with **0** to make them equal in length.

Example:

```text
Original

I
I love
I love deep

↓

Padded

[0,0,0,I]
[0,0,I,love]
[0,I,love,deep]
```

This allows efficient batch training using PyTorch.

---

# 🏗 Model Architecture

The language model consists of:

- Embedding Layer
- Dropout
- 2-layer LSTM
- Dropout
- Fully Connected Layer
- Vocabulary Prediction

---

## Architecture Diagram

```text
                    Input Tokens
                          │
                          ▼
               ┌────────────────────┐
               │  Embedding Layer   │
               │    (384 dims)      │
               └────────────────────┘
                          │
                          ▼
                    Dropout (p)
                          │
                          ▼
          ┌────────────────────────────┐
          │        2-Layer LSTM        │
          │                            │
          │ Hidden Size = 512          │
          │ Batch First = True         │
          └────────────────────────────┘
                          │
                 Final Hidden State
                          │
                          ▼
                    Dropout (p)
                          │
                          ▼
               ┌────────────────────┐
               │  Linear Layer      │
               │ 512 → Vocabulary   │
               └────────────────────┘
                          │
                          ▼
                 Next Word Prediction
```

---

# ⚙ Model Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Size | **384** |
| Hidden Size | **512** |
| LSTM Layers | **2** |
| Dropout | Configurable |
| Batch First | ✅ True |

---

# 🧠 Model Implementation

```python
Embedding (384)
        │
        ▼
Dropout
        │
        ▼
2-Layer LSTM
Hidden Size = 512
        │
        ▼
Final Hidden State
        │
        ▼
Dropout
        │
        ▼
Linear Layer
512 → Vocabulary Size
        │
        ▼
Predicted Next Word
```

---

# 🏋 Training

The model is trained using

- Cross Entropy Loss
- Adam Optimizer
- GPU Support
- Model Checkpointing

During training, checkpoints are saved so training can be resumed later without starting from scratch.

---

# 🚀 Inference

Given an input sentence:

```text
hey man
```

the model predicts

```text
hey man ...
```

The predicted word is appended back to the sentence and fed into the model again to generate multiple words.

---

# 📚 Concepts Practiced

- Byte Pair Encoding (Mechanism)
- Word-Level Tokenization
- Vocabulary Creation
- Sequence Padding
- Embedding Layers
- LSTM Networks
- Language Modeling
- Teacher Forcing Dataset Creation
- Next Word Prediction
- Model Checkpointing
- PyTorch Training Pipeline

---

# 🛠 Tech Stack

- Python
- PyTorch
- NumPy

---

# 🎯 Future Improvements

- Implement the **official Byte Pair Encoding** algorithm using raw bytes.
- Train on a significantly larger corpus.
- Replace the LSTM with a GRU and compare results.
- Add Beam Search for inference.
- Implement a Transformer-based language model for comparison.
- Support top-k and top-p sampling during text generation.

---

# 📌 Learning Objective

The primary goal of this project was **not to build the most accurate language model**, but to understand every stage of a language modeling pipeline by implementing the components manually.

This project serves as a strong foundation for understanding modern Transformer-based language models.
