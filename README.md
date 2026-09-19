# 🤖 ArminText

**A lightweight Persian language model built from scratch with PyTorch for text generation.**

ArminText is an experimental Persian language model trained from scratch using **PyTorch** and a **GRU-based neural network**.

The goal of this project is to understand how a language model can learn patterns from Persian text and generate new text based on a given prompt.

> 🚧 ArminText is an educational and experimental project, not a production-ready language model.

---

## ✨ Features

* 🇮🇷 Persian text generation
* 🧠 Trained from scratch with PyTorch
* 🔥 GRU-based language model
* 📚 Word-level tokenizer
* 🎯 Next-token prediction
* 🌡️ Temperature-based text generation
* 🔝 Top-K sampling
* 💾 Save and load trained model
* ☁️ Trained using Google Colab

---

## 🧠 Model Architecture

ArminText uses the following pipeline:

```text
Persian Text
     ↓
Word Tokenization
     ↓
Vocabulary
     ↓
Embedding
     ↓
GRU
     ↓
Linear Layer
     ↓
Next-Token Probabilities
     ↓
Generated Persian Text
```

The model predicts the next token based on the tokens that came before it.

---

## ⚙️ Model Configuration

| Parameter           |             Value |
| ------------------- | ----------------: |
| Architecture        |               GRU |
| Embedding Dimension |               192 |
| Hidden Dimension    |               256 |
| GRU Layers          |                 2 |
| Dropout             |               0.2 |
| Sequence Length     |                32 |
| Maximum Vocabulary  |            12,000 |
| Training Epochs     |                 5 |
| Optimizer           |             AdamW |
| Learning Rate       |             0.002 |
| Dataset             | TinyStories-Farsi |

---

## 📊 Training

ArminText-v1 was trained on a subset of the **TinyStories-Farsi** dataset.

Training was performed using:

* Google Colab
* NVIDIA GPU
* PyTorch

Final training results:

```text
Train Loss: 2.1447
Validation Loss: 4.8202
```

The difference between training and validation loss indicates that the model learned the training data substantially better than unseen validation data.

---

## 💬 Example

Given the prompt:

```text
روزی روزگاری
```

ArminText can generate text such as:

```text
روزی روزگاری دختربچه ای به اسم سو در یک خانه کوچک زندگی می‌کرد.
سو دوست داشت در حیاط بازی کند...
```

Another example:

```text
یک روز آفتابی
```

The model can continue the story using patterns learned during training.

---

## 🎛️ Text Generation

ArminText supports adjustable generation parameters:

```python
generate_text(
    "روزی روزگاری",
    max_new_tokens=50,
    temperature=0.8,
    top_k=20
)
```

### Temperature

Controls the randomness of generated text.

* Lower values → more predictable
* Higher values → more random

### Top-K

Limits sampling to the most probable K tokens.

---

## 📁 Project Files

```text
ArminText/
├── ArminText.ipynb
├── ArminText-v1.pt
└── README.md
```

### `ArminText.ipynb`

The complete Google Colab notebook containing data preparation, model architecture, training, saving, and text generation.

### `ArminText-v1.pt`

The trained PyTorch model checkpoint, including model weights, vocabulary, and configuration.

---

## 🛠️ Technologies

* Python
* PyTorch
* Hugging Face Datasets
* Google Colab
* GRU Neural Networks

---

## 🚀 Future Improvements

Possible improvements for future versions:

* Better Persian tokenization
* Subword tokenization
* Larger training dataset
* More training epochs
* Improved text coherence
* Better handling of Persian punctuation
* Transformer-based architecture
* Faster and cleaner inference
* Interactive web demo

---

## 📌 Version

**ArminText-v1**

This is the first experimental version of the project.

---

## 👨‍💻 Creator

**Armin Hamzeh**

An educational AI project focused on learning how language models are built and trained from scratch.

---

## 📜 License

This project is intended for educational and experimental purposes.
