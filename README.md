# 💡 Safe and Responsible Multi-Modal Toxic Content Moderation

## 📘 Overview

This project builds a dual-stage content moderation system capable of detecting harmful content in both **text and images**. It leverages **Natural Language Processing (NLP)** and **Computer Vision** techniques using models such as **DistilBERT**, **Llama Guard**, and **BLIP**, all deployed through a user-friendly **Streamlit** interface.

---

## 🎯 Objectives

- ✅ Classify user-generated content into 9 safety/toxicity categories.
- ✅ Filter *extremely dangerous* content using **Llama Guard**.
- ✅ Apply **DistilBERT fine-tuned with LoRA** for nuanced classification.
- ✅ Use **BLIP** to caption uploaded images, then classify the caption using the same text pipeline.
- ✅ Compare performance of DistilBERT (LoRA) with a **CNN-LSTM baseline**.
- ✅ Deploy the complete system using **Streamlit**.

---

## 🗃️ Dataset

A multi-class dataset containing 9 categories:

- `Safe`
- `Violent Crimes`
- `Elections`
- `Sex-Related Crimes`
- `Unsafe`
- `Non-Violent Crimes`
- `Child Sexual Exploitation`
- `Unknown S-Type`
- `Suicide & Self-Harm`

---

## 🔧 System Architecture

### 🛡️ Stage 1: Hard Filter — **Llama Guard**

- Detects and blocks **legally disallowed** or **highly dangerous** content.
- Protects against catastrophic moderation failures (e.g., child exploitation, terrorism).

### 🎯 Stage 2: Soft Classifier — **DistilBERT with LoRA**

- Provides **fine-grained classification** of allowed but harmful content.
- Competes with a **CNN-LSTM baseline** to validate performance.

### 🖼️ Image Extension — **BLIP + Text Pipeline**

- Captions images using **BLIP**.
- Applies the **same text moderation** process on the generated caption.

---

## ⚙️ Technical Stack

| Component             | Tool/Library                        |
|----------------------|-------------------------------------|
| Text Tokenization     | Hugging Face Transformers (DistilBERT) |
| Text Cleaning         | `re`, `spaCy`                       |
| Image Captioning      | BLIP (Bootstrapped Language-Image Pretraining) |
| Classification        | DistilBERT + LoRA / CNN-LSTM       |
| Evaluation            | Accuracy, Precision, Recall, F1     |
| Web Interface         | Streamlit                          |

---

## 🧼 Preprocessing Steps

1. Lowercase and normalize text
2. Remove URLs, HTML tags, and mentions
3. Retain punctuation for emotion/context
4. Tokenize using DistilBERT tokenizer
5. Encode labels as integers
6. Split into train, validation, and test sets

---

## 📈 Benchmarking & Evaluation

| Model                 | Description                        | Purpose                         |
|----------------------|------------------------------------|----------------------------------|
| **CNN-LSTM**         | Deep learning on word embeddings   | Baseline comparison              |
| **DistilBERT + LoRA**| Fine-tuned transformer with PEFT   | Main classifier                  |
| **Metrics**          | Accuracy, F1-score, Recall, Precision | Evaluate performance           |

---

## 🚀 Streamlit Web App

- Accepts both **text input** and **image upload**
- For images:
  - ➡️ Uses **BLIP** to generate a caption
  - ➡️ Passes the caption through the moderation pipeline
- Displays:
  - Predicted category
  - Category probabilities
  - Safety flag (Blocked / Allowed)



---

## 🔐 Why Dual-Stage Moderation?

| Stage         | Purpose                                      |
|---------------|----------------------------------------------|
| **Llama Guard** | Detects and blocks illegal or extreme content |
| **DistilBERT + LoRA** | Provides detailed classification for moderation teams |
| ✅ Benefits    | Higher safety, fewer false negatives, compliance with law and policy |

