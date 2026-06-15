# 🛍️ ReviewSense: NLP-Based Review Summarization & Rating Prediction

> An end-to-end NLP pipeline that automatically summarizes product reviews and predicts ratings using state-of-the-art deep learning models.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![BART](https://img.shields.io/badge/BART-Summarization-blueviolet)
![BERT](https://img.shields.io/badge/BERT-Rating%20Prediction-informational)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

**ReviewSense** is an end-to-end NLP system that transforms lengthy customer reviews into concise summaries and automatically predicts review ratings — built on top of the **Amazon Reviews 2023 Electronics** dataset.

The system combines:

- 🤖 **BART Large CNN** — Abstractive review summarization
- ⚡ **QLoRA** — Memory-efficient fine-tuning via Quantized Low-Rank Adaptation
- 🧪 **Multiple Prompt Engineering Techniques** — Baseline, instruction, chain-of-thought, self-consistency
- 🎯 **Multiple Decoding Strategies** — Greedy, beam search, top-k, top-p, temperature
- 📊 **BERT-based Rating Predictor** — Estimates ratings from generated summaries
- 📏 **Comprehensive Evaluation** — ROUGE, MAE, and RMSE metrics

---

## ❓ Problem Statement

Online shopping platforms contain millions of customer reviews. Reading every review is time-consuming and impractical.

ReviewSense addresses this by:

- 📝 Generating concise summaries of lengthy customer reviews
- 💡 Preserving key product opinions and sentiments
- ⭐ Automatically predicting the corresponding review rating
- 🔁 Creating an end-to-end pipeline from raw reviews to actionable insights

---

## 🎯 Objectives

- [x] Perform comprehensive review data preprocessing
- [x] Fine-tune BART for review summarization
- [x] Apply QLoRA for parameter-efficient training
- [x] Experiment with multiple prompting techniques
- [x] Compare different decoding strategies
- [x] Evaluate summaries using ROUGE metrics
- [x] Build a BERT-based rating prediction model
- [x] Evaluate rating prediction using MAE and RMSE
- [x] Create a complete Summarization → Rating Prediction pipeline

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Source** | [McAuley Lab — Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) |
| **Category** | Electronics |
| **Format** | JSONL (one review per line) |

**Fields used:** `text`, `rating`, `title`, `verified_purchase`, `user_id`, `timestamp`

```json
{
  "text": "Excellent battery life...",
  "rating": 5,
  "title": "Great phone"
}
```

---

## 🏗️ System Architecture

```
Amazon Review
      │
      ▼
Data Cleaning & EDA
      │
      ▼
BART Large CNN
(QLoRA Fine-Tuning)
      │
      ▼
Generated Summary
      │
      ▼
Prompt Engineering
      │
      ▼
Decoding Strategy
      │
      ▼
ROUGE Evaluation
      │
      ▼
BERT Rating Predictor
      │
      ▼
Predicted Rating
      │
      ▼
MAE & RMSE Evaluation
```

---

## 🔬 Project Phases

### Phase 1 — Data Collection

- Electronics dataset downloaded from the Amazon Reviews 2023 repository
- Stored in JSONL format; each line represents one review

---

### Phase 2 — Exploratory Data Analysis (EDA)

| Analysis | Purpose |
|---|---|
| **Reviews Over Time** | Identify review trends and dataset growth |
| **Review Length Distribution** | Understand word/character size variations |
| **Rating Distribution** | Detect class imbalance; understand satisfaction patterns |
| **Word Clouds** (Positive ≥ 4★ / Negative ≤ 2★) | Visualize frequent words and customer opinions |
| **N-Gram Analysis** (Uni/Bi/Trigrams via CountVectorizer) | Discover common phrases and product features |
| **Sentiment Analysis** (TextBlob) | Validate consistency between text sentiment and ratings |

#### 🚨 Fake Review Detection

Heuristic-based indicators used:

| Indicator | Condition |
|---|---|
| Short Review | Length < 20 characters |
| Unverified Purchase | `verified_purchase == False` |
| Rating-Text Mismatch | High rating with minimal content |
| High Activity Users | Unusually large number of reviews per user |

A combined **fake review score** was computed across all indicators.

---

### Phase 3 — Data Preprocessing

**Cleaning Steps:**

| Step | Example |
|---|---|
| Lowercasing | `Excellent Phone` → `excellent phone` |
| HTML Removal | `<b>Great</b>` → `Great` |
| URL Removal | Removes all hyperlinks |
| Emoji Removal | `😍 Great phone` → `Great phone` |
| Unicode Normalization | Standardizes text encoding |
| Special Character Removal | Removes unnecessary symbols |
| Whitespace Normalization | Collapses multiple spaces |

**Length Filtering:**

```
50 ≤ Review Length ≤ 400 words
Title Length ≥ 3 words
```

**Train/Validation Split:** `90% Training` / `10% Validation` *(stratified on ratings)*

---

### Phase 4 — Review Summarization

#### Model: `facebook/bart-large-cnn`

BART combines a **bidirectional encoder** (like BERT) with an **autoregressive decoder** (like GPT), making it highly effective for summarization, translation, and text generation.

---

#### ⚡ QLoRA Fine-Tuning

Instead of full fine-tuning, **QLoRA** was used for parameter-efficient training:

**Quantization:**
```python
load_in_4bit = True
bnb_4bit_quant_type = "nf4"
```

**LoRA Configuration:**
```python
r = 8
lora_alpha = 16
lora_dropout = 0.05
target_modules = ["q_proj", "v_proj"]
```

> Only LoRA adapters were trained. Original BART weights remained **frozen**.

**Training Configuration:**

| Parameter | Value |
|---|---|
| Epochs | 2 |
| Learning Rate | 2e-4 |
| Batch Size | 2 |
| Precision | FP16 |

---

#### 🧪 Prompt Engineering Techniques

| Technique | Input Format |
|---|---|
| **Baseline** | `<review text>` |
| **Instruction** | `Summarize the following product review concisely: <review>` |
| **Chain-of-Thought** | `Identify key points and sentiment, then write a concise summary: <review>` |
| **Self-Consistency** | Generate multiple summaries → select best candidate |

---

#### 🎛️ Decoding Strategies

| Strategy | Description |
|---|---|
| **Greedy Search** | Selects highest probability token; fast but less diverse |
| **Beam Search** | Maintains multiple candidate sequences; higher quality |
| **Top-K Sampling** | Samples from top K probable tokens; improves diversity |
| **Top-P Sampling** | Samples from cumulative probability mass; balances quality & diversity |
| **Temperature Sampling** | Controls generation randomness |

---

#### 📏 Summary Evaluation Metrics

| Metric | Measures |
|---|---|
| **ROUGE-1** | Unigram overlap |
| **ROUGE-2** | Bigram overlap |
| **ROUGE-L** | Longest common subsequence |
| **ROUGE-Lsum** | Sentence-level structural similarity |

---

### Phase 5 — Rating Prediction

#### Model: `bert-base-uncased`

```
Generated Summary
       │
       ▼
BERT Tokenizer
       │
       ▼
BERT Encoder
       │
       ▼
[CLS] Sentence Embedding
       │
       ▼
Dropout Layer
       │
       ▼
Linear Regression Layer
       │
       ▼
Predicted Rating
```

The **[CLS] token embedding** serves as the full summary representation, feeding into a regression head.

**Loss Function:** Mean Squared Error (MSE)

$$MSE = \frac{1}{n} \sum (y_{true} - y_{pred})^2$$

**Evaluation Metrics:**

| Metric | Formula | Purpose |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum\|y_{true} - y_{pred}\|$ | Average absolute prediction error |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_{true} - y_{pred})^2}$ | Penalizes larger prediction errors |

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch |
| **Transformers** | Hugging Face Transformers, PEFT |
| **Efficient Training** | BitsAndBytes (QLoRA) |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **NLP Utilities** | NLTK, SpaCy, TextBlob |
| **Evaluation** | ROUGE Score |
| **Visualization** | Matplotlib, Seaborn |

---

## ✅ Key Achievements

- ✔️ Data preprocessing and cleaning completed
- ✔️ Exploratory data analysis performed
- ✔️ Fine-tuned BART Large CNN using QLoRA
- ✔️ Implemented parameter-efficient fine-tuning
- ✔️ Applied multiple prompting techniques
- ✔️ Compared multiple decoding strategies
- ✔️ Evaluated summarization using ROUGE metrics
- ✔️ Built BERT-based rating prediction model
- ✔️ Evaluated ratings using MAE and RMSE
- ✔️ Developed a complete end-to-end pipeline
- ✔️ Ready for deployment and academic presentation

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/your-username/ReviewSense.git
cd ReviewSense
pip install -r requirements.txt
```

### Requirements

```bash
pip install torch transformers peft bitsandbytes pandas numpy scikit-learn \
            matplotlib seaborn nltk spacy textblob rouge-score
```

### Run the Pipeline

```bash
# Step 1: Preprocess data
python src/preprocess.py

# Step 2: Fine-tune BART with QLoRA
python src/train_summarizer.py

# Step 3: Generate summaries
python src/generate_summaries.py

# Step 4: Train BERT rating predictor
python src/train_rating_predictor.py

# Step 5: Evaluate end-to-end pipeline
python src/evaluate.py
```

---

## 🔭 Future Enhancements

- [ ] Deploy as a web application using **Streamlit**
- [ ] Add **real-time review analysis**
- [ ] Support **multilingual reviews**
- [ ] Integrate **sentiment-aware summarization**
- [ ] Improve rating prediction using **larger encoder models**
- [ ] Add **explainable AI visualizations** for summaries and ratings

---



<p align="center">Built with ❤️ for smarter, faster product review intelligence</p>
