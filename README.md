# 🔤 NLP Tokenization — Complete Guide

A practical guide on **what tokenization is**, **all 5 types**, and most importantly — **when to use which one**.

---

## 📌 What is Tokenization?

Tokenization is the process of breaking raw text into smaller units called **tokens** so that a machine learning model can process it. It is always the **first step** in any NLP pipeline.

```
"Let's tokenize!" → ["Let", "'s", "tokenize", "!"]   ← word-based example
```

---

## 🗂️ The 5 Types at a Glance

| # | Type | Splits On | Vocab Size | OOV Risk | Best Known Use |
|---|------|-----------|------------|----------|----------------|
| 1 | Word-based | Spaces / punctuation | Large (50k+) | ❌ High | spaCy, NLTK |
| 2 | Subword | Frequent byte/char pairs | Medium (30–50k) | ✅ None | BERT, GPT, LLaMA |
| 3 | Character | Every character | Tiny (~100) | ✅ None | CharRNN |
| 4 | Byte-level | Raw UTF-8 bytes | Fixed (256) | ✅ None | GPT-2, ByT5 |
| 5 | Custom/Rule-based | Your own regex/logic | Flexible | Varies | Domain NLP |

---

## 1. 📝 Word-Based Tokenization

### How it works
Splits text on whitespace and punctuation. Each unique word is one token.

```python
# spaCy example
tokens = [token.text for token in nlp("Hello, world!")]
# → ['Hello', ',', 'world', '!']
```

### ✅ Use it when
- You are doing **simple, classical NLP** tasks (keyword extraction, basic search)
- Your text is **clean and well-structured** (formal English documents)
- You need **human-readable, interpretable** tokens
- Building **rule-based systems** (search engines, simple chatbots)
- Working with **small, domain-specific vocabularies** (e.g. legal codes, product catalogs)

### ❌ Avoid it when
- Your text has **rare, misspelled, or made-up words** — they become `<UNK>`
- Working with **multiple languages** in the same corpus
- Building **large-scale deep learning** models (vocab size explodes)
- Text contains **social media slang**, hashtags, or emojis

### ⚠️ Key Weakness
> "unhappy", "happy", "happily" are **3 separate unrelated tokens** — the model learns no connection between them.

---

## 2. 🧩 Subword-Based Tokenization

Subword is a **family** of 3 algorithms. They all share the same idea: keep common words whole, split rare words into meaningful pieces.

```
"unhappiness" → ["un", "##happy", "##ness"]   ← WordPiece (BERT)
"tokenizing"  → ["token", "izing"]             ← BPE (GPT)
```

### The 3 Subword Algorithms

| Algorithm | Prefix Style | Used By | Key Idea |
|-----------|-------------|---------|----------|
| **BPE** (Byte-Pair Encoding) | `Ġ` for space | GPT-2, RoBERTa | Merge most frequent pairs iteratively |
| **WordPiece** | `##` for continuation | BERT, DistilBERT | Maximize language model likelihood |
| **SentencePiece / Unigram** | `▁` for space | T5, LLaMA, XLNet | Language-agnostic, trains on raw text |

### ✅ Use it when
- Training or fine-tuning **transformer models** (BERT, GPT, LLaMA, T5)
- Your text has **mixed languages** or **multilingual data**
- You want a **balanced vocab** — not too large, not too small
- Text contains **morphologically rich words** (e.g. German, Finnish, Turkish)
- You need **zero OOV (out-of-vocabulary)** risk in production

### ❌ Avoid it when
- You need **fully human-readable** tokens (subwords like `##izing` are less natural)
- Building a **simple pipeline** where word tokens are sufficient
- You need **exact word boundaries** for downstream rules

### 💡 Which subword algorithm to pick?
- Fine-tuning **BERT** → WordPiece (already built in)
- Fine-tuning **GPT / LLaMA** → BPE (already built in)
- Training your **own model from scratch** → SentencePiece (most flexible)

---

## 3. 🔡 Character-Based Tokenization

### How it works
Every single character (including spaces and punctuation) becomes one token.

```python
tokens = list("Hello!")
# → ['H', 'e', 'l', 'l', 'o', '!']
```

### ✅ Use it when
- Working with **noisy or informal text** (typos, SMS, tweets)
- Building models for **languages with no clear word boundaries** (Chinese, Japanese, Thai)
- Tasks where **morphology matters deeply** (e.g. spelling correction, password strength)
- Doing **text generation at the character level** (CharRNN, creative writing models)
- Your domain has many **out-of-vocabulary or invented words**

### ❌ Avoid it when
- Working with **long documents** — sequences become extremely long, increasing memory and compute cost
- You need the model to understand **word-level semantics** quickly
- **Latency matters** — more tokens = slower inference

### ⚠️ Key Weakness
> "cat" needs 3 tokens. A 500-word document becomes ~2500 tokens. Transformers struggle with this due to quadratic attention cost.

---

## 4. 🔢 Byte-Level Tokenization

### How it works
Converts text to raw **UTF-8 bytes**. Vocab is always exactly 256 (one per byte value 0–255).

```python
tokens = list("Hi!".encode("utf-8"))
# → [72, 105, 33]
```

### ✅ Use it when
- Building **language-agnostic** models that must handle any script (Arabic, Chinese, emoji, code)
- You need **absolute zero OOV** — every possible input is representable
- Working with **binary or mixed-encoding data**
- Reproducing GPT-2 / ByT5 style architectures
- Processing **code, URLs, file paths** that contain unusual characters

### ❌ Avoid it when
- Sequence length is a constraint — bytes produce even **longer sequences than characters**
- You need **linguistic structure** in your tokens
- Working on a **resource-constrained** device

### 💡 Fun fact
> This is why GPT-2 can handle any language and emoji without ever seeing `<UNK>` — it never needs one.

---

## 5. ⚙️ Custom / Rule-Based Tokenization

### How it works
You define your own splitting logic using **regex patterns** or hand-written rules.

```python
import re

# Keep words, numbers, and punctuation as separate tokens
tokens = re.findall(r"\w+|[^\w\s]", "for i in range(10): print(i**2)")
# → ['for', 'i', 'in', 'range', '10', ':', 'print', 'i', '**', '2']
```

### ✅ Use it when
- Working in a **specialized domain** with unique vocabulary:
  - 🏥 Medical: `ICD-10`, drug names, dosages
  - 💻 Code: operators, brackets, indentation
  - 💬 Social media: `#hashtags`, `@mentions`, URLs
  - 🧬 Biology: gene names, protein sequences
- You need **deterministic, reproducible** tokenization with no model needed
- Preprocessing text **before** feeding into a larger tokenizer
- Building **fast, lightweight** pipelines without ML dependencies

### ❌ Avoid it when
- You want the tokenizer to **learn** from data — rules don't adapt
- Covering **all edge cases** is not feasible
- Your text is too **diverse or unpredictable** for fixed rules

---

## 🧭 Quick Decision Guide

```
What are you building?
│
├── Simple NLP (search, keywords, rules)?
│     └── ✅ Word-based (spaCy / NLTK)
│
├── Training / fine-tuning a Transformer?
│     ├── BERT family?      → ✅ WordPiece
│     ├── GPT / LLaMA?      → ✅ BPE
│     └── Custom / multilingual? → ✅ SentencePiece
│
├── Noisy text or character-level generation?
│     └── ✅ Character-based
│
├── Multilingual / any-script / zero OOV needed?
│     └── ✅ Byte-level
│
└── Domain-specific (medical, code, social media)?
      └── ✅ Custom / Rule-based
```

---

## 📦 Libraries & Tools

| Library | Tokenizers Supported | Install |
|---------|---------------------|---------|
| `spaCy` | Word-based | `pip install spacy` |
| `nltk` | Word, Rule-based | `pip install nltk` |
| `tokenizers` (HuggingFace) | BPE, WordPiece, Unigram | `pip install tokenizers` |
| `transformers` (HuggingFace) | All pretrained tokenizers | `pip install transformers` |
| `sentencepiece` | SentencePiece / Unigram | `pip install sentencepiece` |
| `re` (built-in) | Custom / Rule-based | Built into Python |

---

## 📚 Summary

> **For most modern deep learning NLP work → use Subword (BPE or WordPiece).**  
> It is the sweet spot between vocabulary size, coverage, and model performance.  
> Fall back to Character or Byte-level only when you need to handle truly arbitrary input.  
> Use Word-based or Custom when you need speed, interpretability, or domain-specific rules.
