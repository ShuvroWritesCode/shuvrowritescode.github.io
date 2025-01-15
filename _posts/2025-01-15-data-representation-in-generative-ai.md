---
layout: post
title: Data Representation in Generative AI
dates: 15-01-2025
categories: [Generative AI]
tag: [Generative AI, Data Representation, one hot encoding, word2vec, BOW, Machine Learning, Natural Language Processing, Vectorization, Deep Learning]
description: 
image:
    path: /assets/img/headers/post3.webp
    lqip: data:image/webp;base64,UklGRoIAAABXRUJQVlA4IHYAAACQAwCdASoUAAoAPpE4l0eloyIhMAgAsBIJZQAAW9ltDR1TBgMAAP7XP869nDh+xTmrig7IkwQJYz4nl3S1T9r4SvqGrcLw0o9XjvA/RQogLuZW9ifqn33D9vfC24dKrB7eJXh5ti61ZU2lvMYwmPELghGL3AAA
---

# Understanding Data Representation in Generative AI

Generative AI relies on raw data—text, images, audio, and more. However, computers don't inherently understand this kind of data except numbers. To use it effectively in AI models, we need to convert it into numerical values. This process is called **data representation or vectorization**.

## What is Feature Extraction?

Feature extraction is like digging into the data and finding its essential characteristics. We convert these features into numbers (mathematical notations) that a machine can understand. Imagine speaking a foreign language to someone who doesn't know it. You'd need a translator. Similarly, feature extraction is the translator of raw data for AI. It converts raw data—text, images, or audio—into machine understandable numerical vectors.

- **Text Data** → Words are broken down into smaller units (e.g., tokens or vectors).
- **Image Data** → Every pixel in an image has a numerical value (e.g., 0–255).
- **Audio Data** → Frequencies and amplitudes are represented in numbers.

## Why is Feature Extraction Important?

AI models process numbers, not raw data. Without converting data into numbers:
- Models wouldn't "understand" the data.
- You couldn't use AI for tasks like text generation, image synthesis, or speech recognition.

## Why is Feature Extraction from Text Difficult?

In traditional Machine Learning, we work with structured data (e.g., tables). For example:
- A table with 5 rows and 4 columns has 20 data points.
- Input sizes are fixed, making it easier to process.

But in **Generative AI**, the input is unstructured and it's huge:
- **Text**: Sentences vary in length, tone and complexity.
- **Images**: Thousands of pixels need to be processed.
- **Audio**: Frequencies continuously change.

This variability makes feature extraction more challenging.

## Techniques for Text Data Representation

### A. Tokens
Sentences are broken into tokens (individual words or phrases). For example:
- Sentence: **"This is an apple."**
  Tokens: **This, is, an, apple**

### B. One-Hot Encoding
This method turns words into binary vectors. Each word is mapped to a unique position.
**Example:**  
- **Corpus (all text):**  
  ```
  I got a pen  
  A pen got ink  
  Ink is black  
  I love black  
  ```

- **Vocabulary:** `i, got, a, pen, ink, is, black, love`  
  (There are 8 unique words.)

### **One-Hot Encoded Representation:**

| i | got | a | pen | ink | is | black | love |
|---|-----|---|-----|-----|----|-------|------|
| 1 | 1   | 1 | 1   | 0   | 0  | 0     | 0    |
| 0 | 1   | 1 | 1   | 1   | 0  | 0     | 0    |
| 0 | 0   | 1 | 0   | 1   | 1  | 1     | 0    |
| 1 | 0   | 0 | 0   | 0   | 0  | 1     | 1    |

### Drawbacks of One-Hot Encoding

1. **High Dimensionality**: As the vocabulary grows, the size of the vector increases.
2. **Sparsity**: Most values are zero, wasting computational resources.
3. **No Semantic Meaning**: Words like "apple" and "fruit" aren't recognized as related.

### C. Bag of Words (BOW)

BOW counts the occurrences of words in a sentence or document. It doesn’t consider word order, but it captures word frequency.

| i | got | a | pen | ink | is | black | love |
|---|-----|---|-----|-----|----|-------|------|
| 1 | 1   | 1 | 1   | 0   | 0  | 0     | 0    |
| 0 | 1   | 1 | 1   | 1   | 0  | 0     | 0    |
| 0 | 0   | 1 | 0   | 1   | 1  | 1     | 0    |
| 1 | 0   | 0 | 0   | 0   | 0  | 1     | 1    |

### D. Advanced Techniques

1. **TF-IDF (Term Frequency-Inverse Document Frequency):**
   Weighs words based on their importance in a document and across a corpus.
   
2. **Word2Vec:**
   - Uses deep learning to create vector representations of words.
   - Captures semantic meaning (e.g., "king - man + woman = queen").

3. **Transformers:**
   - Modern techniques like BERT and GPT use contextual embeddings.
   - Words are represented based on their meaning in a sentence.

[View Notebook](/assets/ipynb-files/Text_Representation_Word_Embeddings.ipynb)

## Core Idea

In Generative AI, data representation is the bridge between humans and machines. While early techniques like One-Hot Encoding are simple, they don’t capture meaning well. Advanced methods like Word2Vec and Transformers revolutionize how data is understood, enabling more accurate AI models.

---

By mastering feature extraction and representation, you’re not just preparing data—you’re teaching machines to "think" and "understand."