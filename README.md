# CSL 7640 – Natural Language Understanding
## Assignment 2: Word Embeddings & Character-Level Name Generation


## Problem 1 — Word Embeddings on IIT Jodhpur Corpus

Trains Word2Vec models (CBOW and Skip-gram) **from scratch in PyTorch** on text scraped from IIT Jodhpur institutional sources, and compares results against **Gensim** reference implementations.

### Data Sources
- IIT Jodhpur official website (departments, programmes, research pages)
- Academic Regulation Documents

### Preprocessing Pipeline
| Step | Operation |
|------|-----------|
| 1 | URL and email removal |
| 2 | Non-ASCII / non-English character removal |
| 3 | Standalone number removal |
| 4 | Punctuation normalisation |
| 5 | Lowercasing |
| 6 | Sentence tokenisation (min 3 tokens) |
| 7 | Word tokenisation and filtering (min length 2, alphabetic only) |
| 8 | Minimum frequency filtering (min\_count = 2) |

### Models & Configurations
Both CBOW and Skip-gram are trained across 4 hyperparameter configs:

| Config | Embedding Dim | Window | Negative Samples |
|--------|--------------|--------|-----------------|
| 1 | 50 | 2 | 5 |
| 2 | 100 | 2 | 5 |
| 3 | 100 | 4 | 5 |
| 4 | 100 | 2 | 10 |

### Key Results
- Top-5 nearest neighbours (cosine similarity) for: `research`, `student`, `phd`, `exam`
- Analogy experiments using 3CosAdd: `UG:BTech :: PG:?`
- PCA and t-SNE visualisations of semantic clusters

---

## Problem 2 — Character-Level Name Generation

Implements and compares three sequence models for character-level Indian name generation, all **from scratch in PyTorch**.

### Models

| Model | Architecture | Params |
|-------|-------------|--------|
| **Vanilla RNN** | Embedding → RNN (tanh) → Linear | ~1.5M |
| **BLSTM** | Embedding → Bidirectional LSTM (2 layers) → Linear | ~3.2M |
| **RNN + Attention** | Embedding → LSTM → Bahdanau Attention → Linear | ~1.8M |

### Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **Novelty Rate** | % of generated names not present in the training set |
| **Diversity** | Unique generated names / total generated names |

### Training Setup
- Optimizer: Adam (lr = 0.001)
- Epochs: 30 | Batch size: 64
- Gradient clipping: max norm = 1.0
- LR scheduler: ReduceLROnPlateau (patience = 5)
- 80/20 train/validation split

---

## How to Run

Both notebooks are designed for **Google Colab** (GPU recommended).

```
1. Open the notebook in Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Upload the required data file (raw_corpus.txt or TrainingNames.txt)
4. Run all cells top to bottom
```

**Dependencies** (auto-installed in notebooks):
`torch` `gensim` `wordcloud` `scikit-learn` `matplotlib` `numpy`

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Gensim 4.x (Problem 1 only)
