"""
Module 6 Week B — Lab: Embeddings Comparison

Compare three text representation methods — TF-IDF, GloVe, and
DistilBERT — on the BBC News corpus (5 categories).
"""
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def build_tfidf(texts):
    """Build TF-IDF representations for a list of texts.

    Returns (tfidf_matrix, vectorizer).
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


def compute_tfidf_similarity(tfidf_matrix):
    """Compute pairwise cosine similarity from a TF-IDF matrix.

    Returns a numpy array of shape (n, n).
    """
    return sklearn_cosine(tfidf_matrix)


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Returns a dict mapping each word to a numpy array.
    """
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings


def text_to_glove(text, embeddings):
    """Compute the average GloVe embedding for a text.

    Skip out-of-vocabulary words. If every word is OOV, return a zero
    vector of shape (50,).
    """
    dim = 50  
    vectors = []
    for word in text.lower().split():
        if word in embeddings:
            vectors.append(embeddings[word])
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vectors, axis=0)


def extract_bert_embedding(text, tokenizer, model):
    """Extract a sentence embedding from DistilBERT.

    Returns a numpy array of shape (768,).
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
 
    with torch.no_grad():
        outputs = model(**inputs)
 
    # last_hidden_state: (batch=1, seq_len, hidden=768)
    last_hidden_state = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]  # (1, seq_len)
 
    # Expand mask to match hidden dimension for broadcasting
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
 
    # Sum only non-padding token vectors
    sum_hidden = torch.sum(last_hidden_state * mask_expanded, dim=1)
    # Divide by the count of non-padding tokens (clamp avoids /0)
    token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
    mean_pooled = sum_hidden / token_counts  # (1, 768)
    return mean_pooled.squeeze(0).numpy()

def compare_similarities(texts, queries, tfidf_sim, glove_embeddings,
                         bert_model, bert_tokenizer):
    """Compare similarity rankings across TF-IDF, GloVe, and BERT.
 
    Parameters
    ----------
    texts           : list[str]  – full corpus
    queries         : list[str]  – query strings (subset / equal to texts)
    tfidf_sim       : np.ndarray – precomputed (n, n) TF-IDF similarity matrix
    glove_embeddings: dict       – word → numpy array from load_glove()
    bert_model      : AutoModel  – loaded DistilBERT model (eval mode)
    bert_tokenizer  : AutoTokenizer
 
    Returns
    -------
    dict keyed by query string:
        {
            query_text: {
                "tfidf": [(text, score), ...],  # top-3, excluding query itself
                "glove": [(text, score), ...],
                "bert":  [(text, score), ...],
            }
        }
    """
    # Pre-compute GloVe embeddings for the whole corpus once
    corpus_glove = np.array([text_to_glove(t, glove_embeddings) for t in texts])
 
    results = {}
 
    for query in queries:
        # ── find the query's index in the corpus (first occurrence) ──────────
        try:
            q_idx = texts.index(query)
        except ValueError:
            q_idx = None  # query not in corpus — handle gracefully
 
        entry = {}
 
        # ── TF-IDF ───────────────────────────────────────────────────────────
        if q_idx is not None:
            tfidf_scores = tfidf_sim[q_idx].copy()
            tfidf_scores[q_idx] = -1.0  # exclude self
        else:
            # query not in corpus: vectorize it on-the-fly
            # (requires vectorizer — not passed, so fall back to zeros)
            tfidf_scores = np.full(len(texts), 0.0)
 
        top3_tfidf = _top_k(texts, tfidf_scores, k=3)
        entry["tfidf"] = top3_tfidf
 
        # ── GloVe ────────────────────────────────────────────────────────────
        q_glove = text_to_glove(query, glove_embeddings).reshape(1, -1)
        glove_scores = sklearn_cosine(q_glove, corpus_glove).flatten()
        if q_idx is not None:
            glove_scores[q_idx] = -1.0
        top3_glove = _top_k(texts, glove_scores, k=3)
        entry["glove"] = top3_glove
 
        # ── BERT ─────────────────────────────────────────────────────────────
        q_bert = extract_bert_embedding(query, bert_tokenizer, bert_model).reshape(1, -1)
        corpus_bert = np.array([
            extract_bert_embedding(t, bert_tokenizer, bert_model)
            for t in texts
        ])
        bert_scores = sklearn_cosine(q_bert, corpus_bert).flatten()
        if q_idx is not None:
            bert_scores[q_idx] = -1.0
        top3_bert = _top_k(texts, bert_scores, k=3)
        entry["bert"] = top3_bert
 
        results[query] = entry
 
    return results
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────
 
def _top_k(texts, scores, k=3):
    """Return the k (text, score) pairs with the highest scores."""
    indices = np.argsort(scores)[::-1][:k]
    return [(texts[i], float(scores[i])) for i in indices]
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModel
 
    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv("data/bbc_news.csv")
    texts = df["text"].tolist()
    print(f"Loaded {len(texts)} texts")
 
    # ── Task 1: TF-IDF ────────────────────────────────────────────────────────
    tfidf_matrix, vectorizer = build_tfidf(texts)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    tfidf_sim = compute_tfidf_similarity(tfidf_matrix)
    print(f"TF-IDF similarity matrix shape: {tfidf_sim.shape}")
 
    # ── Task 2: GloVe ─────────────────────────────────────────────────────────
    glove = load_glove("data/glove_50k_50d.txt")
    print(f"Loaded {len(glove)} GloVe vectors")
    sample_emb = text_to_glove(texts[0], glove)
    print(f"Sample GloVe text embedding shape: {sample_emb.shape}")
 
    # OOV rate (Task 5 analysis helper)
    all_words = " ".join(texts).lower().split()
    oov_count = sum(1 for w in all_words if w not in glove)
    oov_rate = oov_count / len(all_words) if all_words else 0
    print(f"OOV rate: {oov_rate:.2%}  ({oov_count}/{len(all_words)} tokens)")
 
    # ── Task 3: DistilBERT ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    sample_bert = extract_bert_embedding(texts[0], tokenizer, model)
    print(f"Sample BERT embedding shape: {sample_bert.shape}")
 
    # ── Task 4: Compare — one query per category ──────────────────────────────
    queries = [
        df[df["category"] == cat]["text"].iloc[0]
        for cat in df["category"].unique()
    ]
 
    comparison = compare_similarities(
        texts, queries, tfidf_sim, glove, model, tokenizer
    )
 
    # ── Print comparison table ────────────────────────────────────────────────
    for q in comparison:
        print(f"\n{'='*80}")
        print(f"Query: {q[:120]}...")
        print(f"{'─'*80}")
        for method in ["tfidf", "glove", "bert"]:
            print(f"\n  [{method.upper()}]")
            for rank, (text, score) in enumerate(comparison[q][method], 1):
                print(f"    {rank}. (score={score:.4f}) {text[:80]}...")