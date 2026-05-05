"""
stretch_embedding_explorer.py
Module 6 Week B — Stretch: Embedding Space Explorer

Visualizes GloVe word embeddings and DistilBERT document embeddings
in 2D using t-SNE dimensionality reduction.

Requirements:
    pip install numpy matplotlib scikit-learn transformers torch gensim requests tqdm
"""

import os
import csv
import random
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 150

# ─────────────────────────────────────────────
# 0.  PATHS / CONFIG
# ─────────────────────────────────────────────
GLOVE_FILE = "data/glove_50k_50d.txt"   # الملف اللي اشتغلنا عليه
BBC_CSV    = "data/bbc_news.csv"
OUTPUT_DIR = "stretch_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# 1.  WORD CATEGORIES  (200 words, ≥5 categories)
# ─────────────────────────────────────────────
WORD_CATEGORIES = {
    "Countries": [
        "france", "germany", "japan", "china", "india",
        "brazil", "australia", "canada", "russia", "italy",
        "spain", "mexico", "argentina", "egypt", "nigeria",
        "sweden", "norway", "denmark", "finland", "poland",
        "turkey", "iran", "iraq", "israel", "pakistan",
        "indonesia", "malaysia", "thailand", "vietnam", "kenya",
        "ghana", "ethiopia", "morocco", "portugal", "greece",
        "austria", "switzerland", "belgium", "netherlands", "chile",
    ],
    "Sports": [
        "football", "basketball", "tennis", "cricket", "golf",
        "swimming", "athletics", "rugby", "hockey", "baseball",
        "volleyball", "boxing", "wrestling", "cycling", "skiing",
        "soccer", "marathon", "tournament", "championship", "league",
        "coach", "stadium", "athlete", "referee", "medal",
        "olympics", "penalty", "goalkeeper", "striker", "dribble",
        "serve", "forehand", "backhand", "wicket", "innings",
        "quarterback", "touchdown", "homerun", "slam", "sprint",
    ],
    "Technology": [
        "computer", "software", "internet", "algorithm", "database",
        "network", "server", "mobile", "smartphone", "laptop",
        "processor", "memory", "bandwidth", "encryption", "protocol",
        "browser", "search", "cloud", "artificial", "intelligence",
        "robot", "automation", "semiconductor", "transistor", "pixel",
        "resolution", "interface", "programming", "compiler", "debug",
        "startup", "silicon", "wireless", "bluetooth", "satellite",
        "cybersecurity", "hacker", "virus", "firewall", "bandwidth",
    ],
    "Finance": [
        "bank", "stock", "market", "investment", "profit",
        "revenue", "currency", "inflation", "recession", "gdp",
        "trade", "export", "import", "dividend", "portfolio",
        "equity", "bond", "interest", "loan", "mortgage",
        "commodity", "futures", "hedge", "capital", "asset",
        "liability", "budget", "deficit", "surplus", "tax",
        "audit", "accounting", "insurance", "broker", "nasdaq",
        "dow", "economy", "fiscal", "monetary", "treasury",
    ],
    "Emotions": [
        "happy", "sad", "angry", "fear", "joy",
        "love", "hate", "anxiety", "excitement", "grief",
        "pride", "shame", "guilt", "envy", "jealousy",
        "hope", "despair", "trust", "disgust", "surprise",
        "calm", "stress", "euphoria", "melancholy", "nostalgia",
        "loneliness", "empathy", "compassion", "courage", "confidence",
        "frustration", "disappointment", "relief", "satisfaction", "boredom",
        "enthusiasm", "passion", "rage", "delight", "sorrow",
    ],
}

# Flatten to list preserving category labels
ALL_WORDS  = []
ALL_LABELS = []
for cat, words in WORD_CATEGORIES.items():
    for w in words:
        ALL_WORDS.append(w)
        ALL_LABELS.append(cat)

assert len(ALL_WORDS) == 200, f"Expected 200 words, got {len(ALL_WORDS)}"

# ─────────────────────────────────────────────
# 2.  LOAD GloVe
# ─────────────────────────────────────────────

def load_glove_vectors(words):
    """
    Return a dict {word: np.array(50)} for each word in `words`
    that exists in the GloVe vocabulary.
    Words not found are skipped (a warning is printed).
    """
    if not os.path.exists(GLOVE_FILE):
        raise FileNotFoundError(
            f"ما لقيت ملف GloVe في: {GLOVE_FILE}\n"
            "تأكد إن data/glove_50k_50d.txt موجود في مجلد الـ repo."
        )
    target = set(words)
    found  = {}
    print(f"[GloVe] Scanning {GLOVE_FILE} …")
    with open(GLOVE_FILE, encoding="utf-8") as f:
        for line in tqdm(f, total=50_000):
            parts = line.rstrip().split(" ")
            w = parts[0]
            if w in target:
                found[w] = np.array(parts[1:], dtype=np.float32)
                if len(found) == len(target):
                    break
    missing = target - found.keys()
    if missing:
        print(f"[GloVe] Warning — {len(missing)} words not found: {missing}")
    return found


# ─────────────────────────────────────────────
# 3.  LOAD BBC NEWS  (20 articles, ≥3 per category)
# ─────────────────────────────────────────────

BBC_CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]
ARTICLES_PER_CAT = 4   # 4 × 5 = 20 total

def load_bbc_articles(csv_path: str, n_per_cat: int = 4):
    """
    Read bbc_news.csv and return exactly n_per_cat articles per category.
    Expected columns: category, text   (or similar — adapted below).
    """
    articles = {cat: [] for cat in BBC_CATEGORIES}

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = [c.lower().strip() for c in reader.fieldnames]

        # Flexible column detection
        cat_col  = next((c for c in reader.fieldnames if c.lower().strip() in
                         ("category", "label", "class", "topic")), reader.fieldnames[0])
        text_col = next((c for c in reader.fieldnames if c.lower().strip() in
                         ("text", "body", "article", "content", "description")), reader.fieldnames[1])

        for row in reader:
            cat = row[cat_col].strip().lower()
            if cat in articles and len(articles[cat]) < n_per_cat:
                text = row[text_col].strip()
                if len(text) > 50:
                    articles[cat].append({"category": cat, "text": text})

    flat = [art for cat_arts in articles.values() for art in cat_arts]
    for cat, arts in articles.items():
        print(f"  [{cat}] loaded {len(arts)} articles")
    assert all(len(v) >= 3 for v in articles.values()), (
        "Some categories have fewer than 3 articles — "
        "check your bbc_news.csv path and column names."
    )
    return flat


# ─────────────────────────────────────────────
# 4.  DistilBERT EMBEDDINGS
# ─────────────────────────────────────────────

def compute_distilbert_embeddings(texts):
    """
    Return (N, 768) mean-pooled CLS embeddings via DistilBERT.
    Falls back gracefully if GPU unavailable.
    """
    from transformers import AutoTokenizer, AutoModel
    import torch

    print("[DistilBERT] Loading model …")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model     = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    embeddings = []
    for text in tqdm(texts, desc="[DistilBERT] Encoding"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling over token dimension
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(emb)

    return np.array(embeddings)


# ─────────────────────────────────────────────
# 5.  DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────

def reduce_to_2d(vectors: np.ndarray, method: str = "tsne", perplexity: float = 30):
    """
    Reduce (N, D) → (N, 2).
    method: 'tsne' | 'pca'
    """
    print(f"[Reduction] Applying {method.upper()} to {vectors.shape} …")
    if method == "tsne":
        import sklearn
        tsne_kwargs = dict(
            n_components=2,
            perplexity=perplexity,
            random_state=RANDOM_SEED,
            init="pca",
            learning_rate="auto",
        )
        # n_iter renamed to max_iter in sklearn 1.5+
        sk_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        if sk_version >= (1, 5):
            tsne_kwargs["max_iter"] = 1000
        else:
            tsne_kwargs["n_iter"] = 1000
        reducer = TSNE(**tsne_kwargs)
    else:
        reducer = PCA(n_components=2, random_state=RANDOM_SEED)
    return reducer.fit_transform(vectors)


# ─────────────────────────────────────────────
# 6.  PLOTTING HELPERS
# ─────────────────────────────────────────────

CATEGORY_COLORS = {
    # Word categories
    "Countries":   "#e63946",
    "Sports":      "#2a9d8f",
    "Technology":  "#457b9d",
    "Finance":     "#f4a261",
    "Emotions":    "#8338ec",
    # BBC categories
    "business":     "#e63946",
    "entertainment":"#f4a261",
    "politics":     "#2a9d8f",
    "sport":        "#457b9d",
    "tech":         "#8338ec",
}

# Words to annotate (≥10 required)
ANNOTATE_WORDS = {
    # Countries
    "france", "china", "brazil", "australia", "nigeria",
    # Sports
    "olympics", "championship", "marathon",
    # Technology
    "algorithm", "artificial", "smartphone",
    # Finance
    "inflation", "recession", "nasdaq",
    # Emotions
    "euphoria", "nostalgia", "melancholy",
}


def plot_word_embeddings(coords_2d, words, labels, out_path):
    fig, ax = plt.subplots(figsize=(16, 12))

    for cat in WORD_CATEGORIES:
        mask  = [i for i, l in enumerate(labels) if l == cat]
        xs    = coords_2d[mask, 0]
        ys    = coords_2d[mask, 1]
        color = CATEGORY_COLORS[cat]
        ax.scatter(xs, ys, c=color, s=55, alpha=0.75,
                   edgecolors="white", linewidths=0.4, zorder=3)

    # Annotations
    for i, word in enumerate(words):
        if word in ANNOTATE_WORDS:
            ax.annotate(
                word,
                (coords_2d[i, 0], coords_2d[i, 1]),
                fontsize=7.5,
                fontweight="bold",
                xytext=(4, 4),
                textcoords="offset points",
                color="#1d1d1d",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )

    legend_handles = [
        mpatches.Patch(color=CATEGORY_COLORS[c], label=c)
        for c in WORD_CATEGORIES
    ]
    ax.legend(handles=legend_handles, title="Semantic Category",
              fontsize=9, title_fontsize=10, loc="best",
              framealpha=0.9, edgecolor="#cccccc")

    ax.set_title(
        "GloVe 50d Word Embeddings — t-SNE Projection (200 words)",
        fontsize=14, fontweight="bold", pad=14
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[Plot] Saved → {out_path}")
    plt.close(fig)


def plot_document_embeddings(coords_2d, articles, out_path):
    fig, ax = plt.subplots(figsize=(14, 10))

    cats = [a["category"] for a in articles]
    for cat in BBC_CATEGORIES:
        mask  = [i for i, c in enumerate(cats) if c == cat]
        xs    = coords_2d[mask, 0]
        ys    = coords_2d[mask, 1]
        color = CATEGORY_COLORS[cat]
        ax.scatter(xs, ys, c=color, s=110, alpha=0.85,
                   edgecolors="white", linewidths=0.6,
                   label=cat.capitalize(), zorder=3)

    # Annotate every article with short label
    for i, art in enumerate(articles):
        short = art["text"][:32].replace("\n", " ") + "…"
        ax.annotate(
            short,
            (coords_2d[i, 0], coords_2d[i, 1]),
            fontsize=5.8,
            xytext=(6, 6),
            textcoords="offset points",
            color="#2d2d2d",
        )

    ax.legend(title="BBC Category", fontsize=9, title_fontsize=10,
              loc="best", framealpha=0.9, edgecolor="#cccccc")
    ax.set_title(
        "DistilBERT Document Embeddings — t-SNE Projection (20 BBC Articles)",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[Plot] Saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Embedding Space Explorer — Module 6 Week B Stretch")
    print("=" * 60)

    # ── 7A. GloVe word embeddings ──────────────────────────────
    print("\n[1/4] Loading GloVe vectors …")
    glove = load_glove_vectors(ALL_WORDS)

    # Build ordered arrays, dropping missing words
    valid_words  = [w for w in ALL_WORDS  if w in glove]
    valid_labels = [ALL_LABELS[ALL_WORDS.index(w)] for w in valid_words]
    word_matrix  = np.stack([glove[w] for w in valid_words])
    print(f"      Using {len(valid_words)}/200 words  (shape: {word_matrix.shape})")

    # ── 7B. t-SNE on GloVe ─────────────────────────────────────
    print("\n[2/4] Reducing word vectors to 2D with t-SNE …")
    word_2d = reduce_to_2d(word_matrix, method="tsne", perplexity=30)

    word_plot_path = os.path.join(OUTPUT_DIR, "word_embedding_visualization.png")
    plot_word_embeddings(word_2d, valid_words, valid_labels, word_plot_path)

    # ── 7C. BBC articles + DistilBERT ─────────────────────────
    print("\n[3/4] Loading BBC News articles …")
    bbc_csv = BBC_CSV if os.path.exists(BBC_CSV) else _find_bbc_csv()
    articles = load_bbc_articles(bbc_csv, n_per_cat=ARTICLES_PER_CAT)

    texts      = [a["text"] for a in articles]
    doc_embeds = compute_distilbert_embeddings(texts)
    print(f"      Document embeddings shape: {doc_embeds.shape}")

    # ── 7D. t-SNE on documents ─────────────────────────────────
    print("\n[4/4] Reducing document vectors to 2D with t-SNE …")
    # Perplexity must be < n_samples; 5 is safe for 20 docs
    doc_2d = reduce_to_2d(doc_embeds, method="tsne", perplexity=5)

    doc_plot_path = os.path.join(OUTPUT_DIR, "document_embedding_visualization.png")
    plot_document_embeddings(doc_2d, articles, doc_plot_path)

    print("\n✓ All done!  Outputs saved to:", OUTPUT_DIR)
    print("  ├─", word_plot_path)
    print("  └─", doc_plot_path)


def _find_bbc_csv():
    """Search common locations for bbc_news.csv."""
    candidates = [
        "bbc_news.csv",
        "data/bbc_news.csv",
        "datasets/bbc_news.csv",
        "../data/bbc_news.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Cannot find bbc_news.csv.  "
        "Place it at data/bbc_news.csv or set BBC_CSV at the top of this script."
    )


if __name__ == "__main__":
    main()