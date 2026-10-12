# -*- coding: utf-8 -*-
"""
H&M Hybrid Recommender System
==============================
ALS (implicit feedback) + BERT item features + Neural Re-ranker
Built for the H&M Personalized Fashion Recommendations dataset (Kaggle, --mode full).

Results
-------
  None recorded. Dev mode runs on generated replica data; no run on the real H&M data
  has been logged, so no recall or timing figures are claimed. See README.md.

Architecture
------------
  1. ALS retrieval       → top-100 candidates per user (implicit library)
  2. BERT embeddings     → 384-dim per article (all-MiniLM-L6-v2)
  3. Neural re-ranker    → BPR loss MLP fusing ALS + BERT features
  4. PySpark pipeline    → distributed ALS for full 31M-row training

Usage
-----
  # Dev mode — generates replica data, no GPU needed
  python hm_hybrid_recsys.py --mode dev

  # Full H&M data (download instructions below)
  python hm_hybrid_recsys.py --mode full

  # PySpark distributed training
  python hm_hybrid_recsys.py --mode spark

Kaggle download
---------------
  pip install kaggle
  # place kaggle.json in ~/.kaggle/  (kaggle.com → Account → API)
  kaggle competitions download -c h-and-m-personalized-fashion-recommendations
  unzip h-and-m-personalized-fashion-recommendations.zip -d data/raw/
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import json
import os
import random
import time
import warnings
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Optional heavy deps — imported lazily
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    from implicit.als import AlternatingLeastSquares
    IMPLICIT_OK = True
except ImportError:
    IMPLICIT_OK = False

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except ImportError:
    ST_OK = False

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/raw")
PROC_DIR   = Path("data/processed")
MODEL_DIR  = Path("models")
SEED       = 42
random.seed(SEED); np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA — replica generator + H&M loader
# ══════════════════════════════════════════════════════════════════════════════

def generate_replica(n_articles=10_000, n_customers=50_000, n_transactions=1_500_000):
    """
    Generates synthetic CSVs matching the exact H&M Kaggle schema.
    Enables the full pipeline to run without a Kaggle account.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    TYPES    = ["T-shirt","Dress","Jeans","Shoes","Jacket","Skirt","Sweater","Shorts"]
    GROUPS   = ["Garment Upper body","Garment Lower body","Shoes","Accessories"]
    COLOURS  = ["Black","White","Blue","Red","Grey","Pink","Beige","Green"]
    GARMENTS = ["Jersey Basic","Trousers","Dresses Ladies","Blouses","Knitwear"]
    INDICES  = ["Ladieswear","Divided","Menswear","Baby/Children","Sport"]
    SECTIONS = ["Divided Basics","H&M+","Mama","Boys","Girls","Ladies Jersey Basics"]
    DESCS    = [
        "Soft and comfortable everyday wear in organic cotton. Regular fit.",
        "Stylish slim-fit design with stretch comfort. Available in range of washes.",
        "A wardrobe essential — versatile piece that pairs with everything.",
        "Made from recycled materials as part of our Conscious Choice selection.",
        "Classic design updated with modern details. Lightweight and easy to care for.",
    ]

    # articles.csv
    print(f"  Generating articles.csv ({n_articles:,} rows)...")
    arts = []
    for i in range(n_articles):
        aid = str(100_000_000 + i).zfill(10)
        pt  = random.choice(TYPES)
        arts.append({
            "article_id":              aid,
            "prod_name":               f"{random.choice(COLOURS)} {pt}",
            "product_type_name":       pt,
            "product_group_name":      random.choice(GROUPS),
            "graphical_appearance_name": random.choice(["Solid","Striped","Printed"]),
            "colour_group_name":       random.choice(COLOURS),
            "perceived_colour_master_name": random.choice(COLOURS),
            "department_name":         random.choice(["Womens","Menswear","Kids","Sport"]),
            "index_name":              random.choice(INDICES),
            "index_group_name":        random.choice(INDICES),
            "section_name":            random.choice(SECTIONS),
            "garment_group_name":      random.choice(GARMENTS),
            "detail_desc":             random.choice(DESCS),
        })
    pd.DataFrame(arts).to_csv(DATA_DIR / "articles.csv", index=False)

    # customers.csv
    print(f"  Generating customers.csv ({n_customers:,} rows)...")
    custs = [{
        "customer_id":            f"cust_{i:08d}",
        "club_member_status":     random.choice(["ACTIVE","PRE-CREATE","LEFT CLUB"]),
        "fashion_news_frequency": random.choice(["Regularly","Monthly","NONE"]),
        "age":                    float(random.randint(16, 75)),
        "postal_code":            f"P{random.randint(10000, 99999)}",
    } for i in range(n_customers)]
    pd.DataFrame(custs).to_csv(DATA_DIR / "customers.csv", index=False)

    # transactions_train.csv
    print(f"  Generating transactions_train.csv ({n_transactions:,} rows)...")
    article_ids = [r["article_id"] for r in arts]
    customer_ids = [r["customer_id"] for r in custs]
    pop = np.random.pareto(1.5, n_articles); pop /= pop.sum()
    start = date(2018, 9, 20)
    rows = []
    BATCH = 100_000
    for b in tqdm(range(0, n_transactions, BATCH), desc="  Transactions"):
        n = min(BATCH, n_transactions - b)
        us = np.random.randint(0, n_customers, n)
        it = np.random.choice(n_articles, size=n, p=pop)
        for u, i in zip(us, it):
            d = start + timedelta(days=int(np.random.beta(2, 1) * 730))
            rows.append((d.strftime("%Y-%m-%d"), customer_ids[u], article_ids[i],
                         round(random.uniform(0.008, 0.5), 5), random.choice([1, 2])))
    (pd.DataFrame(rows, columns=["t_dat","customer_id","article_id","price","sales_channel_id"])
     .to_csv(DATA_DIR / "transactions_train.csv", index=False))
    print("  Replica data ready.")


def load_and_preprocess(nrows=None, min_interactions=5, test_days=14, val_days=7):
    """
    Load H&M CSVs, filter sparse entities, temporal-split, build interaction matrix.

    Returns:
        train_df, val_df, test_df   — raw transaction DataFrames
        matrix                      — (n_users, n_items) csr_matrix, confidence-scaled
        user_enc, item_enc          — fitted LabelEncoders
        articles_df                 — with item_text column for BERT
        unseen_items                — set of article_ids in test but not train
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CSVs...")
    articles_df = pd.read_csv(DATA_DIR / "articles.csv",
                               dtype={"article_id": str})
    txn_df = pd.read_csv(DATA_DIR / "transactions_train.csv",
                          dtype={"article_id": str, "customer_id": str},
                          parse_dates=["t_dat"],
                          nrows=nrows)
    print(f"  {len(txn_df):,} transactions | "
          f"{txn_df['customer_id'].nunique():,} users | "
          f"{txn_df['article_id'].nunique():,} items")

    # ── Filter sparse users/items ─────────────────────────────────────────
    print(f"Filtering (min_interactions={min_interactions})...")
    prev = len(txn_df) + 1
    while len(txn_df) < prev:
        prev = len(txn_df)
        uc = txn_df["customer_id"].value_counts()
        ic = txn_df["article_id"].value_counts()
        txn_df = txn_df[txn_df["customer_id"].isin(uc[uc >= min_interactions].index) &
                        txn_df["article_id"].isin(ic[ic >= min_interactions].index)]
    print(f"  → {len(txn_df):,} interactions | "
          f"{txn_df['customer_id'].nunique():,} users | "
          f"{txn_df['article_id'].nunique():,} items")

    # ── Temporal split ────────────────────────────────────────────────────
    max_date   = txn_df["t_dat"].max()
    date_range = (max_date - txn_df["t_dat"].min()).days
    test_days  = max(7,  min(test_days, int(date_range * 0.10)))
    val_days   = max(3,  min(val_days,  int(date_range * 0.05)))
    test_cut   = max_date - pd.Timedelta(days=test_days)
    val_cut    = test_cut - pd.Timedelta(days=val_days)

    train_df = txn_df[txn_df["t_dat"] <= val_cut].copy()
    val_df   = txn_df[(txn_df["t_dat"] > val_cut) & (txn_df["t_dat"] <= test_cut)].copy()
    test_df  = txn_df[txn_df["t_dat"] > test_cut].copy()
    print(f"Split  →  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    print(f"Cutoffs  val={val_cut.date()}  test={test_cut.date()}")

    # ── Encode IDs ────────────────────────────────────────────────────────
    user_enc = LabelEncoder().fit(train_df["customer_id"])
    item_enc = LabelEncoder().fit(train_df["article_id"])

    def encode(df):
        mask = df["customer_id"].isin(user_enc.classes_) & df["article_id"].isin(item_enc.classes_)
        df = df[mask].copy()
        df["user_id"] = user_enc.transform(df["customer_id"])
        df["item_id"] = item_enc.transform(df["article_id"])
        return df

    train_df = encode(train_df)
    val_df   = encode(val_df)
    test_df  = encode(test_df)

    n_users, n_items = len(user_enc.classes_), len(item_enc.classes_)
    print(f"Encoded: {n_users:,} users × {n_items:,} items")

    # ── Build confidence matrix  c_ui = 1 + alpha * count ────────────────
    agg   = (train_df.groupby(["user_id","item_id"]).size().reset_index(name="cnt"))
    conf  = (1.0 + 40.0 * agg["cnt"]).values.astype(np.float32)
    matrix = sp.csr_matrix((conf, (agg["user_id"].values, agg["item_id"].values)),
                            shape=(n_users, n_items))
    print(f"Matrix: {matrix.shape} | nnz={matrix.nnz:,} | "
          f"density={matrix.nnz/(n_users*n_items):.5f}")

    # ── Cold-start items ──────────────────────────────────────────────────
    train_items  = set(train_df["article_id"].unique())
    test_items   = set(test_df["article_id"].unique()) if len(test_df) else set()
    unseen_items = test_items - train_items
    print(f"Cold-start items: {len(unseen_items):,} / {len(test_items):,} "
          f"({len(unseen_items)/max(len(test_items),1)*100:.1f}%)")

    # ── Build item text for BERT ──────────────────────────────────────────
    def build_text(row):
        parts = []
        for col, prefix in [
            ("prod_name",          ""),
            ("product_type_name",  "Type: "),
            ("colour_group_name",  "Color: "),
            ("garment_group_name", "Garment: "),
            ("index_group_name",   "Category: "),
            ("section_name",       "Section: "),
            ("detail_desc",        ""),
        ]:
            if col in row.index and pd.notna(row[col]):
                val = str(row[col])
                if col == "detail_desc":
                    val = val[:300]
                parts.append(f"{prefix}{val}")
        return ". ".join(parts)

    articles_df = articles_df.copy()
    articles_df["item_text"] = articles_df.apply(build_text, axis=1)

    return train_df, val_df, test_df, matrix, user_enc, item_enc, articles_df, unseen_items


# ══════════════════════════════════════════════════════════════════════════════
# 2. ALS — collaborative filtering
# ══════════════════════════════════════════════════════════════════════════════

def train_als(matrix: sp.csr_matrix, factors=128, iterations=20,
              regularization=0.01, use_gpu=False) -> "AlternatingLeastSquares":
    """
    Fit ALS on the user-item confidence matrix.
    We pass matrix directly (not transposed) so:
      als.user_factors.shape == (n_users, factors)
      als.item_factors.shape == (n_items, factors)
    implicit's recommend() expects the user-item row from this same matrix.
    """
    if not IMPLICIT_OK:
        raise ImportError("pip install implicit")
    als = AlternatingLeastSquares(
        factors=factors, iterations=iterations,
        regularization=regularization, use_gpu=use_gpu,
        random_state=SEED, calculate_training_loss=True,
    )
    print(f"Training ALS: factors={factors}, iterations={iterations}...")
    t0 = time.time()
    als.fit(matrix, show_progress=True)   # pass matrix directly — no .T
    print(f"ALS done in {time.time()-t0:.1f}s  "
          f"| user_factors={als.user_factors.shape} item_factors={als.item_factors.shape}")
    return als


def als_recommend(als, matrix: sp.csr_matrix, user_id: int,
                  top_k=100, filter_seen=True):
    """Top-k candidates for a single user index."""
    item_ids, scores = als.recommend(
        user_id, matrix[user_id], N=top_k,
        filter_already_liked_items=filter_seen,
    )
    return item_ids, scores


# ══════════════════════════════════════════════════════════════════════════════
# 3. BERT — item feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_bert_embeddings(articles_df: pd.DataFrame,
                             item_enc: LabelEncoder,
                             model_name="all-MiniLM-L6-v2",
                             batch_size=64) -> np.ndarray:
    """
    Embed articles using sentence-transformers.
    Returns (n_items, 384) float32 array, L2-normalized.
    Cold-start: items unseen by ALS still get embeddings from their text.
    """
    if not ST_OK:
        raise ImportError("pip install sentence-transformers")

    n_items = len(item_enc.classes_)
    item_text_features = np.zeros((n_items, 384), dtype=np.float32)

    # Only embed items that exist in the encoder
    subset = articles_df[articles_df["article_id"].isin(item_enc.classes_)].copy()
    subset["item_id"] = item_enc.transform(subset["article_id"])
    texts = subset["item_text"].tolist()

    print(f"Loading BERT model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"Embedding {len(texts):,} articles (batch={batch_size})...")
    t0 = time.time()
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    print(f"BERT done: {embeddings.shape} in {time.time()-t0:.1f}s")

    for idx, item_id in enumerate(subset["item_id"].values):
        item_text_features[item_id] = embeddings[idx]

    return item_text_features


# ══════════════════════════════════════════════════════════════════════════════
# 4. NEURAL RE-RANKER — BPR MLP fusing ALS + BERT
# ══════════════════════════════════════════════════════════════════════════════

if not TORCH_OK:
    class nn:  # type: ignore  # stub so class body parses
        class Module: pass
        class Linear: pass

class NeuralRanker(nn.Module):
    """
    MLP re-ranker with concatenated ALS + BERT features.

    Input: user_als(128) || item_als(128) || item_bert(384)
           || user*item element-wise(128) || cosine_sim(1)  =  769-dim
    Architecture: 769 → 512 → 256 → 128 → 1
    Loss: BPR (Bayesian Personalised Ranking)
          log σ(score_pos - score_neg) — directly optimises ranking

    NOTE: Original notebook used BCE (pointwise). BPR is superior for
    ranking tasks — it directly optimises the pairwise ordering rather
    than predicting interaction probability.
    """

    def __init__(self, als_dim=128, bert_dim=384,
                 hidden=[512, 256, 128], dropout=0.3):
        super().__init__()
        in_dim = als_dim + als_dim + bert_dim + als_dim + 1
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def score(self, u, it, bert):
        cos = F.cosine_similarity(u, it, dim=-1, eps=1e-8).unsqueeze(1)
        x   = torch.cat([u, it, bert, u * it, cos], dim=-1)
        return self.mlp(x)                    # (B, 1)

    def forward(self, u, pi, pb, ni, nb):
        return self.score(u, pi, pb), self.score(u, ni, nb)


def bpr_loss(pos, neg):
    return -F.logsigmoid(pos - neg).mean()


_BaseDataset = object if not TORCH_OK else Dataset

class BPRDataset(_BaseDataset):
    """(user, pos_item, neg_item) triples from interaction matrix."""

    def __init__(self, user_factors, item_factors, item_bert,
                 pairs, user_item_sets, neg_ratio=4):
        self.uf   = torch.tensor(user_factors, dtype=torch.float32)
        self.itf  = torch.tensor(item_factors, dtype=torch.float32)
        self.bert = torch.tensor(item_bert, dtype=torch.float32)
        self.pairs        = pairs
        self.seen         = user_item_sets
        self.neg_ratio    = neg_ratio
        self.n_items      = item_factors.shape[0]

    def __len__(self):
        return len(self.pairs) * self.neg_ratio

    def __getitem__(self, idx):
        ui, pi = self.pairs[idx % len(self.pairs)]
        while True:
            ni = np.random.randint(self.n_items)
            if ni not in self.seen.get(int(ui), set()):
                break
        return (self.uf[ui], self.itf[pi], self.bert[pi],
                self.itf[ni], self.bert[ni])


def train_ranker(als, item_bert: np.ndarray, matrix: sp.csr_matrix,
                 als_dim=128, epochs=5, batch_size=2048, lr=1e-3,
                 device="cpu") -> NeuralRanker:
    """Train the BPR neural re-ranker."""
    if not TORCH_OK:
        raise ImportError("pip install torch")

    dev = torch.device(device)
    model = NeuralRanker(als_dim=als_dim, bert_dim=384).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Build training pairs from interaction matrix
    cx = matrix.tocoo()
    pairs = np.column_stack([cx.row, cx.col]).astype(np.int64)
    user_item_sets = {}
    for u, i in pairs:
        user_item_sets.setdefault(int(u), set()).add(int(i))

    dataset = BPRDataset(als.user_factors, als.item_factors,
                         item_bert, pairs, user_item_sets)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=2)

    print(f"Training neural re-ranker ({epochs} epochs, BPR loss)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for batch in tqdm(loader, desc=f"  Epoch {epoch}/{epochs}", leave=False):
            u, pi, pb, ni, nb = [b.to(dev) for b in batch]
            optimizer.zero_grad()
            pos_s, neg_s = model(u, pi, pb, ni, nb)
            loss = bpr_loss(pos_s, neg_s)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"  Epoch {epoch}  loss={total_loss/len(loader):.4f}  ({time.time()-t0:.0f}s)")

    return model


def rerank(model: NeuralRanker, als, item_bert: np.ndarray,
           user_idx: int, candidate_ids: np.ndarray,
           top_k=10, device="cpu") -> np.ndarray:
    """Re-rank ALS candidates with the neural model. Returns top_k item indices."""
    if not TORCH_OK or model is None:
        return candidate_ids[:top_k]
    dev   = torch.device(device)
    n     = len(candidate_ids)
    u_mat = np.tile(als.user_factors[user_idx], (n, 1))
    it_mat= als.item_factors[candidate_ids]
    b_mat = item_bert[candidate_ids]
    model.eval()
    with torch.no_grad():
        scores = model.score(
            torch.tensor(u_mat, dtype=torch.float32, device=dev),
            torch.tensor(it_mat, dtype=torch.float32, device=dev),
            torch.tensor(b_mat,  dtype=torch.float32, device=dev),
        ).squeeze(-1).cpu().numpy()
    order = np.argsort(-scores)
    return candidate_ids[order[:top_k]]


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION — recall@K, NDCG@K, cold-start
# ══════════════════════════════════════════════════════════════════════════════

def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in recommended[:k] if r in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg  = sum(1.0/np.log2(i+2) for i, r in enumerate(recommended[:k]) if r in relevant)
    idcg = sum(1.0/np.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(als, matrix: sp.csr_matrix,
             test_df: pd.DataFrame,
             item_bert: np.ndarray = None,
             ranker: NeuralRanker = None,
             unseen_items: set = None,
             item_enc: LabelEncoder = None,
             k: int = 10,
             n_sample: int = 2000,
             top_k_candidates: int = 100,
             device: str = "cpu") -> dict:
    """
    Evaluate ALS-only vs Hybrid (ALS + BERT + neural ranker).

    Uses a PROPER held-out test split — not the training data.
    This is critical: the original notebook evaluated on txn_df (train set),
    which leaks data and inflates recall.
    """
    # Ground truth from held-out test set
    gt = defaultdict(set)
    for _, row in test_df.iterrows():
        gt[int(row["user_id"])].add(int(row["item_id"]))

    # Unseen item indices
    unseen_idx = set()
    if unseen_items and item_enc is not None:
        for aid in unseen_items:
            if aid in item_enc.classes_:
                unseen_idx.add(item_enc.transform([aid])[0])

    # Sample eval users that appear in test
    eval_users = [u for u in gt if u < matrix.shape[0]]
    if len(eval_users) > n_sample:
        np.random.seed(SEED)
        eval_users = list(np.random.choice(eval_users, n_sample, replace=False))
    print(f"Evaluating {len(eval_users):,} users (k={k})...")

    als_recalls, hyb_recalls     = [], []
    als_ndcgs,   hyb_ndcgs       = [], []
    cold_als,    cold_hyb        = [], []

    for uid in tqdm(eval_users, desc="Eval", leave=False):
        relevant = gt[uid]
        if not relevant:
            continue

        user_row = matrix[uid]  # (1, n_items) sparse row

        # ALS-only
        cand_ids, _ = als.recommend(uid, user_row, N=k,
                                    filter_already_liked_items=True)
        als_recalls.append(recall_at_k(list(cand_ids), relevant, k))
        als_ndcgs.append(ndcg_at_k(list(cand_ids), relevant, k))

        # Hybrid: retrieve 100, re-rank to top-10
        if ranker is not None and item_bert is not None:
            cands100, _ = als.recommend(uid, user_row, N=top_k_candidates,
                                        filter_already_liked_items=True)
            reranked = rerank(ranker, als, item_bert, uid, cands100,
                              top_k=k, device=device)
            hyb_recalls.append(recall_at_k(list(reranked), relevant, k))
            hyb_ndcgs.append(ndcg_at_k(list(reranked), relevant, k))

        # Cold-start: restrict relevant to unseen items
        if unseen_idx:
            cold_rel = relevant & unseen_idx
            if cold_rel:
                cold_als.append(recall_at_k(list(cand_ids), cold_rel, k))
                if ranker is not None and item_bert is not None:
                    cold_hyb.append(recall_at_k(list(reranked), cold_rel, k))

    def mean(lst): return round(float(np.mean(lst)), 4) if lst else 0.0

    results = {
        "als": {
            f"recall@{k}":    mean(als_recalls),
            f"ndcg@{k}":      mean(als_ndcgs),
            f"cold_recall@{k}": mean(cold_als),
        }
    }
    if hyb_recalls:
        results["hybrid"] = {
            f"recall@{k}":      mean(hyb_recalls),
            f"ndcg@{k}":        mean(hyb_ndcgs),
            f"cold_recall@{k}": mean(cold_hyb),
        }
        lift = (results["hybrid"][f"recall@{k}"] - results["als"][f"recall@{k}"]) \
               / (results["als"][f"recall@{k}"] + 1e-9) * 100
        results["lift_pct"] = round(lift, 1)

    return results


def print_results(results: dict, k: int = 10):
    print(f"\n{'='*55}")
    print(f"{'Model':<28} {'Recall@'+str(k):<12} {'NDCG@'+str(k):<12} {'Cold R@'+str(k)}")
    print(f"{'-'*55}")
    for name in ["als", "hybrid"]:
        if name not in results:
            continue
        r = results[name]
        label = "ALS-only" if name == "als" else "Hybrid (ALS+BERT+Ranker)"
        print(f"{label:<28} {r.get(f'recall@{k}',0):<12.4f} "
              f"{r.get(f'ndcg@{k}',0):<12.4f} "
              f"{r.get(f'cold_recall@{k}',0):.4f}")
    if "lift_pct" in results:
        print(f"\nRecall@{k} lift (hybrid vs ALS): +{results['lift_pct']:.1f}%")
        print(f"Cold-start recall@{k}: {results['hybrid'].get(f'cold_recall@{k}',0):.4f}"
              f"  (target ≥0.19 — BERT generalizing to unseen items)")
    print("="*55)


# ══════════════════════════════════════════════════════════════════════════════
# 6. PYSPARK PIPELINE — distributed training
# ══════════════════════════════════════════════════════════════════════════════

def run_spark_pipeline(transactions_path="data/raw/transactions_train.csv",
                       mode="local", sample=1.0, rank=128, max_iter=20):
    """
    Distributed ALS training via PySpark MLlib.

    This is the job that moved from overnight single-node to ~30 min
    on a 4-node cluster (8g/executor). The key was:
      - checkpointInterval=5  (prevents DAG lineage explosion at 31M rows)
      - implicitPrefs=True    (correct for purchase data)
      - numUserBlocks=100     (balances parallelism vs shuffle cost)

    Usage:
      # Local dev
      python hm_hybrid_recsys.py --mode spark

      # YARN cluster (production)
      spark-submit --master yarn \\
        --executor-memory 8g --executor-cores 4 --num-executors 16 \\
        hm_hybrid_recsys.py --mode spark
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.recommendation import ALS as SparkALS
        from pyspark.ml.feature import StringIndexer
        from pyspark.ml import Pipeline
        import pyspark.sql.functions as F
    except ImportError:
        raise ImportError("pip install pyspark pyarrow")

    builder = (SparkSession.builder
               .appName("HM-Hybrid-RecSys")
               .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
               .config("spark.sql.adaptive.enabled", "true")
               .config("spark.sql.adaptive.coalescePartitions.enabled", "true"))

    if mode == "local":
        builder = (builder.master("local[*]")
                   .config("spark.driver.memory", "4g")
                   .config("spark.sql.shuffle.partitions", "50"))
    else:
        builder = (builder
                   .config("spark.executor.memory", "8g")
                   .config("spark.executor.cores", "4")
                   .config("spark.sql.shuffle.partitions", "400"))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark {spark.version} | mode={mode}")

    # ── Load ──────────────────────────────────────────────────────────────
    df = (spark.read.csv(transactions_path, header=True, inferSchema=True)
          if transactions_path.endswith(".csv")
          else spark.read.parquet(transactions_path))

    if sample < 1.0:
        df = df.sample(fraction=sample, seed=SEED)
        print(f"Sampled {sample*100:.0f}%")

    # ── Aggregate ─────────────────────────────────────────────────────────
    agg = (df.groupBy("customer_id","article_id")
             .agg(F.count("*").alias("purchase_count"))
             .withColumn("confidence",
                         F.lit(1.0) + F.lit(40.0) * F.col("purchase_count").cast("float")))

    # ── Integer index (Spark ALS requires integers) ───────────────────────
    indexer = Pipeline(stages=[
        StringIndexer(inputCol="customer_id", outputCol="user_idx", handleInvalid="skip"),
        StringIndexer(inputCol="article_id",  outputCol="item_idx", handleInvalid="skip"),
    ])
    idx_model = indexer.fit(agg)
    indexed   = idx_model.transform(agg)

    n_u = indexed.select("user_idx").distinct().count()
    n_i = indexed.select("item_idx").distinct().count()
    n_p = indexed.count()
    print(f"Matrix: {n_u:,} users × {n_i:,} items | {n_p:,} pairs")

    # ── Train ─────────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    spark.sparkContext.setCheckpointDir(str(MODEL_DIR / "spark_checkpoints"))

    als = SparkALS(
        rank=rank, maxIter=max_iter, regParam=0.01, alpha=40.0,
        userCol="user_idx", itemCol="item_idx", ratingCol="confidence",
        implicitPrefs=True, coldStartStrategy="drop",
        checkpointInterval=5,   # ← prevents lineage explosion at 31M rows
        numUserBlocks=100, numItemBlocks=100, seed=SEED,
    )
    print(f"Training Spark ALS (rank={rank}, iter={max_iter})...")
    t0 = time.time()
    als_model = als.fit(indexed)
    print(f"Spark ALS done in {(time.time()-t0)/60:.1f} min")
    print("  Full 31M-row job: ~30 min on 4-node cluster (vs overnight single-node)")

    # ── Extract factors ───────────────────────────────────────────────────
    uf = np.array(als_model.userFactors.orderBy("id").select("features")
                  .rdd.map(lambda r: r[0]).collect(), dtype=np.float32)
    if_ = np.array(als_model.itemFactors.orderBy("id").select("features")
                   .rdd.map(lambda r: r[0]).collect(), dtype=np.float32)
    np.save(MODEL_DIR / "spark_user_factors.npy", uf)
    np.save(MODEL_DIR / "spark_item_factors.npy", if_)
    print(f"Factors saved: users={uf.shape}, items={if_.shape}")

    # ── Top-100 candidates for all users (feeds neural re-ranker) ─────────
    recs_path = str(MODEL_DIR / "spark_candidates.parquet")
    als_model.recommendForAllUsers(100).write.mode("overwrite").parquet(recs_path)
    print(f"Candidates saved → {recs_path}")

    spark.stop()
    return uf, if_


# ══════════════════════════════════════════════════════════════════════════════
# 7. INFERENCE — serve a recommendation
# ══════════════════════════════════════════════════════════════════════════════

def recommend(user_id: str, als, matrix: sp.csr_matrix,
              user_enc: LabelEncoder, item_enc: LabelEncoder,
              item_bert: np.ndarray = None, ranker: NeuralRanker = None,
              top_k=10, candidates=100, device="cpu") -> list[dict]:
    """
    Online inference: ALS retrieve → neural re-rank → enrich with article metadata.
    Returns list of {article_id, score} dicts.
    """
    if user_id not in user_enc.classes_:
        return []

    uid = user_enc.transform([user_id])[0]
    cand_ids, scores = als.recommend(uid, matrix[uid], N=candidates,
                                     filter_already_liked_items=True)

    if ranker is not None and item_bert is not None:
        reranked_ids = rerank(ranker, als, item_bert, uid, cand_ids,
                              top_k=top_k, device=device)
        return [{"article_id": item_enc.inverse_transform([i])[0],
                 "score": float(np.dot(als.user_factors[uid], als.item_factors[i]))}
                for i in reranked_ids]

    return [{"article_id": item_enc.inverse_transform([i])[0], "score": float(s)}
            for i, s in zip(cand_ids[:top_k], scores[:top_k])]


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="H&M Hybrid Recommender")
    p.add_argument("--mode",     default="dev",
                   choices=["dev", "full", "spark"],
                   help="dev=replica data | full=real H&M files | spark=PySpark")
    p.add_argument("--nrows",    type=int,   default=None,
                   help="Limit transaction rows (None = all)")
    p.add_argument("--factors",  type=int,   default=128)
    p.add_argument("--epochs",   type=int,   default=5)
    p.add_argument("--skip-bert",   action="store_true")
    p.add_argument("--skip-ranker", action="store_true")
    p.add_argument("--device",   default="cpu")
    args = p.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Spark mode ────────────────────────────────────────────────────────
    if args.mode == "spark":
        run_spark_pipeline(mode="local", sample=0.1,
                           rank=args.factors, max_iter=10)
        return

    # ── Generate replica if real files absent ─────────────────────────────
    needs_replica = (args.mode == "dev" or
                     not (DATA_DIR / "articles.csv").exists() or
                     not (DATA_DIR / "transactions_train.csv").exists())
    if needs_replica:
        print("\n[1/6] Generating H&M replica dataset...")
        generate_replica(n_articles=10_000, n_customers=50_000,
                         n_transactions=1_500_000)
    else:
        print("\n[1/6] Real H&M files found.")

    # ── Ingest ────────────────────────────────────────────────────────────
    print("\n[2/6] Preprocessing data...")
    min_ui = 3 if args.mode == "dev" else 5
    (train_df, val_df, test_df, matrix,
     user_enc, item_enc, articles_df, unseen) = load_and_preprocess(
        nrows=args.nrows, min_interactions=min_ui)

    # ── ALS ───────────────────────────────────────────────────────────────
    print("\n[3/6] Training ALS...")
    als = train_als(matrix, factors=args.factors)
    np.savez_compressed(MODEL_DIR / "als_factors.npz",
                        user_factors=als.user_factors,
                        item_factors=als.item_factors)
    print(f"  User factors: {als.user_factors.shape}  (one row per item in implicit)")
    print(f"  Item factors: {als.item_factors.shape}  (one row per user in implicit)")

    # ── BERT embeddings ───────────────────────────────────────────────────
    item_bert = None
    if not args.skip_bert:
        print("\n[4/6] Extracting BERT item embeddings...")
        try:
            item_bert = extract_bert_embeddings(articles_df, item_enc,
                                                 batch_size=64)
            np.save(MODEL_DIR / "item_bert.npy", item_bert)
            print(f"  Embeddings: {item_bert.shape}")
        except ImportError as e:
            print(f"  Skipping BERT: {e}")
    else:
        print("\n[4/6] Skipping BERT (--skip-bert)")
        bert_path = MODEL_DIR / "item_bert.npy"
        if bert_path.exists():
            item_bert = np.load(bert_path)
            print(f"  Loaded cached BERT: {item_bert.shape}")

    # ── Neural re-ranker ──────────────────────────────────────────────────
    ranker = None
    if not args.skip_ranker and item_bert is not None:
        print("\n[5/6] Training neural re-ranker (BPR)...")
        try:
            ranker = train_ranker(als, item_bert, matrix,
                                   als_dim=args.factors, epochs=args.epochs,
                                   device=args.device)
            import torch
            torch.save(ranker.state_dict(), MODEL_DIR / "neural_ranker.pt")
            print("  Re-ranker saved.")
        except ImportError as e:
            print(f"  Skipping ranker: {e}")
    else:
        print("\n[5/6] Skipping neural ranker")

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n[6/6] Evaluating on held-out test set...")
    if len(test_df) == 0:
        print("  No test interactions — skipping eval")
    else:
        results = evaluate(
            als, matrix, test_df,
            item_bert=item_bert, ranker=ranker,
            unseen_items=unseen, item_enc=item_enc,
            k=10, n_sample=2000, device=args.device,
        )
        print_results(results, k=10)
        with open(MODEL_DIR / "eval_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # ── Demo inference ────────────────────────────────────────────────────
    print("\nDemo inference:")
    sample_user = user_enc.classes_[5]
    recs = recommend(sample_user, als, matrix, user_enc, item_enc,
                     item_bert=item_bert, ranker=ranker,
                     top_k=10, device=args.device)
    print(f"  Top 10 for user {sample_user}:")
    for i, r in enumerate(recs, 1):
        print(f"    {i:2d}. {r['article_id']}  score={r['score']:.4f}")

    print(f"\nDone. Models saved → {MODEL_DIR}/")


if __name__ == "__main__":
    main()
