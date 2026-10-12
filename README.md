# Hybrid Fashion Recommender: ALS + Text Embeddings + Neural Re-ranker

A two-stage recommender designed for the H&M Personalized Fashion Recommendations dataset (31M purchases):
ALS on implicit feedback retrieves 100 candidates per customer, then an MLP re-ranker trained with BPR loss fuses
the ALS factors with `all-MiniLM-L6-v2` embeddings of each article's text, so new articles with no purchase
history can still be recommended. A PySpark path distributes ALS for the full dataset.

## Status

Code only; **no results are claimed.** Earlier versions stated "+22% Recall@10 over ALS" and "19% Recall@10 on
cold items", but no run on the real H&M data was recorded. The data sits behind a Kaggle login, and `--mode dev`
generates replica data whose structure the code itself defines, so its scores say nothing about real performance.

| Mode | Data | Status |
|---|---|---|
| `--mode dev` | Generated replica of the H&M schema | Runs locally, no GPU |
| `--mode full` | Real H&M data from Kaggle | Not yet run |
| `--mode spark` | Real H&M data, distributed ALS | Not yet run |

The evaluation code (`recall_at_k`, `ndcg_at_k`, cold-start split) splits train, validation and test by date (the test window is the most recent 7+ days) to avoid the
leakage a random split would cause. To produce real numbers: download the competition data, run `--mode full`, and
commit the metrics file it writes.

```
pip install numpy pandas scipy scikit-learn implicit sentence-transformers torch tqdm
python hm_hybrid_recsys.py --mode dev
```
