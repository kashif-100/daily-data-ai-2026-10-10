# Review Sentiment Analyzer

**Day 3 of the 15-day daily data-AI series** — synthetic product reviews,
pandas EDA, TF-IDF + LogisticRegression sentiment classifier, and an
auto-generated AI narrative of praise vs complaints.

## What it does

1. **Generates** 1,800 synthetic product reviews (`generate_data.py`) across
   5 categories and 11 products, dated 2026-04-01 → 2026-10-09, with a fixed
   random seed so the dataset is reproducible.
2. **Explores** the data with pandas: sentiment share, rating distribution by
   category, negative-review rate by category, review length, monthly trends.
3. **Trains** a TF-IDF + LogisticRegression classifier to predict positive vs
   negative sentiment from review text (neutral 3-star reviews excluded).
4. **Generates** a plain-English insights report (`ai_insights.py` →
   `AI_INSIGHTS.md`) — the "AI integrated" part of this project.

## Key findings

- **Sentiment split:** 64.6% positive, 24.4% negative, 10.9% neutral
  (3-star) out of 1,800 reviews.
- **Worst category: Running Shoes** — 26.2% of its reviews are negative.
  **Best: Laptop** — 3.73/5 average (Coffee Maker is essentially tied at
  3.73/5).
- **Classifier performance** (TF-IDF unigrams+bigrams, LogisticRegression,
  80/20 stratified split): accuracy **99.4%**, precision **99.1%**,
  recall **100%**, F1 **0.996**, ROC-AUC **1.00** on 321 held-out reviews.
- **Top praise signals:** "highly recommend", "excellent", "superb",
  "fantastic" — shoppers praise build quality, ease of use, fast delivery,
  and value for money.
- **Top complaint signals:** "poor build", "frustrating", "waste of money",
  "terrible", "defective" — the recurring pattern is products arriving
  defective or breaking early, plus unhelpful support.
- Negative reviewers write the longest reviews (avg 73 chars) — unhappy
  customers explain themselves; neutral 3-star reviews are the shortest
  (avg 41 chars).

## Charts

| File | What it shows |
|---|---|
| `charts/rating_by_category.png` | 1–5 star distribution per category |
| `charts/confusion_matrix.png` | Classifier errors on the test set |
| `charts/praise_vs_complaints.png` | Top model-weighted words, red vs green |
| `charts/sentiment_trend.png` | Positive vs negative share by month |

## How to run

```bash
pip install -r requirements.txt
python generate_data.py   # builds data/reviews.csv (seed 42, reproducible)
python analysis.py        # EDA + trains classifier -> charts/ + findings.json
python ai_insights.py     # plain-English report -> AI_INSIGHTS.md
```

## Limitations (read before citing the metrics)

- **The data is synthetic.** Reviews are assembled from phrase banks, so the
  vocabulary is cleaner and more separable than real review text. The ~99%
  classifier accuracy reflects that — on real Amazon-style reviews the same
  approach typically lands around 85–92%. See `INTERVIEW_NOTES.md` for an
  honest discussion.
- Sentiment labels come from star ratings (4–5 = positive, 1–2 = negative),
  not from human annotation, so sarcasm and mixed feelings are only
  approximated.
- The dataset is a storefront snapshot (Apr–Oct 2026); October covers only
  9 days, so its monthly share should not be over-read.
