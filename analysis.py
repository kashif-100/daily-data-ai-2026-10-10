"""
analysis.py — EDA + sentiment classification for synthetic product reviews.

Pipeline:
  1. Load data/reviews.csv (built by generate_data.py)
  2. Exploratory analysis with pandas (ratings, categories, time trends)
  3. TF-IDF + LogisticRegression to predict positive vs negative sentiment
     from review text (neutral 3-star reviews are excluded from training)
  4. Save charts to charts/ and key numbers to findings.json

Run:  python analysis.py
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed — we only save PNGs
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------ load
df = pd.read_csv("data/reviews.csv", parse_dates=["review_date"])
df["month"] = df["review_date"].dt.to_period("M").astype(str)
df["review_len"] = df["review_text"].str.len()

print(f"Loaded {len(df):,} reviews, {df['category'].nunique()} categories")

# --------------------------------------------------------------- EDA
sentiment_share = df["sentiment"].value_counts(normalize=True).round(4)
rating_dist = df["rating"].value_counts().sort_index()
avg_rating_by_cat = df.groupby("category")["rating"].mean().round(2).sort_values()
neg_rate_by_cat = (df.assign(is_neg=(df["sentiment"] == "negative").astype(int))
                     .groupby("category")["is_neg"].mean().round(4).sort_values(ascending=False))
avg_len_by_sent = df.groupby("sentiment")["review_len"].mean().round(1)
monthly = (df.groupby(["month", "sentiment"]).size().unstack(fill_value=0))
monthly_share = monthly.div(monthly.sum(axis=1), axis=0)

worst_category = neg_rate_by_cat.index[0]
best_category = avg_rating_by_cat.index[-1]

print("\nSentiment share:\n", sentiment_share)
print("\nAvg rating by category:\n", avg_rating_by_cat)
print("\nNegative-review rate by category:\n", neg_rate_by_cat)

# ------------------------------------------------- sentiment classifier
# Neutral (3-star) reviews are excluded: mixed language would confuse the
# binary classifier and the goal is clean positive-vs-negative signal.
clf_df = df[df["sentiment"].isin(["positive", "negative"])].copy()
clf_df["label"] = (clf_df["sentiment"] == "positive").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    clf_df["review_text"], clf_df["label"],
    test_size=0.20, random_state=RANDOM_STATE, stratify=clf_df["label"],
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",          # drop noise words like "the", "and"
    ngram_range=(1, 2),            # single words + two-word phrases
    min_df=2,                      # ignore one-off typos
    max_features=5000,
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                           random_state=RANDOM_STATE)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
y_prob = model.predict_proba(X_test_tfidf)[:, 1]

metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1": round(f1_score(y_test, y_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}
print("\nClassifier metrics:", metrics)

# Most influential words/phrases from the model's coefficients
feature_names = np.array(vectorizer.get_feature_names_out())
coefs = model.coef_[0]
top_pos_idx = np.argsort(coefs)[-15:][::-1]
top_neg_idx = np.argsort(coefs)[:15]
top_positive_words = [(feature_names[i], round(float(coefs[i]), 3)) for i in top_pos_idx]
top_negative_words = [(feature_names[i], round(float(coefs[i]), 3)) for i in top_neg_idx]
print("\nTop praise words:", [w for w, _ in top_positive_words])
print("Top complaint words:", [w for w, _ in top_negative_words])

# Most helpful negative reviews = priority complaints for the business
top_complaints = (df[df["sentiment"] == "negative"]
                  .sort_values("helpful_votes", ascending=False)
                  .head(3)[["product_name", "rating", "helpful_votes", "review_text"]])
print("\nTop-voted complaints:")
for _, r in top_complaints.iterrows():
    print(f"  [{r['helpful_votes']} votes] {r['product_name']}: {r['review_text'][:90]}...")

# ------------------------------------------------------------ charts
# 1. Rating distribution stacked by category
fig, ax = plt.subplots(figsize=(10, 5))
pd.crosstab(df["category"], df["rating"]).plot(kind="barh", stacked=True, ax=ax,
                                               cmap="RdYlGn", edgecolor="white")
ax.set_xlabel("Number of reviews")
ax.set_ylabel("")
ax.set_title("Rating distribution by product category")
ax.legend(title="Stars", loc="lower right")
plt.tight_layout()
plt.savefig("charts/rating_by_category.png", dpi=110)
plt.close()

# 2. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Predicted negative", "Predicted positive"],
            yticklabels=["Actual negative", "Actual positive"])
ax.set_title("Sentiment classifier — confusion matrix (test set)")
plt.tight_layout()
plt.savefig("charts/confusion_matrix.png", dpi=110)
plt.close()

# 3. Top positive vs negative n-grams
words = ([w for w, _ in top_negative_words][::-1]
         + [w for w, _ in top_positive_words])
vals = ([v for _, v in top_negative_words][::-1]
        + [v for _, v in top_positive_words])
colors = ["#d62728"] * 15 + ["#2ca02c"] * 15
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(words, vals, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Logistic regression coefficient (word weight)")
ax.set_title("Top complaint words (red) vs top praise words (green)")
plt.tight_layout()
plt.savefig("charts/praise_vs_complaints.png", dpi=110)
plt.close()

# 4. Sentiment trend over time
fig, ax = plt.subplots(figsize=(10, 5))
monthly_share[["positive", "negative"]].plot(ax=ax, marker="o")
ax.set_ylabel("Share of reviews")
ax.set_title("Positive vs negative review share by month")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("charts/sentiment_trend.png", dpi=110)
plt.close()
print("\nCharts saved to charts/")

# ------------------------------------------------------------ findings
findings = {
    "n_reviews": int(len(df)),
    "date_range": [df["review_date"].min().date().isoformat(),
                   df["review_date"].max().date().isoformat()],
    "sentiment_share": sentiment_share.to_dict(),
    "avg_rating_by_category": avg_rating_by_cat.to_dict(),
    "negative_rate_by_category": neg_rate_by_cat.to_dict(),
    "worst_category": worst_category,
    "worst_category_negative_rate": float(neg_rate_by_cat.iloc[0]),
    "best_category": best_category,
    "best_category_avg_rating": float(avg_rating_by_cat.iloc[-1]),
    "avg_review_len_by_sentiment": avg_len_by_sent.to_dict(),
    "monthly_sentiment_share": monthly_share.round(4).to_dict(),
    "classifier_metrics": metrics,
    "top_positive_words": [w for w, _ in top_positive_words],
    "top_negative_words": [w for w, _ in top_negative_words],
    "top_complaints": top_complaints.to_dict(orient="records"),
}
with open("findings.json", "w") as f:
    json.dump(findings, f, indent=2)
print("findings.json written")
