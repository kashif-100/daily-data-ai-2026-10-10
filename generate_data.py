"""
generate_data.py — synthetic product-review dataset for the
Day 3 "Review Sentiment Analyzer" project.

Creates 1,800 reviews across 5 product categories, dated
2026-04-01 .. 2026-10-09, with a fixed random seed (42) so the
dataset is identical on every run. No external data or APIs.

Columns: review_id, review_date, category, product_name, rating
(1-5), sentiment (positive/neutral/negative, derived from rating),
verified_purchase, helpful_votes, review_text
"""

import random
import csv
from datetime import date, timedelta

SEED = 42
N_REVIEWS = 1800
START_DATE = date(2026, 4, 1)
END_DATE = date(2026, 10, 9)

random.seed(SEED)

# ---------------------------------------------------------------- products
CATEGORIES = {
    "Smartphone": ["Nova X1", "Nova X1 Pro", "Orbit Lite"],
    "Laptop": ["AeroBook 14", "AeroBook 15 Pro"],
    "Headphones": ["SoundPulse 500", "SoundPulse Buds Air"],
    "Running Shoes": ["FleetRun Air", "FleetRun Trail"],
    "Coffee Maker": ["BrewMaster 12", "BrewMaster Espresso"],
}

# ------------------------------------------------------------ phrase banks
# Reviews are assembled from fragments so the text reads naturally but
# stays clearly tied to the star rating (with a little noise, like reality).
POSITIVE = [
    "Absolutely love it",
    "Works perfectly",
    "Exceeded my expectations",
    "Great value for money",
    "Very happy with this purchase",
    "Highly recommend",
    "Superb quality",
    "Does exactly what it promises",
    "Arrived quickly and well packed",
    "The build quality is excellent",
    "Easy to use right out of the box",
    "I use it every day and it has not disappointed",
    "Five stars, no complaints",
    "Better than I expected for the price",
    "A solid product all round",
]
NEGATIVE = [
    "Very disappointed",
    "Stopped working within a month",
    "Poor build quality",
    "Not worth the price",
    "I want a refund",
    "Terrible experience",
    "Would not recommend to anyone",
    "Arrived damaged and the return process was a headache",
    "Broke after a few days of normal use",
    "Does not match the description at all",
    "Waste of money",
    "Customer support was unhelpful",
    "One star is too generous",
    "Frustrating to use",
    "Defective on arrival",
]
MILD_POSITIVE = [
    "Pretty good overall",
    "Does the job well",
    "Happy enough with it",
    "Good for the price",
    "Works as described",
]
MILD_NEGATIVE = [
    "Could be better",
    "Not quite what I hoped for",
    "Average at best",
    "Has a few flaws",
    "Decent, but overpriced",
]
ASPECTS = {
    "Smartphone": ["battery life", "camera", "screen", "charging speed", "fingerprint sensor"],
    "Laptop": ["battery life", "keyboard", "screen", "fan noise", "trackpad"],
    "Headphones": ["sound quality", "noise cancellation", "comfort", "bass", "microphone"],
    "Running Shoes": ["cushioning", "grip", "fit", "weight", "durability"],
    "Coffee Maker": ["brew quality", "noise", "cleaning", "water tank", "heating speed"],
}
POSITIVE_ASPECT = [
    "the {a} is fantastic",
    "really impressed with the {a}",
    "the {a} alone is worth the price",
    "no complaints about the {a}",
]
NEGATIVE_ASPECT = [
    "the {a} is terrible",
    "biggest issue is the {a}",
    "the {a} failed within weeks",
    "very unhappy with the {a}",
]


def build_review(rating, category):
    """Assemble review text that (mostly) matches the star rating."""
    aspect = random.choice(ASPECTS[category])
    parts = []
    if rating == 5:
        parts = [random.choice(POSITIVE), random.choice(POSITIVE)]
        if random.random() < 0.6:
            parts.append(random.choice(POSITIVE_ASPECT).format(a=aspect))
        # ~3% noise: a 5-star review with a gripe (happens in real data)
        if random.random() < 0.03:
            parts.append(random.choice(MILD_NEGATIVE).lower())
    elif rating == 4:
        parts = [random.choice(POSITIVE)]
        if random.random() < 0.5:
            parts.append(random.choice(POSITIVE_ASPECT).format(a=aspect))
        if random.random() < 0.4:
            parts.append(", although " + random.choice(MILD_NEGATIVE).lower())
        if random.random() < 0.15:
            parts.append(random.choice(NEGATIVE_ASPECT).format(a=aspect))  # real flaw in a 4-star review
    elif rating == 3:
        parts = [random.choice(MILD_POSITIVE), random.choice(MILD_NEGATIVE)]
    elif rating == 2:
        parts = [random.choice(NEGATIVE)]
        if random.random() < 0.4:
            parts.append(random.choice(NEGATIVE_ASPECT).format(a=aspect))
        if random.random() < 0.5:
            # mixed 2-star review: genuinely liked one thing
            parts.append(", although " + random.choice(POSITIVE_ASPECT).format(a=aspect).lower())
    else:  # rating == 1
        parts = [random.choice(NEGATIVE), random.choice(NEGATIVE)]
        if random.random() < 0.6:
            parts.append(random.choice(NEGATIVE_ASPECT).format(a=aspect))
        # ~3% noise: a 1-star review that still praises one thing
        if random.random() < 0.03:
            parts.append(", that said, " + random.choice(POSITIVE_ASPECT).format(a=aspect).lower())
    sentences = []
    for p in parts:
        p = p.rstrip(".")
        if p.startswith(", "):
            sentences[-1] = sentences[-1] + p  # connector clause, same sentence
        else:
            sentences.append(p[0].upper() + p[1:])
    text = ". ".join(sentences) + "."

    # Realistic noise: a small share of reviews contains one sentence that
    # contradicts the star rating (e.g. a 1-star review praising fast
    # delivery, or a 5-star review complaining about packaging). This keeps
    # the classifier honest instead of perfectly separable.
    if rating >= 4 and random.random() < 0.09:
        text += " That said, " + random.choice(NEGATIVE_ASPECT).format(
            a=random.choice(ASPECTS[category])).lower() + "."
    elif rating <= 2 and random.random() < 0.09:
        text += " On the plus side, " + random.choice(POSITIVE_ASPECT).format(
            a=random.choice(ASPECTS[category])).lower() + "."
    return text


def main():
    total_days = (END_DATE - START_DATE).days
    # Roughly even spread across categories, ratings skewed positive
    # (like a real storefront) but with a heavy long tail of bad reviews.
    rating_weights = {5: 0.38, 4: 0.27, 3: 0.12, 2: 0.11, 1: 0.12}
    ratings = list(rating_weights)
    weights = list(rating_weights.values())

    rows = []
    for i in range(1, N_REVIEWS + 1):
        category = random.choice(list(CATEGORIES))
        product = random.choice(CATEGORIES[category])
        rating = random.choices(ratings, weights=weights, k=1)[0]
        sentiment = "positive" if rating >= 4 else ("negative" if rating <= 2 else "neutral")
        review_date = START_DATE + timedelta(days=random.randint(0, total_days))
        verified = random.random() < 0.82
        # Helpful votes: better reviews collect more votes, with noise
        helpful = int(random.expovariate(1 / (3 + rating * 2)))
        rows.append({
            "review_id": f"R{i:05d}",
            "review_date": review_date.isoformat(),
            "category": category,
            "product_name": product,
            "rating": rating,
            "sentiment": sentiment,
            "verified_purchase": verified,
            "helpful_votes": helpful,
            "review_text": build_review(rating, category),
        })

    rows.sort(key=lambda r: r["review_date"])
    out_path = "data/reviews.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["sentiment"] == "positive")
    n_neg = sum(1 for r in rows if r["sentiment"] == "negative")
    n_neu = sum(1 for r in rows if r["sentiment"] == "neutral")
    print(f"Wrote {len(rows)} reviews -> {out_path}")
    print(f"positive={n_pos} ({n_pos/len(rows):.1%})  "
          f"neutral={n_neu} ({n_neu/len(rows):.1%})  "
          f"negative={n_neg} ({n_neg/len(rows):.1%})")


if __name__ == "__main__":
    main()
