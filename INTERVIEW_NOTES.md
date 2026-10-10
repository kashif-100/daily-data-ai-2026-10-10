# INTERVIEW_NOTES.md — Review Sentiment Analyzer (Day 3)

Interview Q&A prep for THIS project. Every answer is grounded in what the
code actually does — no bluffing.

## 30-second pitch

"I built a review sentiment analyzer over 1,800 synthetic product reviews
across 5 categories. It does pandas EDA on ratings and trends, then trains
a TF-IDF plus LogisticRegression classifier that predicts positive vs
negative sentiment from review text at 99.4% accuracy on a held-out test
set. The headline findings: 24% of reviews are negative, Running Shoes is
the worst category at 26% negative, and the top complaint themes are poor
build quality and defective arrivals. An auto-generated report turns all
of that into a plain-English action plan."

## Project walkthrough (end-to-end)

**Q: Walk me through the pipeline.**
`generate_data.py` builds the dataset (1,800 reviews, seed 42, so it's
reproducible), `analysis.py` loads it with pandas, runs the exploratory
analysis, trains the classifier on an 80/20 stratified split, and writes
four chart PNGs plus `findings.json`. `ai_insights.py` reads
`findings.json` and renders `AI_INSIGHTS.md`, a plain-English narrative
with recommendations. Everything runs with three commands; the README
documents the order.

**Q: Why synthetic data instead of real reviews?**
Two reasons: this is a portfolio series where every project must be
reproducible offline with a fixed seed, and real review datasets carry
licensing and PII concerns. The trade-off is honesty — I document in the
README that the generator's phrase banks make the data cleaner and more
separable than real text, so the ~99% accuracy should not be quoted as a
real-world result.

## Technical deep-dives

**Q: How does TF-IDF turn text into numbers?**
TF-IDF scores each word or two-word phrase by how often it appears in a
review (term frequency) discounted by how common it is across all reviews
(inverse document frequency). So "the" gets a near-zero weight while
"defective arrival" — rare overall but frequent in bad reviews — gets a
high one. I used unigrams plus bigrams (`ngram_range=(1,2)`), English
stop-words removed, `min_df=2` to drop one-off typos, capped at 5,000
features.

**Q: Why LogisticRegression for text?**
It's the honest baseline for text classification: fast, interpretable, and
strong on high-dimensional sparse TF-IDF vectors. The learned coefficients
are directly readable — a large positive coefficient on "highly recommend"
means that phrase pushes the prediction toward positive. Limitation: it's
linear, so it can't capture interactions or sarcasm ("great, another
broken one") the way a transformer could.

**Q: Why is your accuracy 99.4%? Isn't that suspicious?**
Yes, and I'd say so in the interview. The number is high because the
synthetic generator draws from distinct positive and negative phrase
banks, so the classes are nearly linearly separable by construction. The
real achievement is the pipeline, not the score — on genuine Amazon-style
reviews the identical approach typically lands around 85–92%. I kept the
metric honest by reporting precision, recall, F1, and ROC-AUC alongside
accuracy rather than cherry-picking.

**Q: Why did you exclude neutral 3-star reviews from training?**
Their language is mixed ("pretty good overall, but overpriced"), which
blurs the binary decision boundary without adding signal for a
positive-vs-negative task. They're still used in all the EDA — sentiment
share, trends, review length — just not in the classifier. In production
I'd either build a 3-class model or route neutrals to a separate "mixed
feelings" queue.

**Q: How do you know the model isn't overfitting?**
The metrics come from a held-out 20% test set (321 reviews) the model
never saw during training, with the split stratified so the class ratio
matches. The confusion matrix shows only 2 misclassifications. Honest
caveat: with template-generated text, train and test share the same phrase
banks, so this test measures generalization within the generator's
vocabulary, not to genuinely new language.

**Q: What do the top-weighted words tell you?**
They're a free topic model. Positive weights cluster on praise vocabulary
("highly recommend", "excellent", "superb") and negative weights on
complaint vocabulary ("poor build", "waste of money", "frustrating").
Business value: these are the exact phrases to watch in incoming reviews,
and they feed the praise-vs-complaints section of the auto-report.

## Data-analyst thinking

**Q: What KPIs did you define?**
Sentiment share (64.6% positive / 24.4% negative / 10.9% neutral),
average rating by category, negative-review rate by category, average
review length by sentiment, and monthly positive-share trend. The
category-level negative rate is the operational KPI — it tells support
where to look first.

**Q: What was the single most important insight?**
Nearly one review in four is negative and they concentrate in Running
Shoes (26.2%) — plus the complaint vocabulary points at a single root
cause: products arriving defective or breaking within weeks. That's not a
marketing problem, it's a quality-control problem, and the classifier
exists to surface those reviews to support within minutes instead of
days.

**Q: How did analysis become recommendations?**
Each recommendation traces to a finding: fix Running Shoes first (highest
negative rate) → auto-route negative reviews to support (classifier at
99%+ precision makes this safe) → reply publicly to the most-voted
complaints (social proof that issues get fixed) → protect the praise
drivers (fast, well-packed delivery). The report lists them in priority
order, not as a grab bag.

## The "AI integrated" part

**Q: What does "AI integrated" mean in this project?**
Two things. First, the LogisticRegression classifier is a machine-learning
model that reads review text and scores sentiment — it can auto-triage
incoming reviews without a human reading each one. Second, `ai_insights.py`
is a deterministic narrative generator: it converts the model's outputs
and the EDA statistics into a plain-English report with prioritized
actions, so the analysis is consumable by a non-technical store manager.

**Q: Why not just call an LLM API for the insights?**
An LLM would write a prettier report but could hallucinate numbers and
needs an API key, cost, and network. The template-based generator can only
state numbers that came from `findings.json`, so every claim is
verifiable. Trade-off: it's rigid — if a finding doesn't fit a template
slot, it won't be expressed as well as an LLM would. For a portfolio
project, verifiability beats fluency.

## With more time I would…

- Pull a real dataset (e.g., Amazon reviews) and re-run the same pipeline
  to report honest real-world metrics.
- Add a 3-class model (or a regression on star rating) so neutral reviews
  are predicted, not discarded.
- Benchmark against a stronger baseline like a fine-tuned transformer or
  at least Naive Bayes, and compare calibration, not just accuracy.
- Build the "review router" properly: a small service that scores new
  reviews and files support tickets for high-confidence negatives.
- Track aspect-level sentiment (battery vs screen vs delivery) instead of
  whole-review sentiment, which is what a store actually acts on.
