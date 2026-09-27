# Reading the results

After `python src/server.py` finishes, `models/metrics.csv` has one row per FL
round: `round, accuracy, f1_macro, precision_macro, recall_macro` (weighted
average across all clients' held-out test sets).

Quick plot:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("models/metrics.csv")
plt.plot(df["round"], df["accuracy"], label="accuracy")
plt.plot(df["round"], df["f1_macro"], label="f1 (macro)")
plt.xlabel("FL round")
plt.legend()
plt.title("Federated global model performance over rounds")
plt.savefig("models/training_curve.png")
```

## What to report in a writeup

1. **Convergence curve**: accuracy/F1 vs. round (above). Expect it to be noisier
   and converge slower than the centralized baseline — that's the cost of both
   federation (non-IID clients) and DP noise.
2. **Privacy-utility tradeoff**: re-run `src/server.py` with a few different
   `--dp_noise_multiplier` values (e.g. 0.0/no-DP, 0.5, 1.0, 2.0) and plot final
   accuracy vs. noise multiplier. This is the single most important chart for
   this kind of project — it shows you understand DP isn't free.
3. **Federated vs. centralized gap**: compare `models/metrics.csv`'s final round
   against `src/centralized_baseline.py`'s printed accuracy. A small gap says
   your FL setup is working well; a large gap suggests trying more local epochs,
   a better strategy (FedProx), or more rounds.
4. **Per-attack-class performance**: extend `compute_metrics` in `src/utils.py`
   to also return a full `sklearn.metrics.classification_report` (per-class
   precision/recall) — useful to show, e.g., that DDoS is easy to catch but
   brute-force is harder under high non-IID skew.
5. **Estimated epsilon**: run `python src/privacy.py` (or call
   `estimate_epsilon` with your actual round count / noise multiplier) and
   report it alongside your accuracy numbers, with the caveat noted in that
   file that it's a loose upper bound, not a tight accounting.
