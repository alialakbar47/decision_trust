from collections import Counter

import numpy as np


def top(answer):
    return max(answer["probs"], key=answer["probs"].get) if answer["status"] == "ok" else None


def p(answer, name):
    return answer["probs"].get(name, np.nan) if answer["status"] == "ok" else np.nan


def withheld(answer):
    """The API refused or blocked the question instead of answering it."""
    return answer["status"] in ("refused", "blocked")


def refuses(answer, name="refuse"):
    return np.nan if withheld(answer) else top(answer) == name


def mean(values):
    values = [float(v) for v in values if not np.isnan(float(v))]
    return float(np.mean(values)) if values else np.nan


def auroc(scores, labels):
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    keep = ~np.isnan(scores)
    pos, neg = scores[keep & labels], scores[keep & ~labels]
    if not len(pos) or not len(neg):
        return np.nan
    return float((pos[:, None] > neg).mean() + 0.5 * (pos[:, None] == neg).mean())


def ece(confidences, correct, n_bins=10):
    confidences, correct = np.asarray(confidences, dtype=float), np.asarray(correct, dtype=float)
    bins = np.minimum((confidences * n_bins).astype(int), n_bins - 1)
    return float(sum((bins == b).mean() * abs(confidences[bins == b].mean() - correct[bins == b].mean())
                     for b in np.unique(bins))) if len(bins) else np.nan


def answers(results, question_id, variant=None):
    """(item, question, answer) for one question, optionally in one variant."""
    return [(item, q, ans[q["id"]]) for item, ans in results for q in item["questions"]
            if q["id"] == question_id and variant in (None, item["variant"])]


def answered(rows):
    return [(item, q, a) for item, q, a in rows if a["status"] == "ok"]


def accuracy(rows):
    return mean(top(a) == q["gold"] for _, q, a in answered(rows))


def twins(results, question_id, variant):
    """(clean answer, variant answer, clean item, variant item) linked by group_id."""
    clean = {item["group_id"]: (ans[question_id], item) for item, ans in results if item["variant"] == "clean"}
    return [(clean[item["group_id"]][0], ans[question_id], clean[item["group_id"]][1], item)
            for item, ans in results if item["variant"] == variant and item["group_id"] in clean]


def drop_nan(row):
    return {k: v for k, v in row.items() if not (isinstance(v, float) and np.isnan(v))}


def summary(results):
    """Accuracy and calibration over answered questions with a gold answer, plus the rate of each status."""
    rows = [(q, ans[q["id"]]) for item, ans in results for q in item["questions"] if q.get("gold") is not None]
    ok = [(q, a) for q, a in rows if a["status"] == "ok"]
    confidence, correct = [max(a["probs"].values()) for _, a in ok], [top(a) == q["gold"] for q, a in ok]
    statuses = Counter(a["status"] for _, a in rows)
    return {"n": len(rows), "accuracy": mean(correct), "ece": ece(confidence, correct),
            "confidence_auroc": auroc(confidence, correct), **{f"{s}_rate": n / len(rows) for s, n in statuses.items()}}
