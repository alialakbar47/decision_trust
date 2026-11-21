from datasets import variant
from metrics import accuracy, answered, answers, auroc, mean, p, top, twins

A, B = "\n\nResponse A:\n", "\n\nResponse B:\n"
OTHER = {"response_a": "response_b", "response_b": "response_a"}


def pair_variants(item):
    """The pair as given, then with the two responses exchanged."""
    head, rest = item["input"].split(A)
    a, b = rest.split(B)
    q = item["questions"][0]
    yield item
    yield variant(item, "swapped", f"{head}{A}{b}{B}{a}", [{**q, "gold": OTHER[q["gold"]]}])


def pairwise(results):
    """Accuracy in each order, agreement across orders, and accuracy when both orders are right."""
    pairs = [(a, b, i["questions"][0]["gold"]) for a, b, i, _ in twins(results, "better", "swapped") if a["status"] == b["status"] == "ok"]
    return {"accuracy_clean": accuracy(answers(results, "better", "clean")),
            "accuracy_swapped": accuracy(answers(results, "better", "swapped")),
            "order_consistency": mean(OTHER[top(a)] == top(b) for a, b, _ in pairs),
            "accuracy_both_orders": mean(top(a) == gold and top(b) == OTHER[gold] for a, b, gold in pairs)}


def healthbench(results):
    """Agreement with unanimous physician grades: macro F1 over met and not met, AUROC of P(met), and the met rate."""
    rows = answered(answers(results, "criterion"))
    f1 = []
    for c in ("yes", "no"):
        tp = sum(top(a) == q["gold"] == c for _, q, a in rows)
        wrong = sum((top(a) == c) != (q["gold"] == c) for _, q, a in rows)
        f1.append(2 * tp / (2 * tp + wrong) if tp + wrong else float("nan"))
    return {"macro_f1": mean(f1), "auroc_met": auroc([p(a, "yes") for _, _, a in rows], [q["gold"] == "yes" for _, q, _ in rows]),
            "predicted_met_rate": mean(top(a) == "yes" for _, _, a in rows),
            "gold_met_rate": mean(q["gold"] == "yes" for _, q, _ in rows)}


def choice(results):
    """Accuracy against the chance level of each question."""
    rows = [(q, ans[q["id"]]) for item, ans in results for q in item["questions"]]
    return {"chance": mean(1 / len(q["options"]) for q, _ in rows)}


def processbench(results):
    """Accuracy on solutions with an error and on correct ones, and their harmonic mean (the paper's F1)."""
    rows = answered(answers(results, "first_error"))
    wrong = mean(top(a) == q["gold"] for _, q, a in rows if q["gold"] != "no_error")
    right = mean(top(a) == q["gold"] for _, q, a in rows if q["gold"] == "no_error")
    return {"accuracy_erroneous": wrong, "accuracy_correct": right, "f1": 2 * wrong * right / (wrong + right) if wrong + right else float("nan")}


VARIANTS = {"llmbar": pair_variants, "judgebench": pair_variants}
BENCHMARKS = {"healthbench": healthbench, "llmbar": pairwise, "judgebench": pairwise, "rewardbench2": choice,
              "processbench": processbench, "prefeval": choice}
