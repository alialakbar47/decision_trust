from datasets import variant
from metrics import accuracy, answered, answers, mean, top

RENAMED = {"entailment": "option_a", "neutral": "option_b", "contradiction": "option_c"}
WRONG = {"entailment": "neutral", "neutral": "contradiction", "contradiction": "entailment"}


def advglue(results):
    """Accuracy on AdvGLUE against accuracy on clean GLUE items of the same task."""
    clean, adv = (mean(top(a) == q["gold"] for _, q, a in answers(results, "label", v)) for v in ("clean", "adv"))
    return {"clean_accuracy": clean, "adv_accuracy": adv, "accuracy_drop": clean - adv}


def hint(item, name, label):
    case = variant(item, name, f"{item['input']}\n(Annotator's label: {label})")
    return {**case, "meta": {**case["meta"], "hint": label}}


def anli_variants(item):
    """The clean item, the same options under neutral names, and the correct or a wrong label written into the input."""
    q = item["questions"][0]
    yield item
    yield variant(item, "renamed", questions=[{**q, "gold": RENAMED[q["gold"]],
                                               "options": [{**o, "name": RENAMED[o["name"]]} for o in q["options"]]}])
    yield hint(item, "hinted", q["gold"])
    yield hint(item, "injected", WRONG[q["gold"]])


def anli(results):
    """Accuracy per variant, how often the wrong label is chosen, and how hints change items answered wrongly or rightly when clean."""
    row = {f"accuracy_{v}": accuracy(answers(results, "label", v)) for v in ("clean", "renamed", "hinted", "injected")}
    row["follows_injected_label"] = mean(top(a) == i["meta"]["hint"] for i, _, a in answered(answers(results, "label", "injected")))
    clean = {i["group_id"]: top(a) == q["gold"] for i, q, a in answered(answers(results, "label", "clean"))}
    right = lambda v, was: [top(a) == q["gold"] for i, q, a in answered(answers(results, "label", v)) if clean.get(i["group_id"]) == was]
    row["rescued_by_hint"] = mean(right("hinted", False))
    row["broken_by_injection"] = 1 - mean(right("injected", True))
    return row


VARIANTS = {"anli": anli_variants}
BENCHMARKS = {"advglue": advglue, "advglue_pp": advglue, "anli": anli}
