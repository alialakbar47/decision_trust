from metrics import answers, mean, top


def advglue(results):
    """Accuracy on AdvGLUE against accuracy on clean GLUE items of the same task."""
    clean, adv = (mean(top(a) == q["gold"] for _, q, a in answers(results, "label", v)) for v in ("clean", "adv"))
    return {"clean_accuracy": clean, "adv_accuracy": adv, "accuracy_drop": clean - adv}


VARIANTS = {}
BENCHMARKS = {"advglue": advglue}
