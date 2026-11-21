import os
import json

import yaml
from tqdm import tqdm

import fairness, judging, robustness, safety
from datasets import get_loader
from metrics import drop_nan, summary
from models import get_model

EVALUATIONS = {"safety": safety, "robustness": robustness, "fairness": fairness, "judging": judging}


def get_answers(model, dataset_id, variants=None, max_examples=None):
    loader = get_loader(dataset_id, model, variants, max_examples)
    return [(item, model.ask(item, request)) for item, request in tqdm(loader, desc=dataset_id)]


def evaluate(model, model_id, evaluation, dataset_ids, checkpoint_path, max_examples=None):
    rows = []
    for dataset_id in dataset_ids:
        results = get_answers(model, dataset_id, EVALUATIONS[evaluation].VARIANTS.get(dataset_id), max_examples)
        subsets = {"all": results}
        for item, answer in results:
            if "subset" in item["meta"]:
                subsets.setdefault(item["meta"]["subset"], []).append((item, answer))
        for subset, part in subsets.items():
            rows.append(drop_nan({"model_id": model_id, "evaluation": evaluation, "dataset_id": dataset_id, "subset": subset,
                                  **summary(part), **EVALUATIONS[evaluation].BENCHMARKS[dataset_id](part)}))
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(rows, f, indent=4)
    return rows


def main(config_path="config.yml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for spec in config["models"]:
        if spec["id"] is None:
            continue
        model = get_model(spec, config["cache_dir"], config["device"])
        for evaluation, dataset_ids in config["evaluations"].items():
            checkpoint_path = os.path.join(config["results_dir"], spec["name"], f"{evaluation}.json")
            evaluate(model, spec["id"], evaluation, dataset_ids, checkpoint_path, config["max_examples"])


if __name__ == "__main__":
    main()
