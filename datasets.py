import os
import json
import random

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "data_processed")
YESNO = [{"name": "yes"}, {"name": "no"}]


def options(question):
    return question.get("options") or YESNO


def reverse_options(question):
    return {**question, "options": question["options"][::-1]}


def variant(item, name, text=None, questions=None):
    """A copy of an item under a new variant name, sharing its group_id."""
    return {**item, "id": f"{item['group_id']}-{name}", "variant": name, "input": text or item["input"],
            "questions": questions or item["questions"]}


def get_dataset(dataset_id, max_examples=None):
    path = os.path.join(DATA_DIR, f"{dataset_id}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found at '{path}'")
    with open(path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    return random.Random(0).sample(items, min(max_examples, len(items))) if max_examples else items


def get_loader(dataset_id, model, variants=None, max_examples=None):
    """Yield each item, and its variants if given, with the request in the model's own format."""
    for item in get_dataset(dataset_id, max_examples):
        for case in variants(item) if variants else [item]:
            yield case, model.render(case)
