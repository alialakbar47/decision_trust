import os

from utils import matches, read_jsonl

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
BENCHMARKS = {
    "agent_abstain": {"topic": "agent safety", "paper": "Liu et al., AgentAbstain, NeurIPS 2026 Evaluations & Datasets"},
}


def load_dataset(name, where=None):
    """Every item of a benchmark in the unified form, optionally filtered on meta fields."""
    return [item for item in read_jsonl(os.path.join(DATASETS_DIR, f"{name}.jsonl")) if matches(item, where)]


def text(item):
    """The policy followed by the input, which attacks may have changed."""
    return "\n\n".join(part for part in (item["policy"], item["input"]) if part)


def options(q):
    return {o["name"]: o.get("description", o["name"]) for o in q["options"]}


def typesafe(item, spec):
    questions = {q["id"]: {"type": "noul", "instructions": q["text"]} if q["type"] == "yesno" else
                 {"type": q["type"], "instructions": q["text"],
                  "criteria": list(options(q).values()) if q["type"] == "score" else options(q)} for q in item["questions"]}
    return {"model": spec.get("request_id", spec["id"]), "state": text(item), "questions": questions}


def google(item, spec):
    request = typesafe(item, spec)
    request["input"] = request.pop("state")
    return request


def openai(item, spec):
    questions = []
    for q in item["questions"]:
        question = {"type": "predicate" if q["type"] == "yesno" else q["type"], "name": q["id"], "instructions": q["text"]}
        if q["type"] == "choice":
            question["choices"] = [{"value": name, "description": d} for name, d in options(q).items()]
        if q["type"] == "score":
            question["levels"] = [{"label": name, "description": d} for name, d in options(q).items()]
        questions.append(question)
    return {"model": spec["id"], "input": text(item), "questions": questions}


FORMATS = {"typesafe": typesafe, "google": google, "openai": openai}


def load_benchmark(items, spec):
    """The items, each with its request in the target model's format, ready to be sent."""
    return [{**item, "request": FORMATS[spec["format"]](item, spec)} for item in items]
