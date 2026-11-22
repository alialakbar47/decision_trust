import os
import random

from utils import matches, read_jsonl

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BENCHMARKS = {
    "xstest": {"topic": "safety", "paper": "Röttger et al., XSTest, NAACL 2024"},
    "bbq": {"topic": "fairness", "paper": "Parrish et al., BBQ, Findings of ACL 2022"},
    "advglue_pp": {"topic": "robustness", "paper": "Wang et al., DecodingTrust (AdvGLUE++), NeurIPS 2023"},
}


def sample(groups, n, seed):
    return groups if n is None or n >= len(groups) else random.Random(seed).sample(groups, n)


def load_benchmark(name, max_examples=None, per_subset=None, where=None, seed=0):
    """Items of one benchmark, filtered on meta fields and sampled by group so linked variants stay together."""
    groups = {}
    for item in read_jsonl(os.path.join(DATA_DIR, f"{name}.jsonl")):
        if matches(item, where):
            groups.setdefault(item["group_id"], []).append(item)
    groups = list(groups.values())
    if per_subset:
        subsets = {}
        for group in groups:
            subsets.setdefault(group[0]["meta"]["subset"], []).append(group)
        groups = [g for subset, part in sorted(subsets.items()) for g in sample(part, per_subset, f"{seed}-{subset}")]
    return [item for group in sample(groups, max_examples, seed) for item in group]


def load_benchmarks(config):
    return {name: load_benchmark(name, **(config["benchmarks"].get(name) or {})) for name in config["run"]["benchmarks"]}
