import os

from tqdm import tqdm

from attacks import attack_cases, load_attacks
from benchmarks import load_benchmarks
from metrics import score
from models import ask, load_model, model_specs
from utils import load_config, write_json, write_jsonl


def build_cases(config):
    """Every benchmark's sampled items, each followed by its attacked copies."""
    attacks = load_attacks(config)
    return {name: [case for item in items for case in attack_cases(item, name, attacks)]
            for name, items in load_benchmarks(config).items()}


def evaluate(model, spec, benchmark, cases, results_dir):
    results = [(case, ask(model, case)) for case in tqdm(cases, desc=f"{spec['name']} {benchmark}")]
    rows = [{"model": spec["name"], "model_id": spec["id"], "benchmark": benchmark, **row} for row in score(results, benchmark)]
    out = os.path.join(results_dir, spec["name"])
    write_jsonl(os.path.join(out, f"{benchmark}_answers.jsonl"),
                [{"id": case["id"], "variant": case["variant"], "answers": answer} for case, answer in results])
    write_json(os.path.join(out, f"{benchmark}.json"), rows)
    return rows


def run(config, specs=None):
    """Run every model on the benchmarks it is allowed, one model in memory at a time."""
    cases = build_cases(config)
    rows = []
    for spec in specs or model_specs(config):
        names = [name for name in cases if name in spec.get("benchmarks", cases)]
        if not names:
            continue
        model = load_model(spec, config["run"])
        try:
            for name in names:
                rows += evaluate(model, spec, name, cases[name], config["run"]["results_dir"])
        finally:
            model.close()
    return rows


def main(config_path="config.yml"):
    run(load_config(config_path))


if __name__ == "__main__":
    main()
