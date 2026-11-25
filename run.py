import os

from models import test_model
from utils import load_config, write_json, write_jsonl


def run(config):
    """Test every model of the run on every benchmark, clean and under the attacks that target each benchmark."""
    settings, rows = config["run"], []
    for model in settings["models"]:
        try:
            model_rows, answers = test_model(model, settings["benchmarks"], settings["attacks"], config)
        except Exception as e:
            print(f"{model} FAILED: {type(e).__name__}: {e}")
            continue
        write_json(os.path.join(settings["results_dir"], f"{model}.json"), model_rows)
        write_jsonl(os.path.join(settings["results_dir"], f"{model}_answers.jsonl"), answers)
        rows += model_rows
    return rows


def main(config_path="config.yml"):
    run(load_config(config_path))


if __name__ == "__main__":
    main()
