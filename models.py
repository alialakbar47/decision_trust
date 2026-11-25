import os
import json
import time
import subprocess

import requests
from tqdm import tqdm

from attacks import apply_attacks, load_attacks
from benchmarks import load_benchmark, load_dataset
from metrics import score
from utils import RateLimit, cache_path, load_config

BLOCK_CODES = {"safety", "prohibited_content", "blocklist", "image_safety", "content_blocked"}


class Blocked(Exception):
    pass


class DecisionModel:
    def __init__(self, spec, run):
        self.spec, self.device = spec, run["device"]
        self.cache_dir = os.path.join(run["cache_dir"], spec["name"])
        self.limit = (RateLimit(os.path.join(run["cache_dir"], f"{spec['name']}_usage.json"), **spec["rate_limit"])
                      if spec.get("rate_limit") else None)

    def answer(self, item):
        """Answer every question of a loaded item as {question_id: {"status", "probs"}}."""
        try:
            return self.parse(self.cached(item["request"]), item)
        except Blocked as e:
            return {q["id"]: {"status": "blocked", "error": str(e)} for q in item["questions"]}
        except Exception as e:
            return {q["id"]: {"status": "error", "error": f"{type(e).__name__}: {e}"} for q in item["questions"]}

    def cached(self, request):
        path = cache_path(self.cache_dir, self.spec["id"], request)
        if not os.path.exists(path):
            raw = self.limit.call(self.send, request) if self.limit else self.send(request)
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump(raw, f, default=float)
        with open(path) as f:
            return json.load(f)

    def post(self, url, request, headers=None):
        response = requests.post(url, json=request, headers=headers or {}, timeout=120)
        if response.status_code == 400 and str(response.json().get("error", {}).get("code")) in BLOCK_CODES:
            raise Blocked(response.json()["error"]["code"])
        response.raise_for_status()
        return response.json()

    def close(self):
        pass


class Typesafe(DecisionModel):
    url = "https://api.typesafe.ai/v1/systemone"

    def send(self, request):
        return self.post(self.url, request, {"Authorization": f"Bearer {os.environ[self.spec['key']]}"})

    def parse(self, raw, item):
        return {q["id"]: {"status": "ok", "probs": self.probs(q, raw["answers"][q["id"]])} for q in item["questions"]}

    @staticmethod
    def probs(q, answer):
        if q["type"] == "yesno":
            return {"yes": answer["noul"], "no": 1 - answer["noul"]}
        p = answer["probabilities"]
        return {o["name"]: p[str(i) if q["type"] == "score" else o["name"]] for i, o in enumerate(q["options"])}


class Google(Typesafe):
    url = "https://generativelanguage.googleapis.com/v1beta/decisions"

    def send(self, request):
        return self.post(self.url, request, {"x-goog-api-key": os.environ[self.spec["key"]]})

    def parse(self, raw, item):
        if raw.get("prompt_feedback", {}).get("block_reason"):
            raise Blocked(raw["prompt_feedback"]["block_reason"])
        return super().parse(raw, item)


class OpenAI(DecisionModel):
    url = "https://api.openai.com/v1/decisions"

    def send(self, request):
        return self.post(self.url, request, {"Authorization": f"Bearer {os.environ[self.spec['key']]}"})

    def parse(self, raw, item):
        return {q["id"]: self.answer_one(q, a) for q, a in zip(item["questions"], raw["answers"])}

    @staticmethod
    def answer_one(q, a):
        if a["type"] == "refusal":
            return {"status": "refused"}
        if a["type"] == "predicate":
            return {"status": "ok", "probs": {"yes": a["probability"], "no": 1 - a["probability"]}}
        names = [o["name"] for o in q["options"]]
        return {"status": "ok", "probs": {names[x["value"]] if q["type"] == "score" else x["value"]: x["probability"]
                                          for x in a["probabilities"]}}


class Kev(Typesafe):
    """Kev served by its own repo's local server, which speaks the Jev request format."""

    def __init__(self, spec, run):
        super().__init__(spec, run)
        root = spec["dir"]
        if not os.path.exists(root):
            subprocess.run(["git", "clone", "-q", spec["repo"], root], check=True)
        subprocess.run(["uv", "sync", "-q", "--extra", "serve"], cwd=root, check=True)
        env = {**os.environ, **{k: str(spec[v]) for k, v in {"KEV_TEMPERATURE": "temperature", "KEV_DTYPE": "dtype"}.items()
                                if spec.get(v) is not None}}
        base = f"http://127.0.0.1:{spec['port']}"
        self.url, self.log = f"{base}/v1/systemone", open(os.path.join(root, f"{spec['name']}.log"), "w")
        self.server = subprocess.Popen(["uv", "run", "--extra", "serve", "python", "-m", "kev.serve", "--run", spec["id"],
                                        "--port", str(spec["port"])], cwd=root, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.time() + spec["startup_timeout"]
        while not self.ready(f"{base}/v1/models"):
            if self.server.poll() is not None or time.time() > deadline:
                self.close()
                raise RuntimeError(f"Kev server for {spec['id']} did not start, see {self.log.name}")
            time.sleep(5)

    @staticmethod
    def ready(url):
        try:
            return requests.get(url, timeout=5).ok
        except requests.ConnectionError:
            return False

    def send(self, request):
        return self.post(self.url, request)

    def close(self):
        self.server.terminate()
        self.server.wait()
        self.log.close()


class Laya(Typesafe):
    def __init__(self, spec, run):
        super().__init__(spec, run)
        import laya
        self.agent = laya.load(spec["id"], device=self.device)

    def send(self, request):
        return {"answers": self.agent.predict(request["state"], request["questions"])["answers"]}

    def close(self):
        import torch
        del self.agent
        torch.cuda.empty_cache()


BACKENDS = {"typesafe": Typesafe, "google": Google, "openai": OpenAI, "kev": Kev, "laya": Laya}


def load_model(spec, run):
    """Load a model: start or load it when it is local, set up its client when it is an API."""
    return BACKENDS[spec["backend"]](spec, run)


def call_model(model, item):
    """Send one loaded benchmark item to the model and return its answer to every question."""
    return model.answer(item)


def test_model(model, benchmarks, attacks=None, config=None):
    """Load a model and run each benchmark clean and under the attacks that target it.
    Returns the metric rows and every answer."""
    config = config or load_config()
    spec, attacks = {"name": model, **config["models"][model]}, load_attacks(attacks, config)
    loaded = load_model(spec, config["run"])
    rows, answers = [], []
    try:
        for benchmark in benchmarks:
            items = apply_attacks(load_dataset(benchmark, **(config["benchmarks"].get(benchmark) or {})), benchmark, attacks)
            results = [(item, call_model(loaded, item)) for item in tqdm(load_benchmark(items, spec), desc=f"{model} {benchmark}")]
            rows += [{"model": model, "model_id": spec["id"], "benchmark": benchmark, **row} for row in score(results, benchmark)]
            answers += [{"model": model, "benchmark": benchmark, "id": item["id"], "attack": item["meta"]["attack"], "answers": answer}
                        for item, answer in results]
    finally:
        loaded.close()
    return rows, answers
