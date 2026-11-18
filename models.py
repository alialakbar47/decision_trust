import os
import json
import hashlib

import requests
import torch

from datasets import options

BLOCK_CODES = {"safety", "prohibited_content", "blocklist", "image_safety", "content_blocked"}


class Blocked(Exception):
    pass


class DecisionModel:
    def __init__(self, model_id, cache_dir="cache", device="cuda"):
        self.model_id, self.cache_dir, self.device = model_id, cache_dir, device

    def ask(self, item, request):
        """Answer every question of an item as {question_id: {"status", "probs"}}."""
        try:
            return self.parse(self.cached(request), item)
        except Blocked as e:
            return {q["id"]: {"status": "blocked", "error": str(e)} for q in item["questions"]}
        except Exception as e:
            return {q["id"]: {"status": "error", "error": f"{type(e).__name__}: {e}"} for q in item["questions"]}

    def cached(self, request):
        key = hashlib.md5(json.dumps([self.model_id, request], sort_keys=True).encode()).hexdigest()
        path = os.path.join(self.cache_dir, f"{key}.json")
        if not os.path.exists(path):
            raw = self.send(request)
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump(raw, f, default=float)
        with open(path) as f:
            return json.load(f)

    def post(self, url, request, headers):
        response = requests.post(url, json=request, headers=headers, timeout=120)
        if response.status_code == 400 and str(response.json().get("error", {}).get("code")) in BLOCK_CODES:
            raise Blocked(response.json()["error"]["code"])
        response.raise_for_status()
        return response.json()


class Jev(DecisionModel):
    url = "https://api.typesafe.ai/v1/systemone"

    def render(self, item):
        return {"model": self.model_id, "state": item["input"],
                "questions": {q["id"]: self.question(q) for q in item["questions"]}}

    @staticmethod
    def question(q):
        if q["type"] == "yesno":
            return {"type": "noul", "instructions": q["text"]}
        criteria = {o["name"]: o.get("description", o["name"]) for o in q["options"]}
        return {"type": q["type"], "instructions": q["text"],
                "criteria": list(criteria.values()) if q["type"] == "score" else criteria}

    def send(self, request):
        return self.post(self.url, request, {"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"})

    def parse(self, raw, item):
        return {q["id"]: {"status": "ok", "probs": self.probs(q, raw["answers"][q["id"]])} for q in item["questions"]}

    @staticmethod
    def probs(q, answer):
        if q["type"] == "yesno":
            return {"yes": answer["noul"], "no": 1 - answer["noul"]}
        p = answer["probabilities"]
        return {o["name"]: p[str(i) if q["type"] == "score" else o["name"]] for i, o in enumerate(q["options"])}


class Google(Jev):
    url = "https://generativelanguage.googleapis.com/v1beta/decisions"

    def render(self, item):
        request = super().render(item)
        request["input"] = request.pop("state")
        return request

    def send(self, request):
        return self.post(self.url, request, {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})

    def parse(self, raw, item):
        if raw.get("prompt_feedback", {}).get("block_reason"):
            raise Blocked(raw["prompt_feedback"]["block_reason"])
        return super().parse(raw, item)


class OpenAI(DecisionModel):
    url = "https://api.openai.com/v1/decisions"

    def render(self, item):
        return {"model": self.model_id, "input": item.get("turns", item["input"]),
                "questions": [self.question(q) for q in item["questions"]]}

    @staticmethod
    def question(q):
        request = {"type": "predicate" if q["type"] == "yesno" else q["type"], "name": q["id"], "instructions": q["text"]}
        if q["type"] == "choice":
            request["choices"] = [{"value": o["name"], "description": o.get("description", o["name"])} for o in q["options"]]
        if q["type"] == "score":
            request["levels"] = [{"label": o["name"], "description": o.get("description", o["name"])} for o in q["options"]]
        return request

    def send(self, request):
        return self.post(self.url, request, {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})

    def parse(self, raw, item):
        return {q["id"]: self.answer(q, a) for q, a in zip(item["questions"], raw["answers"])}

    @staticmethod
    def answer(q, a):
        if a["type"] == "refusal":
            return {"status": "refused"}
        if a["type"] == "predicate":
            return {"status": "ok", "probs": {"yes": a["probability"], "no": 1 - a["probability"]}}
        names = [o["name"] for o in q["options"]]
        return {"status": "ok", "probs": {names[x["value"]] if q["type"] == "score" else x["value"]: x["probability"]
                                          for x in a["probabilities"]}}


class Laya(Jev):
    def __init__(self, model_id, cache_dir="cache", device="cuda"):
        super().__init__(model_id, cache_dir, device)
        import laya
        self.agent = laya.load(model_id, device=device)

    def send(self, request):
        return {"answers": self.agent.predict(request["state"], request["questions"])["answers"]}


class LLM(DecisionModel):
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, model_id, cache_dir="cache", device="cuda"):
        super().__init__(model_id, cache_dir, device)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        self.letters = [self.tokenizer.encode(letter, add_special_tokens=False)[0] for letter in self.LETTERS]

    def render(self, item):
        return {q["id"]: self.prompt(item["input"], q) for q in item["questions"]}

    def prompt(self, text, q):
        lines = [f"{letter}. {o['name']}" + (f": {o['description']}" if "description" in o else "")
                 for letter, o in zip(self.LETTERS, options(q))]
        content = f"Input:\n{text}\n\nQuestion: {q['text']}\n" + "\n".join(lines) + "\nAnswer with the letter only."
        return self.tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def send(self, request):
        logits = {}
        for question_id, prompt in request.items():
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
            logits[question_id] = self.model(**inputs).logits[0, -1, self.letters].float().tolist()
        return logits

    def parse(self, raw, item):
        return {q["id"]: {"status": "ok", "probs": self.probs(q, raw[q["id"]])} for q in item["questions"]}

    @staticmethod
    def probs(q, logits):
        names = [o["name"] for o in options(q)]
        return dict(zip(names, torch.softmax(torch.tensor(logits[:len(names)]), -1).tolist()))


MODELS = {"jev": Jev, "openai": OpenAI, "google": Google, "laya": Laya, "llm": LLM}


def get_model(spec, cache_dir="cache", device="cuda"):
    return MODELS[spec["name"]](spec["id"], os.path.join(cache_dir, spec["name"]), device)
