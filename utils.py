import os
import json
import time
import hashlib
from collections import deque

import yaml
import requests


def load_config(path="config.yml"):
    with open(path) as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False, default=float) + "\n" for row in rows)


def write_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=float)


def matches(item, where):
    """True when every meta field named in `where` has one of the listed values."""
    return all(item["meta"].get(key) in values for key, values in (where or {}).items())


def cache_path(cache_dir, *key):
    return os.path.join(cache_dir, hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest() + ".json")


class RateLimit:
    """Requests and tokens per minute plus a daily cap kept on disk across runs, with retries on HTTP 429.
    The day follows Pacific time, when the Google quota resets."""

    def __init__(self, path, per_minute=None, tokens_per_minute=None, per_day=None):
        self.path, self.per_minute, self.tokens_per_minute, self.per_day = path, per_minute, tokens_per_minute, per_day
        self.window = deque()

    def used(self):
        if not os.path.exists(self.path):
            return {}
        with open(self.path) as f:
            return json.load(f)

    def wait(self, tokens):
        day, used = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 8 * 3600)), self.used()
        if self.per_day and used.get(day, 0) >= self.per_day:
            raise RuntimeError(f"daily cap of {self.per_day} requests reached; rerun tomorrow, cached answers are kept")
        while True:
            now = time.time()
            while self.window and now - self.window[0][0] > 60:
                self.window.popleft()
            if ((not self.per_minute or len(self.window) < self.per_minute) and
                    (not self.tokens_per_minute or sum(t for _, t in self.window) + tokens <= self.tokens_per_minute)):
                break
            time.sleep(1)
        self.window.append((now, tokens))
        write_json(self.path, {**used, day: used.get(day, 0) + 1})

    def call(self, send, request):
        for attempt in range(6):
            self.wait(len(json.dumps(request)) // 3)
            try:
                return send(request)
            except requests.HTTPError as e:
                if e.response.status_code != 429 or attempt == 5:
                    raise
                time.sleep(20 * (attempt + 1))
