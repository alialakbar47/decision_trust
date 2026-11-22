# DecisionTrust v2

Trust evaluation of typed decision models. `config.yml` picks the models, benchmarks and attacks of a run and holds their hyperparameters.

| File | Role |
|---|---|
| `config.yml` | models (backend, id, sizes, keys, limits), benchmark sampling, attack hyperparameters and targets |
| `models.py` | `load_model(spec, run)`, `ask(model, item)`, `test_model(spec, run)`; `model_specs(config)` expands sizes |
| `benchmarks.py` | `load_benchmark(name, max_examples, per_subset, where, seed)`, `load_benchmarks(config)` |
| `attacks.py` | FlipAttack, JAIL-CON (light), DeepInception; `load_attacks(config)`, `attack_cases(item, benchmark, attacks)` |
| `metrics.py` | summary, per-benchmark metrics, attack success; `score(results, benchmark)` |
| `main.py` | `run(config, specs=None)` |
| `data/<benchmark>.jsonl` | full converted benchmarks, built by the data notebook |

Every line of a data file is one item:

```json
{"id": "<group_id>-<variant>", "group_id": "<benchmark>-<source id>", "variant": "clean", "input": "<text the model reads>",
 "questions": [{"id": "<question id>", "type": "yesno | choice | score", "text": "...", "options": [{"name": "...", "description": "..."}], "gold": "..."}],
 "meta": {"subset": "...", "...": "..."}}
```

Items that belong together (an AdvGLUE++ original and its adversarial rewrite, an XSTest prompt and its attacked copies) share a `group_id`, and sampling keeps groups whole. Results have one row per model, benchmark, attack (`none` for the clean items) and subset.
