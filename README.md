# prior-art-search

### Overview
- **Environment ID**: `prior-art-search`
- **Short description**: Tool-use environment for prior-art patent search over a local Chroma database.
- **Tags**: patents, prior-art, search, tools, chroma

### Datasets
- **Primary dataset(s)**: HUPD patent records processed into a local Chroma collection.
- **Source links**: `HUPD/hupd` on Hugging Face.
- **Synthetic scenarios**: generated with `generate_synthetic_queries.py` and keyed by `publication_number`.
- **Split sizes**: depends on generation limits.

Generate scenario rows with the Prime inference endpoint:

```bash
uv run python environments/prior_art_search/generate_synthetic_queries.py \
  --model qwen/qwen3.6-35b-a3b \
  --limit 20
```

Outputs:

```text
environments/prior_art_search/data/synthetic_patent_queries.jsonl
```

### Dataset Caveat
The current synthetic dataset is a v1 retrieval task, not full real-world novelty
assessment.

For each source patent, the generator creates a disguised invention disclosure
and draft claim from that same patent. The gold answer is still the source
`publication_number`. So the task is:

```text
Given a paraphrased unpublished invention disclosure, retrieve the matching
patent from the vector database.
```

This is useful for training the first environment because it teaches tool-use
retrieval: search, inspect candidates, and return the matching patent ID.

Real novelty search is stricter. In production, a user usually has a new draft
invention and wants older prior-art patents that overlap with it. A stronger v2
dataset should use gold answers such as cited patents, examiner prior-art
references, or curated nearest older patents, rather than rewarding retrieval of
the same source patent used to create the scenario.

### Task
- **Type**: multi-turn tool use
- **Tools**: `search_patents`, `lookup_patent`, `return_final_answer`
- **Rubric overview**: binary reward for returning the gold `publication_number` in `return_final_answer(patent_ids=...)`.

### Quickstart
Prepare the Chroma database:

```bash
cd environments/prior_art_search
uv run python prepare.py --limit 500
```

Install the environment:

```bash
prime env install prior-art-search
```

Run a small evaluation:

```bash
prime eval run prior-art-search \
  -m qwen/qwen3.6-35b-a3b \
  -n 5 -r 1 \
```

Configure model and sampling:

```bash
prime eval run prior-art-search \
  -m qwen/qwen3.6-35b-a3b \
  -n 20 -r 3
```

Evaluate only hard scenarios:

```bash
prime eval run prior-art-search \
  -m qwen/qwen3.6-35b-a3b \
  -n 20 \
  -r 4 \
  -a '{"difficulty": "hard"}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_path` | str | `data/synthetic_patent_queries.jsonl` | Synthetic scenario JSONL keyed by `publication_number` |
| `chroma_dir` | str | `.chroma_db` | Local Chroma persistence directory |
| `collection_name` | str | `patent_collection` | Chroma collection to search |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |
| `max_turns` | int | `6` | Maximum tool-use turns per rollout |
| `difficulty` | str | `None` | Optional filter, e.g. `easy`, `medium`, or `hard` |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `correct_patent_returned` | `1.0` if final `patent_ids` include the gold publication number |
| `returned_any_patent` | Diagnostic metric for whether the model used the final-answer tool with IDs |
| `num_turns` | Number of model/tool turns |
| `*_calls` | Tool call counts from Verifiers' tool monitor rubric |
