# prior-art-search

### Overview
- **Environment ID**: `prior-art-search`
- **Short description**: <one-sentence description>
- **Tags**: <comma-separated tags>

### Datasets
- **Primary dataset(s)**: HUPD patent records processed into a local Chroma collection.
- **Source links**: `HUPD/hupd` on Hugging Face.
- **Synthetic scenarios**: generated with `generate_synthetic_queries.py` and keyed by `publication_number`.
- **Split sizes**: depends on generation limits.

Generate scenario rows with the Prime inference endpoint:

```bash
uv run python environments/prior_art_search/generate_synthetic_queries.py \
  --model qwen/qwen3.6-35b-a3b \
  --limit 20 \
  --queries-per-difficulty 2
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
- **Type**: <single-turn | multi-turn | tool use>
- **Output format expectations (optional)**: <e.g., plain text, XML tags, JSON schema>
- **Rubric overview**: <briefly list reward functions and key metrics>

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run prior-art-search
```

Configure model and sampling:

```bash
prime eval run prior-art-search   -m openai/gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_path` | str | `data/synthetic_patent_queries.jsonl` | Synthetic query rows keyed by `publication_number` |
| `chroma_dir` | str | `.chroma_db` | Local Chroma persistence directory |
| `collection_name` | str | `patent_collection` | Chroma collection to search |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy` | Exact match on target answer |
