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
