import argparse
import json
import os
import tarfile
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from openai import OpenAI


DATASET = "HUPD/hupd"
HF_FILE = "data/sample-jan-2016.tar.gz"
OUT = Path(__file__).resolve().parent / "data" / "synthetic_patent_queries.jsonl"


def clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value)


def load_patents(limit: int) -> list[dict[str, str]]:
    archive = hf_hub_download(DATASET, filename=HF_FILE, repo_type="dataset")
    patents = []

    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if len(patents) >= limit:
                break
            if not member.isfile() or not member.name.endswith(".json"):
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            raw = json.load(f)
            patent = {
                "publication_number": clean(raw.get("publication_number")),
                "title": clean(raw.get("title")),
                "abstract": clean(raw.get("abstract")),
                "claims": clean(raw.get("claims")),
            }
            if patent["publication_number"] and patent["abstract"]:
                patents.append(patent)

    return patents


def make_prompt(patent: dict[str, str]) -> str:
    return f"""
Create synthetic novelty-search scenarios from this patent.

The scenario should look like an unpublished invention disclosure that an
inventor or patent attorney would use to search for prior art. Do not write
ordinary questions.

Return JSON only in this format:
{{
  "scenarios": [
    {{
      "difficulty": "easy|medium|hard",
      "invention_disclosure": "...",
      "draft_claim": "...",
      "key_features": ["...", "..."],
      "search_instruction": "..."
    }}
  ]
}}

Generate exactly 3 scenarios total: one easy, one medium, and one hard.
Keep each text field concise. Use at most 4 key features per scenario.

Difficulty:
- easy: close to the abstract, with direct technical wording.
- medium: paraphrased, combining several technical constraints.
- hard: claim-like and indirect, focused on mechanisms and edge cases.

Rules:
- Do not mention the publication number.
- Do not copy the exact title.
- Do not say this came from an existing patent.
- Preserve the core technical mechanism from the title, abstract, and claims.
- The scenario should help retrieve this patent from a vector database.

title: {patent["title"]}
abstract: {patent["abstract"]}
claims: {patent["claims"][:5000]}
"""


def parse_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.startswith("json"):
            content = content[4:].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def generate_queries(client: OpenAI, model: str, patent: dict[str, str]) -> list[dict]:
    last_error = None
    for attempt in range(3):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Return valid JSON only. No markdown. No commentary.",
                },
                {"role": "user", "content": make_prompt(patent)},
            ],
            temperature=0.4,
            max_tokens=10000,
            response_format={"type": "json_object"},
        )
        if not response.choices:
            raise RuntimeError(f"Prime returned no choices: {response.model_dump()}")

        content = response.choices[0].message.content or "{}"
        try:
            payload = parse_json(content)
            scenarios = payload.get("scenarios", [])
            if not isinstance(scenarios, list) or not scenarios:
                raise RuntimeError(f"Expected non-empty scenarios list, got: {payload}")
            return scenarios
        except json.JSONDecodeError as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
        except RuntimeError as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Could not parse JSON for {patent['publication_number']}: {last_error}")


def scenario_to_query(item: dict) -> str:
    features = item.get("key_features", [])
    if isinstance(features, list):
        features = "\n".join(f"- {feature}" for feature in features)

    return f"""Novelty search request:

Invention disclosure:
{item.get("invention_disclosure", "")}

Draft claim:
{item.get("draft_claim", "")}

Key features:
{features}

Search instruction:
{item.get("search_instruction", "Find prior-art patents that disclose the same core invention.")}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen/qwen3.6-35b-a3b")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("PRIME_API_KEY"):
        raise RuntimeError("Set PRIME_API_KEY first.")

    client = OpenAI(api_key=os.environ["PRIME_API_KEY"], base_url=os.environ["PRIME_URL"])
    patents = load_patents(args.limit)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, patent in enumerate(patents, start=1):
            scenarios = generate_queries(client, args.model, patent)
            for item in scenarios:
                row = {
                    "publication_number": patent["publication_number"],
                    "query": scenario_to_query(item),
                    "difficulty": item.get("difficulty", ""),
                    "invention_disclosure": item.get("invention_disclosure", ""),
                    "draft_claim": item.get("draft_claim", ""),
                    "key_features": item.get("key_features", []),
                    "search_instruction": item.get("search_instruction", ""),
                    "abstract": patent["abstract"],
                    "title": patent["title"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(patents)}] {patent['publication_number']}: {len(scenarios)} scenarios")

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
