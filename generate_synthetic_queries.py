import argparse
import json
import os
import tarfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from openai import OpenAI


PRIME_URL = "https://api.pinference.ai/api/v1"
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


def make_prompt(patent: dict[str, str], n: int) -> str:
    return f"""
Generate synthetic prior-art search questions for this patent.

Return JSON only in this format:
{{"queries":[{{"difficulty":"easy|medium|hard","query":"..."}}]}}

Generate {n} queries for each difficulty: easy, medium, hard.

Rules:
- Do not mention the publication number.
- Do not copy the exact title.
- The query should help retrieve this patent from a vector database.

title: {patent["title"]}
abstract: {patent["abstract"]}
claims: {patent["claims"][:5000]}
"""


def generate_queries(client: OpenAI, model: str, patent: dict[str, str], n: int) -> list[dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": make_prompt(patent, n)},
        ],
        temperature=0.7,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    if not response.choices:
        raise RuntimeError(f"Prime returned no choices: {response.model_dump()}")

    content = response.choices[0].message.content or "{}"
    return json.loads(content).get("queries", [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen/qwen3.6-35b-a3b")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--queries-per-difficulty", type=int, default=2)
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("PRIME_API_KEY"):
        raise RuntimeError("Set PRIME_API_KEY first.")

    client = OpenAI(api_key=os.environ["PRIME_API_KEY"], base_url=PRIME_URL)
    patents = load_patents(args.limit)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, patent in enumerate(patents, start=1):
            queries = generate_queries(client, args.model, patent, args.queries_per_difficulty)
            for item in queries:
                row = {
                    "publication_number": patent["publication_number"],
                    "query": item.get("query", ""),
                    "difficulty": item.get("difficulty", ""),
                    "abstract": patent["abstract"],
                    "title": patent["title"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(patents)}] {patent['publication_number']}: {len(queries)} queries")

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
