import json
from pathlib import Path
from typing import Any

import chromadb
import verifiers as vf
from chromadb.utils import embedding_functions
from datasets import Dataset
from verifiers.types import AssistantMessage, Messages, ToolCall, ToolMessage


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "data" / "synthetic_patent_queries.jsonl"
DEFAULT_CHROMA_DIR = BASE_DIR / ".chroma_db"
DEFAULT_COLLECTION = "patent_collection"


SYSTEM_PROMPT = """You are a patent prior-art search assistant.

Use the available tools to search the local patent database, inspect promising
patents, and return the publication numbers of the strongest prior-art matches.

The reward checks whether your final patent_ids include the gold patent. Always
call return_final_answer when you are done.
"""


def canonical_patent_id(value: str) -> str:
    return str(value).split("-", 1)[0].strip().upper()


def resolve_dataset_path(dataset_path: str | None) -> Path:
    if dataset_path:
        path = Path(dataset_path)
        return path if path.is_absolute() else BASE_DIR / path

    data_dirs = [
        BASE_DIR / "data",
        Path.cwd() / "data",
        Path.cwd() / "environments" / "prior_art_search" / "data",
    ]
    for data_dir in data_dirs:
        default = data_dir / "synthetic_patent_queries.jsonl"
        if default.exists():
            return default
        candidates = sorted(data_dir.glob("*synthetic_patent_queries.jsonl"))
        if candidates:
            return candidates[0]
    return DEFAULT_DATASET


def resolve_chroma_path(chroma_dir: str | None) -> Path:
    if chroma_dir:
        path = Path(chroma_dir)
        return path if path.is_absolute() else BASE_DIR / path

    candidates = [
        DEFAULT_CHROMA_DIR,
        Path.cwd() / ".chroma_db",
        Path.cwd() / "environments" / "prior_art_search" / ".chroma_db",
    ]
    for path in candidates:
        if path.exists():
            return path
    return DEFAULT_CHROMA_DIR


def load_rows(dataset_path: Path, max_examples: int) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing dataset {dataset_path}. Run generate_synthetic_queries.py first."
        )

    rows = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "prompt": [{"role": "user", "content": row["query"]}],
                    "answer": canonical_patent_id(row["publication_number"]),
                    "info": json.dumps(row, ensure_ascii=False),
                }
            )
            if max_examples > 0 and len(rows) >= max_examples:
                break
    return rows


def format_search_results(results: dict[str, Any]) -> str:
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not ids:
        return "No patents found."

    lines = []
    for patent_id, metadata, distance in zip(ids, metadatas, distances):
        lines.append(
            json.dumps(
                {
                    "publication_number": patent_id,
                    "title": metadata.get("title", ""),
                    "abstract": metadata.get("abstract", "")[:500],
                    "distance": distance,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def extract_final_patent_ids(completion: Messages) -> list[str]:
    for message in reversed(completion):
        if not isinstance(message, AssistantMessage):
            continue
        tool_calls = getattr(message, "tool_calls", None) or []
        for tool_call in tool_calls:
            if not isinstance(tool_call, ToolCall) or tool_call.name != "return_final_answer":
                continue
            try:
                args = json.loads(tool_call.arguments)
            except json.JSONDecodeError:
                return []
            return [canonical_patent_id(x) for x in args.get("patent_ids", [])]
    return []


class PriorArtSearchEnv(vf.ToolEnv):
    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        tool_messages = await super().env_response(messages, state, **kwargs)
        last_msg = messages[-1]
        tool_calls = getattr(last_msg, "tool_calls", None) or []

        if any(call.name == "return_final_answer" for call in tool_calls):
            state["final_env_response"] = [
                ToolMessage(
                    role="tool",
                    content="Final answer received.",
                    tool_call_id=tool_calls[-1].id,
                )
            ]
        return tool_messages


def load_environment(
    dataset_path: str | None = None,
    chroma_dir: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    max_examples: int = -1,
    max_turns: int = 6,
    **kwargs,
) -> vf.Environment:
    dataset_file = resolve_dataset_path(dataset_path)
    rows = load_rows(dataset_file, max_examples)
    dataset = Dataset.from_list(rows)

    chroma_path = resolve_chroma_path(chroma_dir)
    if not chroma_path.exists():
        raise FileNotFoundError(
            f"Missing Chroma DB at {chroma_path}. Run prepare.py before evaluation."
        )

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
    )

    def search_patents(query: str, n_results: int = 10) -> str:
        """Search the patent database for patents relevant to a novelty-search query."""
        n_results = max(1, min(int(n_results), 20))
        results = collection.query(query_texts=[query], n_results=n_results)
        return format_search_results(results)

    def lookup_patent(publication_number: str) -> str:
        """Look up a specific patent by publication number and return its indexed text."""
        result = collection.get(ids=[publication_number], include=["documents", "metadatas"])
        if not result.get("ids"):
            return f"No patent found for {publication_number}."
        metadata = result["metadatas"][0] or {}
        document = result["documents"][0] or ""
        return json.dumps(
            {
                "publication_number": result["ids"][0],
                "title": metadata.get("title", ""),
                "abstract": metadata.get("abstract", ""),
                "claims": metadata.get("claims", ""),
                "search_text": document[:6000],
            },
            ensure_ascii=False,
        )

    def return_final_answer(answer: str, patent_ids: list[str]) -> str:
        """Return the final prior-art patent IDs and a short explanation."""
        return json.dumps({"answer": answer, "patent_ids": patent_ids}, ensure_ascii=False)

    async def correct_patent_returned(completion: Messages, answer: str) -> float:
        predicted = extract_final_patent_ids(completion)
        return 1.0 if canonical_patent_id(answer) in predicted else 0.0

    async def returned_any_patent(completion: Messages) -> float:
        return 1.0 if extract_final_patent_ids(completion) else 0.0

    rubric = vf.Rubric(
        funcs=[correct_patent_returned, returned_any_patent],
        weights=[1.0, 0.0],
    )

    return PriorArtSearchEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        tools=[search_patents, lookup_patent, return_final_answer],
        rubric=rubric,
        max_turns=max_turns,
        env_id="prior-art-search",
        env_args={
            "dataset_path": str(dataset_file),
            "chroma_dir": str(chroma_path),
            "collection_name": collection_name,
            "max_examples": max_examples,
            "max_turns": max_turns,
        },
        **kwargs,
    )
