from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class ResolvedModel:
    model_id: str
    tokenizer: str | None


def _key(value: str) -> str:
    return value.strip().lower()


def _repo_from_source_url(source_url: str | None) -> str | None:
    if not source_url:
        return None
    match = re.search(r"huggingface\.co/([^/]+/[^/]+)/", source_url)
    return match.group(1) if match else None


def _add_alias(aliases: dict[str, ResolvedModel], alias: str | None, resolved: ResolvedModel) -> None:
    if alias:
        aliases.setdefault(_key(alias), resolved)


def model_aliases(models_data: dict[str, Any]) -> dict[str, ResolvedModel]:
    aliases: dict[str, ResolvedModel] = {}
    for model in models_data.get("models", []):
        model_id = str(model["id"])
        label = str(model.get("label") or model_id)
        repo = _repo_from_source_url(model.get("source_url"))
        resolved = ResolvedModel(model_id=model_id, tokenizer=repo)

        _add_alias(aliases, model_id, resolved)
        _add_alias(aliases, label, resolved)
        _add_alias(aliases, repo, resolved)
        if repo:
            repo_name = repo.rsplit("/", 1)[-1]
            _add_alias(aliases, repo_name, resolved)
            if repo_name.endswith("-Instruct"):
                _add_alias(aliases, repo_name.removesuffix("-Instruct"), resolved)

    return aliases


def resolve_model_alias(model: str, models_data: dict[str, Any]) -> ResolvedModel:
    aliases = model_aliases(models_data)
    resolved = aliases.get(_key(model))
    if resolved:
        return resolved

    examples = ", ".join(sorted(list(aliases))[:5])
    raise ValueError(f"Unknown model: {model}. Known model examples: {examples}")
