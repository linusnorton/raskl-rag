"""JSONL I/O and filesystem utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_jsonl(path: Path | str, records: list[BaseModel]) -> None:
    """Write a list of Pydantic models to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")


def read_jsonl(path: Path | str, model_class: type[T]) -> list[T]:
    """Read a JSONL file into a list of Pydantic model instances."""
    path = Path(path)
    records: list[T] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(model_class.model_validate_json(line))
    return records


def ensure_dir(path: Path | str) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
